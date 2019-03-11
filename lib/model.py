import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import warnings

MODELS = {}

class MLP(nn.Module):

    def __init__(self, neuron_sizes, activation=nn.LeakyReLU, bias=True): 
        super(MLP, self).__init__()
        self.neuron_sizes = neuron_sizes
        
        layers = []
        for s0, s1 in zip(neuron_sizes[:-1], neuron_sizes[1:]):
            layers.extend([
                nn.Linear(s0, s1, bias=bias),
                activation()
            ])
        
        self.classifier = nn.Sequential(*layers[:-1])

    def eval_forward(self, x, y):
        self.eval()
        return self.forward(x)
        
    def forward(self, x):
        x = x.view(-1, self.neuron_sizes[0])
        return self.classifier(x)

######################## not optimized start #######################################
class BaseModelLSTM(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 num_directions=1):
        super(BaseModelLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional = (num_directions == 2)
        self.model = torch.nn.LSTM(self.input_size, self.hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=self.bidirectional)
        self.h2o = nn.Linear(hidden_size * num_directions, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # x.shape: (seq_len, bs, _)
        o, hidden = self.model(x, hidden)

        if type(x) is torch.nn.utils.rnn.PackedSequence:
            # undo packing: second arg is input_lengths (we do not need)
            # o: seq_len x bs x d            
            o, _ = torch.nn.utils.rnn.pad_packed_sequence(o)
            
        seq_len, bs, d  = o.shape
        
        # dim transformation: (seq_len, bs, d) -> (seq_len x bs, d)
        o = o.contiguous()
        o = o.view(-1, o.shape[2])

        # run through prediction layer: 
        o = self.h2o(o)
        o = self.softmax(o)

        # dim transformation
        o = o.view(seq_len, bs, self.output_size)

        return o, hidden
    
class BaseModelMLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(BaseModelMLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.model = MLP([input_size, hidden_size])
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden): # hidden is dummy
        # x.shape: (seq_len, bs, d)
        seq_len, bs, d  = x.shape
        o = self.model(x.view(seq_len * bs, d))

        # run through prediction layer
        o = self.h2o(o)
        o = self.softmax(o)

        # dim transformation
        o = o.view(seq_len, bs, self.output_size)

        return o, hidden

###################### not optimized end #############################################

# reference: https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
# current implementation assumes only one output at the end
# to do a lot of predictions, I need to a) pad target and b) mask out unwanted loss terms
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size,
                 num_layers=1, num_directions=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        assert num_directions == 1, "bidirection haven't impelmented yet"

        self.custom_init()

    def custom_init(self):
        raise NotImplementedError()

    def get_model(self, t, hidden):
        raise NotImplementedError()
    
    def base_model(self):
        # give the base model to use
        raise NotImplementedError()
    
    def eval_forward(self, x, y, x_lengths):
        # this function is for evaluation        
        self.eval()
        seq_len, bs, _ = x.shape
        hidden = self.initHidden(batch_size=bs)
        output, hidden = self.forward(x, hidden, x_lengths)
        return output

    def initHidden(self, batch_size, use_gpu=False):
        # for both cell memory and hidden neurons
        h, c = (torch.zeros(self.num_layers * self.num_directions,
                            batch_size, self.hidden_size),
                torch.zeros(self.num_layers * self.num_directions,
                            batch_size, self.hidden_size))

        cuda_check = next(self.parameters())
        if cuda_check.is_cuda:
            h = h.cuda()
            c = c.cuda()

        return (h, c) # assume base model is lstm

    def after_backward(self):
        # code to run after backward function, before optimizer
        return

    def forward(self, x, hidden, input_lengths):
        # todo: deal with bidirectional here
        seq_len, bs, _ = x.shape

        outputs = []
        for t in range(seq_len):
            # get a new model for each time step            
            model = self.get_model(t, hidden) # use hidden to index
            o, hidden = model(x[t].view(1, bs, -1), hidden)
            outputs.append(o)
        o = torch.cat(outputs, 0) # (seq_len, bs, _)

        return o, hidden
    
#######################################################################################
class RNN_Memory(RNN):

    def base_model(self):
        return BaseModelLSTM(self.input_size, self.hidden_size, self.output_size,
                             self.num_layers, self.num_directions)    
    
class RNN_Memoryless(RNN):
    '''mlp sharing the same interface with RNN, basically 
    ignore time dependence and only use last step information for prediction'''

    def base_model(self):
        return BaseModelMLP(self.input_size, self.hidden_size, self.output_size,
                            self.num_layers)

class RNN_Independent(RNN): # sharing strategy
    ''' independent: each time step assign a different model '''
    def custom_init(self):
        self.models = torch.nn.ModuleList() # modulelist makes module params visible
        self.set_max_length(3)

    def set_max_length(self, t):
        while t >= len(self.models):
            self.models.append(self.base_model())
        
    def get_model(self, t, hidden):
        return self.models[t]

class RNN_Staged(RNN): # sharing strategy
    ''' staged: manully share time steps '''
    def custom_init(self):
        self.groups = []
        self.models = torch.nn.ModuleList()
        self.set_shared_groups([])

    def set_shared_groups(self, groups):
        # each group share one model
        # assumes groups has no member overlap
        # groups is a list of list
        self.groups = groups
        while len(self.models) <= len(self.groups):
            self.models.append(self.base_model())
    
    def get_model(self, t, hidden):
        for i, g in enumerate(self.groups):
            if t in g:
                return self.models[i]
        return self.models[len(self.groups)]

class RNN_MoW(RNN): # sharing strategy
    '''mixture of weights for each time step with total of k clusters'''
    def custom_init(self):
        self.models = torch.nn.ModuleList()
        self.meta_models = torch.nn.ModuleList()
        self.setKT(1, 3)

    def setKT(self, k, t): # k * t weights
        '''k clusters with maximum of t time steps'''
        self.k = k
        self.T = t

        while len(self.models) < t:
            self.models.append(self.base_model())

        while len(self.meta_models) < k:
            self.meta_models.append(self.base_model())            

        self.coef = torch.nn.Parameter(torch.zeros(t, k)) # each row should add to 1
        nn.init.uniform_(self.coef)

    def _combine_weights(self, params, t):
        coef = nn.functional.softmax(self.coef, dim=1) # wasteful computation: todo
        return sum([c*p for c, p in zip(coef[t], params)])
    
    def get_model(self, t, hidden):
        # model has parameters that are not variables, so we have to manully
        # backprop model parameters to their sub module
        model = self.models[t] # the main model
        params_list = list(zip(*[m.parameters() for m in self.meta_models]))
        for (name, p), params in zip(model.named_parameters(), params_list):
            p.data = self._combine_weights(params, t)
        return model

    def after_backward(self):
        # manully back prop into meta networks
        # need list instead of zip alone because zip only traverse once
        params_list = list(zip(*[m.parameters() for m in self.meta_models]))

        for t, model in enumerate(self.models):
            for (name, p), params in zip(model.named_parameters(), params_list):
                if p.grad is None: # sometimes t are large so that not updated yet
                    continue
                # wasteful computation: todo: save this in get_model
                c = self._combine_weights(params, t)
                c.backward(p.grad)

class RNN_MoW_w(RNN): # sharing strategy
    '''mixture of weights for each time step with total of k clusters'''
    def custom_init(self):
        self.models = torch.nn.ModuleList()
        self.meta_models = torch.nn.ModuleList()
        self.setKT(1, 3, 32)

    def setKT(self, k, t, bs): # k * t weights
        '''k clusters with maximum of t time steps'''
        self.k = k
        self.T = t

        while len(self.models) < t * bs: # 32 x t models: too much space requirement
            self.models.append(self.base_model())

        while len(self.meta_models) < k:
            self.meta_models.append(self.base_model())            

        self.coef = torch.nn.Parameter(torch.zeros(t, k)) # each row should add to 1
        nn.init.uniform_(self.coef)

    def forward(self, x, hidden, input_lengths):    
        seq_len, bs, _ = x.shape

        outputs = []
        next_h = []
        next_c = []
        for t in range(seq_len):
            outputs_ = []
            for i in range(bs):
                # get a new model for each time step
                model = self.get_model(t, i, hidden) # use hidden to index
                # hidden: (n_layers, bs, hidden_size),  (n_layers, bs, hidden_size)
                o, (h, c) = model(x[t][i].view(1, 1, -1), hidden)
                outputs_.append(o)
                next_h.append(h)
                next_c.append(c)
            outputs.append(torch.cat(outputs_, 1)) # (1, bs, _)
            hidden = torch.cat(next_h, 1), torch.cat(next_c, 1)
            next_h, next_c = [], []
        o = torch.cat(outputs, 0) # (seq_len, bs, _)

        return o, hidden
        
    def _combine_weights(self, params, t):
        coef = nn.functional.softmax(self.coef, dim=1) # wasteful computation: todo
        return sum([c*p for c, p in zip(coef[t], params)])
    
    def get_model(self, t, batch_index, hidden):
        # need 128 models: too much, just do a for loop todo: save intermediate comput
        # ation
        
        # model has parameters that are not variables, so we have to manully
        # backprop model parameters to their sub modules
        model = self.models[t] # the main model 
        params_list = list(zip(*[m.parameters() for m in self.meta_models]))
        for (name, p), params in zip(model.named_parameters(), params_list):
            p.data = self._combine_weights(params, t)
        return model

    def after_backward(self):
        # manully back prop into meta networks
        # need list instead of zip alone because zip only traverse once
        params_list = list(zip(*[m.parameters() for m in self.meta_models]))

        for t, model in enumerate(self.models):
            for (name, p), params in zip(model.named_parameters(), params_list):
                if p.grad is None: # sometimes t are large so that not updated yet
                    continue
                # wasteful computation: todo: save this in get_model
                c = self._combine_weights(params, t)
                c.backward(p.grad)
                
class RNN_MoO_time(RNN): # mixture of output
    '''mixture of output for each time step with total of k clusters'''
    def custom_init(self):
        self.models = torch.nn.ModuleList()
        self.setKT(1, 3)

    def setKT(self, k, t): # k * t weights
        '''k clusters with maximum of t time steps'''
        self.k = k
        self.T = t

        while len(self.models) < k:
            self.models.append(self.base_model())

        # each row is for a time step
        self.coef = torch.nn.Parameter(torch.zeros(t, k)) 
        nn.init.uniform_(self.coef)

    def forward(self, x, hidden, input_lengths):    
        seq_len, bs, _ = x.shape

        outputs = []
        for t in range(seq_len):
            combine_weights = nn.functional.softmax(self.coef[t], dim=0)
            # combine output of all models
            o_s = []
            h_s = [] # short term memory
            c_s = [] # long term memory
            for model in self.models:
                o, (h, c) = model(x[t].view(1, bs, -1), hidden)
                o_s.append(o)
                h_s.append(h)
                c_s.append(c)

            # [seq_len, bs, _]
            o_ = sum(c * out for c, out in zip(combine_weights, o_s))
            h = sum(c * h for c, h in zip(combine_weights, h_s))
            cell = sum(c * cell for c, cell in zip(combine_weights, c_s))

            hidden = (h, cell)
            outputs.append(o_)
            
        o = torch.cat(outputs, 0) # (seq_len, bs, _)

        return o, hidden
        
class RNN_MoO(RNN): # mixture of output
    '''mixture of output for each time step with total of k clusters'''
    def custom_init(self):
        self.models = torch.nn.ModuleList()
        self.setKT(1, 3)

    def setKT(self, k, t): # k * t weights
        '''k clusters with maximum of t time steps'''
        self.k = k
        self.T = t

        while len(self.models) < k:
            self.models.append(self.base_model())

        # each row of coef index into a model
        self.coef = torch.nn.Parameter(torch.zeros(self.hidden_size, k)) 
        nn.init.uniform_(self.coef)

    def forward(self, x, hidden, input_lengths):    
        seq_len, bs, _ = x.shape

        self.last_combine_weights = [] # book keep coeficient used at each time step
        outputs = []
        for t in range(seq_len):
            # combine output of all models
            combine_weights = self.get_combine_weights(hidden) # bs, k
            self.last_combine_weights.append(combine_weights)
            
            o_s = []
            h_s = []
            c_s = []
            for model in self.models:
                o, (h, c) = model(x[t].view(1, bs, -1), hidden)
                o_s.append(o)
                h_s.append(h.unsqueeze(0))
                c_s.append(c.unsqueeze(0))                

            _, num_layers, bs, hidden_size = h_s[0].shape
            # reshape h_s from [k, n_layers, bs, hidden_size]
            # to [bs, n_layers * hidden_size, k]
            # reshape combine_weights to be [bs, k, 1]
            h_s = torch.cat(h_s, 0).permute(2,1,3,0).contiguous().view(bs,-1,self.k)
            h = torch.bmm(h_s, combine_weights.unsqueeze(2)).squeeze(2).view(
                bs, num_layers, hidden_size
            )
            h = h.permute(1, 0, 2).contiguous()

            # do the same for cell
            c_s = torch.cat(c_s, 0).permute(2,1,3,0).contiguous().view(bs,-1,self.k)
            cell = torch.bmm(c_s, combine_weights.unsqueeze(2)).squeeze(2).view(
                bs, num_layers, hidden_size
            )
            cell = cell.permute(1, 0, 2).contiguous()

            hidden = (h, cell)

            # fast version
            # proper way
            # reshape o_s to be [bs, seq_len * _, k], seqlen=1 in our case
            # reshpae combine_weights to be [bs, k, 1]
            # o_ = torch.bmm(o_s, combine_weights) # output [bs, _, 1]
            # reshape o_ to be [1, bs, _]
            o_s = torch.cat(o_s, 0).permute(1, 2, 0) # from [k, bs, _] to [bs, _, k]
            o_ = torch.bmm(o_s, combine_weights.unsqueeze(2)).squeeze(2)
            o_ = o_.unsqueeze(0)
            
            ## slow version:
            # (bs, k), (k, seq_len, bs, _)            
            # o_ = torch.zeros_like(o) # (seq_len, bs, _)
            # for i in range(bs):
            #     o_[:, i, :] = sum([c*out for c, out in zip(
            #         combine_weights[i], [out[:, i, :] for out in o_s])])
            
            outputs.append(o_)
        o = torch.cat(outputs, 0) # (seq_len, bs, _)

        return o, hidden
        
    def get_combine_weights(self, hidden):
        h, c = hidden # num_layers * num_directions, batchsize, hiddensize
        # use last layer's hidden output as indexer: batchsize, hiddensize
        self.last_h = h.view(self.num_layers, self.num_directions,
                             -1, self.hidden_size)[-1].view(-1, self.hidden_size)
        combine_weights = torch.mm(self.last_h, self.coef) # batchsize, k
        combine_weights = nn.functional.softmax(combine_weights, dim=1)
        return combine_weights
    
########################################################################################
class RNN_LSTM_2layers(RNN_Memory):

    def custom_init(self):
        self.num_layers = self.num_layers * 2
        self.model = BaseModelLSTM(self.input_size,
                                   self.hidden_size,
                                   self.output_size,
                                   self.num_layers,
                                   self.num_directions)

    def forward(self, x, hidden, input_lengths):
        # change the padded data with variable length: save computation
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        o, hidden = self.model(x, hidden) # hidden is (h, c)

        return o, hidden

class RNN_LSTM(RNN_Memory):

    def custom_init(self):
        self.model = self.base_model()

    def forward(self, x, hidden, input_lengths):
        # change the padded data with variable length: save computation
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        o, hidden = self.model(x, hidden) # hidden is (h, c)

        return o, hidden

class RNN_ILSTM(RNN_Memory, RNN_Independent):
    def describe(self):
        return '''independent lstm: each time step assign a different model'''
        
class RNN_SLSTM(RNN_Memory, RNN_Staged):
    def describe(self):
        return '''staged LSTM, manully share time steps'''
        
class RNN_LSTM_MoW(RNN_Memory, RNN_MoW):
    def describe(self):
        return '''time based 
        mixture of weights for each time step with total of k clusters'''

class RNN_LSTM_MoO(RNN_Memory, RNN_MoO):
    def describe(self):
        return '''hidden vector based 
        mixture of output for each time step with total of k clusters'''

class RNN_LSTM_MoO_time(RNN_Memory, RNN_MoO_time):
    def describe(self):
        return '''time based mixture of output for each time 
        step with total of k clusters'''
    
###################################################################################### 
class RNN_MLP(RNN_Memoryless):
    '''logistic regression sharing the same interface with RNN, basically 
    ignore time dependence and only use last step information for prediction'''

    def custom_init(self):
        self.model = self.base_model()

    def get_model(self, t, hidden):
        return self.model
    
class RNN_IMLP(RNN_Memoryless, RNN_Independent):
    def describe(self):
        return '''independent MLP sharing the same interface with RNN, 
        basically ignore time dependence and only use last step information for 
        prediction'''
    
class RNN_SMLP(RNN_Memoryless, RNN_Staged):
    def describe(self):
        return '''staged logistic regression sharing the same interface with RNN, 
        basically ignore time dependence and only use last step information for 
        prediction'''

class RNN_MLP_MoW(RNN_Memoryless, RNN_MoW):
    def describe(self):
        return '''mixture of weights for each time step with total of k clusters'''
    
MODELS = {
    'RNN_LSTM': RNN_LSTM,
    'RNN_LSTM_2layers': RNN_LSTM_2layers,
    'RNN_LSTM_MoW': RNN_LSTM_MoW, # this should be called time as well
    'RNN_LSTM_MoO': RNN_LSTM_MoO,
    'RNN_LSTM_MoO_time': RNN_LSTM_MoO_time,    
    'RNN_SLSTM': RNN_SLSTM,
    'RNN_ILSTM': RNN_ILSTM,
    'RNN_MLP': RNN_MLP,
    'RNN_IMLP': RNN_IMLP,
    'RNN_MLP_MoW': RNN_MLP_MoW,
    'RNN_SMLP': RNN_SMLP,
}
