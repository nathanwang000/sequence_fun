import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import warnings

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
        return self.forward(x)
        
    def forward(self, x):
        x = x.view(-1, self.neuron_sizes[0])
        return self.classifier(x)


# reference: https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
# current implementation assumes only one output at the end
# to do a lot of predictions, I need to a) pad target and b) mask out unwanted loss terms
class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.input_size = input_size
        self.output_size = output_size

        self.custom_init()

    def custom_init(self):
        raise NotImplementedError()

    def get_model(self, t):
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

    def initHidden(self, batch_size):
        # for both cell memory and hidden neurons
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, x, hidden, input_lengths):
        raise NotImplementedError()

#######################################################################################
class RNN_Memory(RNN):

    def base_model(self):
        return torch.nn.LSTM(self.input_size, self.hidden_size)

    def forward(self, x, hidden, input_lengths):    
        seq_len, bs, _ = x.shape

        ###### only part different from RNN_LSTM forward function ######
        outputs = []
        for t in range(seq_len):
            # get a new model for each time step            
            model = self.get_model(t)
            o, hidden = model(x[t].view(1, bs, -1), hidden)
            outputs.append(o)
        o = torch.cat(outputs, 0) # (seq_len, bs, _)

        # dim transformation: (seq_len, bs, d) -> (seq_len x bs, d)
        o = o.contiguous()
        o = o.view(-1, o.shape[2])
        
        # run through prediction layer
        o = self.h2o(o)
        o = self.softmax(o)

        # dim transformation
        o = o.view(seq_len, bs, self.output_size)
        
        return o, hidden
    
    def forward_old(self, x, hidden, input_lengths):
        warnings.warn('deprecated', DeprecationWarning)        
        # input of size seq_len x bs x d, padded sequences with input_lengths specified
        # this model assumes one output at the end
        seq_len, bs, d = x.shape
        
        outputs = []
        for t in range(seq_len):
            # get a new lstm for each time step            
            model = self.get_model(t)
            o, hidden = model(x[t].view(1, bs, d), hidden)
            outputs.append(o)
        o = torch.cat(outputs, 0)
        h, c = hidden
        
        # mask o by input_length, seq_len x bs x d
        h_t = o[list(map(lambda l: l-1, input_lengths)), range(bs)]
        
        # only use hidden state, minibatch input: h_t of size bs x d
        output = self.h2o(h_t)
        output = self.softmax(output)
        return output, hidden

class RNN_Memoryless(RNN):
    '''mlp sharing the same interface with RNN, basically 
    ignore time dependence and only use last step information for prediction'''

    def base_model(self):
        return MLP([self.input_size, self.hidden_size])

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape

        ###### only part different from RNN_LSTM forward function ######
        outputs = []
        for t in range(seq_len):
            # get a new model for each time step            
            model = self.get_model(t)
            o  = model(x[t]).unsqueeze(0) # seq_len x bs x output_size
            outputs.append(o)
        o = torch.cat(outputs, 0) # (seq_len, bs, _)

        # dim transformation: (seq_len, bs, d) -> (seq_len x bs, d)
        o = o.contiguous()
        o = o.view(-1, o.shape[2])

        # run through prediction layer
        o = self.h2o(o)
        o = self.softmax(o)

        # dim transformation
        o = o.view(seq_len, bs, self.output_size)
        
        return o, hidden # hidden is dummy here
    
    def forward_old(self, x, hidden, input_lengths):
        warnings.warn('deprecated', DeprecationWarning)                
        # input of size seq_len x bs x d, padded sequences with input_lengths specified
        # this model assumes one output at the end
        seq_len, bs, d = x.shape

        outputs = []
        for i, l in enumerate(input_lengths):
            o = self.get_model(l-1)(x[l-1, i].view(1, d))
            outputs.append(o)
        o = torch.cat(outputs, 0)
        o = self.softmax(o)

        return o, hidden # hidden is dummy here

class RNN_Independent(RNN): # sharing strategy
    ''' independent: each time step assign a different model '''
    def custom_init(self):
        self.models = torch.nn.ModuleList() # modulelist makes module params visible
        self.set_max_length(3)

    def set_max_length(self, t):
        while t >= len(self.models):
            self.models.append(self.base_model())
        
    def get_model(self, t):
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
        self.groups = groups
        while len(self.models) <= len(self.groups):
            self.models.append(self.base_model())
    
    def get_model(self, t):
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
    
    def get_model(self, t):
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
                # wasteful computation: todo: save this in get_model
                c = self._combine_weights(params, t)
                c.backward(p.grad)
    
########################################################################################
class RNN_LSTM(RNN_Memory):

    def custom_init(self):
        self.model = self.base_model()

    def forward(self, x, hidden, input_lengths):
        seq_len, bs, _ = x.shape
        
        # change the padded data with variable length: save computation
        x = torch.nn.utils.rnn.pack_padded_sequence(x, input_lengths)
        
        o, hidden = self.model(x, hidden) # hidden is (h, c)
        # undo packing: second arg is input_lengths (we do not need)
        # o: seq_len x bs x d
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o)

        # dim transformation: (seq_len, bs, d) -> (seq_len x bs, d)
        o = o.contiguous()
        o = o.view(-1, o.shape[2])

        # run through prediction layer
        o = self.h2o(o)
        o = self.softmax(o)

        # dim transformation
        o = o.view(seq_len, bs, self.output_size)

        return o, hidden

class RNN_ILSTM(RNN_Memory, RNN_Independent):
    def describe(self):
        return '''independent lstm: each time step assign a different model'''
        
class RNN_SLSTM(RNN_Memory, RNN_Staged):
    def describe(self):
        return '''staged LSTM, manully share time steps'''
        
class RNN_LSTM_MoW(RNN_Memory, RNN_MoW):
    def describe(self):
        return '''mixture of weights for each time step with total of k clusters'''
    
###################################################################################### 
class RNN_MLP(RNN_Memoryless):
    '''logistic regression sharing the same interface with RNN, basically 
    ignore time dependence and only use last step information for prediction'''

    def custom_init(self):
        self.model = self.base_model()

    def get_model(self, t):
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
    


