import torch, math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

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
    
    def eval_forward(self, x, y, x_lengths):
        # this function is for evaluation        
        self.eval()
        batch_size = len(y)
        hidden = self.initHidden(batch_size=batch_size)
        output, hidden = self.forward(x, hidden, x_lengths)
        return output

    def initHidden(self, batch_size):
        # for both cell memory and hidden neurons
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))

    def forward(self, input, hidden, input_lengths):
        raise NotImplementedError()

class RNN_LSTM(RNN):

    def custom_init(self):
        self.lstm = torch.nn.LSTM(self.input_size, self.hidden_size)        

    def forward(self, input, hidden, input_lengths):
        bs = input.shape[1]
        
        # change the padded data with variable length: save computation
        input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths)
        
        o, (h, cell) = self.lstm(input, hidden) # entire word
        # inverse run: do not need this in my case:
        #o, _ = torch.nn.utils.rnn.pad_packed_sequence(o)

        # RNN_ILSTM is perhaps more appropriate: todo
        # only use hidden state, minibatch input: h of size num_layers x bs x d        
        output = self.h2o(h.transpose(0,1).view(bs, -1)) 
        output = self.softmax(output)
        return output, (h, cell)

class RNN_ILSTM(RNN):
    ''' independent lstm: each time step assign a different model '''
    def custom_init(self):
        self.lstms = torch.nn.ModuleList() # modulelist makes module params visible

    def get_lstm(self, t):
        if t >= len(self.lstms):
            self.lstms.append(torch.nn.LSTM(self.input_size, self.hidden_size))
        return self.lstms[t]
        
    def forward(self, x, hidden, input_lengths):
        # input of size seq_len x bs x d, padded sequences with input_lengths specified
        # this model assumes one output at the end
        seq_len, bs, d = x.shape
        
        outputs = []
        for i in range(seq_len):
            # get a new lstm for each time step            
            lstm = self.get_lstm(i)
            o, hidden = lstm(x[i].view(1, bs, d), hidden)
            outputs.append(o)
        o = torch.cat(outputs, 0)
        h, c = hidden
        
        # mask o by input_length, seq_len x bs x d
        h_t = o[list(map(lambda l: l-1, input_lengths)), range(bs)]
        
        # only use hidden state, minibatch input: h_t of size bs x d
        output = self.h2o(h_t)
        output = self.softmax(output)
        return output, hidden
        
class RNN_SLSTM(RNN):
    ''' staged LSTM, manully share time steps '''
    def custom_init(self):
        self.groups = []
        self.lstms = torch.nn.ModuleList()
        self.set_shared_groups([])

    def set_shared_groups(self, groups):
        # each group share one model
        # assumes groups has no member overlap
        self.groups = groups
        while len(self.lstms) <= len(self.groups):
            self.lstms.append(torch.nn.LSTM(self.input_size, self.hidden_size))
    
    def get_lstm(self, t): # todo
        for i, g in enumerate(self.groups):
            if t in g:
                return self.lstms[i]
        return self.lstms[len(self.groups)]
        
    def forward(self, x, hidden, input_lengths):
        # input of size seq_len x bs x d, padded sequences with input_lengths specified
        # this model assumes one output at the end
        seq_len, bs, d = x.shape
        
        outputs = []
        for i in range(seq_len):
            lstm = self.get_lstm(i)
            o, hidden = lstm(x[i].view(1, bs, d), hidden)
            outputs.append(o)
        o = torch.cat(outputs, 0)
        h, c = hidden
        
        # mask o by input_length, seq_len x bs x d
        h_t = o[list(map(lambda l: l-1, input_lengths)), range(bs)]
        
        # only use hidden state, minibatch input: h_t of size bs x d
        output = self.h2o(h_t)
        output = self.softmax(output)
        return output, hidden
        
class RNN_MLP(RNN):
    '''logistic regression sharing the same interface with RNN, basically 
    ignore time dependence and only use last step information for prediction'''

    def custom_init(self):
        self.classifier = MLP([self.input_size, self.hidden_size, self.output_size])
    
    def forward(self, x, hidden, input_lengths):
        # input of size seq_len x bs x d, padded sequences with input_lengths specified
        # this model assumes one output at the end
        seq_len, bs, d = x.shape

        outputs = []
        for i, l in enumerate(input_lengths):
            o = self.classifier(x[l-1, i].view(1, d))
            outputs.append(o)
        o = torch.cat(outputs, 0)
        o = self.softmax(o)

        return o, hidden # hidden is dummy here

