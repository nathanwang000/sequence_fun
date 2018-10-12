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
class RNN_LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm = torch.nn.LSTM(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size) # try to tinker with this
        self.softmax = nn.LogSoftmax(dim=1)

    def eval_forward(self, x, y, x_lengths):
        # this function is for evaluation        
        self.eval()
        batch_size = len(y)
        hidden = self.initHidden(batch_size=batch_size)
        output, hidden = self.forward(x, hidden, x_lengths)
        return output
        
    def forward(self, input, hidden, input_lengths):
        bs = input.shape[1]
        
        # change the padded data with variable length
        input = torch.nn.utils.rnn.pack_padded_sequence(input, input_lengths)
        
        o, (h, cell) = self.lstm(input, hidden) # entire word
        # inverse run: do not need this in my case:
        #o, _ = torch.nn.utils.rnn.pad_packed_sequence(o)

        # only use hidden state, minibatch input: h of size nseq x bs x d        
        output = self.h2o(h.transpose(0,1).view(bs, -1)) 
        output = self.softmax(output)
        return output, h

    def initHidden(self, batch_size):
        # for both cell memory and hidden neurons
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))
