import numpy as np
import random
import torch

class Data(object):

    def __init__(self, dataset):
        self.dset = dataset # torch dataset
        self.order = np.arange(len(self.dset))
        
        self.custom_init()

    def custom_init(self):
        raise NotImplementedError()

    def get(self, index):
        return self.dset[self.order[index]]

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.order)

    def __len__(self):
        raise NotImplementedError()

    def next_batch(self, batch_size):
        raise NotImplementedError()        
    
class FlatData(Data):

    def custom_init(self):
        self.current_index = 0

    def __len__(self):
        return len(self.dset)

    def _get_batch(self, chosen):
        xs = []
        ys = []
        for index in chosen:
            x, y = self.get(index)
            xs.append(x)
            ys.append(y.unsqueeze(0))
        xs = torch.cat(xs, 0)
        ys = torch.cat(ys)
        return xs, ys

    def next_batch(self, batch_size):
        chosen = [((self.current_index + i) % len(self)) for i in range(batch_size)]
        xs, ys = self._get_batch(chosen)
        if self.current_index + batch_size >= len(self):
            self.random_shuffle()
        self.current_index = (self.current_index + batch_size) % len(self)
        return xs, ys

class SequenceData(Data):
    '''sequence synthetic dataset'''
    def custom_init(self):
        self.min_length = 1
        self.max_length = 3

    def set_seq_length(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length

    def _random_one(self): # random one instance
        raise NotImplementedError()

    def next_batch(self, batch_size):
        '''return a random batch'''
        xs = []
        ys = []

        for _ in range(batch_size):
            x, y = self._random_one()
            xs.append(x)
            ys.append(y)

        # pad both x and y tensors: 
        sort_order = sorted(range(len(xs)), key=lambda i: len(xs[i]), reverse=True)
        xs = [xs[i] for i in sort_order]
        ys = [ys[i] for i in sort_order]
        x_lengths = [len(l) for l in xs]
        max_length = x_lengths[0]
        xs = torch.nn.utils.rnn.pad_sequence(xs)
        ys = torch.nn.utils.rnn.pad_sequence(ys) # seq_len x bs
        return xs, ys, x_lengths
                                
#######################################################################################
class MNIST_add_data(SequenceData):
    # add mnist variable number of times
    def _random_one(self):
        length = np.random.randint(self.min_length, self.max_length+1)
        chosen = np.random.choice(len(self.dset), length)

        xs = []
        ys = []
        for c in chosen:
            x, y = self.get(c)
            xs.append(x)
            ys.append(y.unsqueeze(0))
            
        xs = torch.cat(xs, 0) # (nseq, w, h)
        ys = torch.cat(ys)
            
        xs = xs.view(length, -1) # (nseq, w*h)
        return xs, ys.sum().unsqueeze(0) % 10

class TwoStateData(SequenceData):
    ''' y is not only depend on x, but also on a time related hidden state'''
    def state_transition(self, s, t, x):
        # next state depends on time and current covariate and current state
        raise NotImplementedError()

    def set_target(self, y, s):
        raise NotImplementedError()
        
    def _random_one(self):
        # randomly generate one example
        length = np.random.randint(self.min_length, self.max_length+1)
        chosen = np.random.choice(len(self.dset), length)

        state = 0
        states = []
        xs = []
        ys = []
        for t, c in enumerate(chosen):
            states.append(state)
            x, y = self.get(c)
            xs.append(x)
            y = self.set_target(y, state)
            ys.append(y.unsqueeze(0))
            state = self.state_transition(state, t, x)

        xs = torch.cat(xs, 0) # (nseq, w, h)
        ys = torch.cat(ys)
            
        xs = xs.view(length, -1) # (nseq, w*h)
        # print(states)
        return xs, ys
    
class TwoStateMNISTData(TwoStateData):
    ''' y is not only depend on x, but also on a time related hidden state'''
    def state_transition(self, s, t, x):
        p = 0.1
        if s == 1:
            return 1
        else:
            if np.random.random() < p:
                return 0
            else:
                return 1

    def set_target(self, y, s):
        if s == 0:
            return y
        else:
            return (10 - y) % 10
        

        
        
