import numpy as np
import random, os
import torch, warnings
from sklearn.externals import joblib
from torch.utils.data import Dataset

class Data(Dataset):

    def __init__(self, dataset):
        self.dset = dataset # torch dataset
        self.reset()

    def reset(self):
        self.order = np.arange(len(self.dset))
        self.current_index = 0
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

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, index):
        return self.get(index)
    
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

class LoadedSequenceData(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.xs, self.ys = joblib.load(data_path)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]

class SequenceData(Data):
    '''sequence synthetic dataset'''
    def custom_init(self):
        self.min_length = 1
        self.max_length = 3
        self.load_path = None # if loaded

    def set_seq_length(self, min_length, max_length):
        self.min_length = min_length
        self.max_length = max_length

    def save_data(self, savename, n, override_existing=False):
        file_exist = os.path.exists(savename)
        if file_exist:
            warnings.warn("%s exist, override: %r" % (savename, override_existing))

        if not file_exist or override_existing:
            print("==>save data of size %d in %s" % (n, savename))
            # generate on the fly        
            xs, ys = self._generate_unpadded_data(n, on_the_fly=True) 
            joblib.dump((xs, ys), savename)
            print("==>save data done")

        self.load_data(savename)        

    def load_data(self, load_path):
        print('==>load data %s' % load_path)
        self.dset = LoadedSequenceData(load_path)
        self.reset()
        self.load_path = load_path 
        print('==>load data of size %d done' % len(self.dset))
        
    def __len__(self):
        # this is dummy as sequence data is generated on the fly        
        return len(self.dset)
        
    def __getitem__(self, index):
        return _random_one()

    def _random_one(self): # random one instance
        raise NotImplementedError()

    def _sort_length_pad(self, xs, ys):
        '''sort length from large to small and pad xs and ys'''
        sort_order = sorted(range(len(xs)), key=lambda i: len(xs[i]), reverse=True)
        xs = [xs[i] for i in sort_order]
        ys = [ys[i] for i in sort_order]
        x_lengths = [len(l) for l in xs]
        max_length = x_lengths[0]
        xs = torch.nn.utils.rnn.pad_sequence(xs)
        ys = torch.nn.utils.rnn.pad_sequence(ys) # seq_len x bs
        return xs, ys, x_lengths

    def _generate_unpadded_data(self, batch_size, on_the_fly=True):
        '''return a random batch'''
        xs = []
        ys = []

        for _ in range(batch_size):
            if on_the_fly:
                x, y = self._random_one()
            else:
                x, y = self.get(self.current_index)
                self.current_index = self.current_index + 1
                if self.current_index >= len(self.dset):
                    self.current_index = self.current_index % len(self.dset)
                    self.random_shuffle()
            xs.append(x)
            ys.append(y)
        return xs, ys

    def next_batch(self, batch_size):
        on_the_fly = self.load_path is None
        xs, ys = self._generate_unpadded_data(batch_size, on_the_fly)
        # pad both x and y tensors:
        return self._sort_length_pad(xs, ys)
                                
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

class StateData(SequenceData):
    ''' y is not only depend on x, but also on a time related hidden state'''
    def state_transition(self, s, t, x, y):
        # next state depends on time and current covariate and current state
        raise NotImplementedError()

    def debug_mode(self, debug=True):
        self.debug = debug
    
    def set_target(self, y, s):
        raise NotImplementedError()

    def init_state(self):
        return 0 # initial state
        
    def _random_one(self):
        # randomly generate one example
        length = np.random.randint(self.min_length, self.max_length+1)
        chosen = np.random.choice(len(self.dset), length)

        state = self.init_state()
        states = []
        xs = []
        ys = []
        for t, c in enumerate(chosen):
            states.append(state)
            x, orig_y = self.get(c)
            xs.append(x)
            y = self.set_target(orig_y, state)
            ys.append(y.unsqueeze(0))
            state = self.state_transition(state, t, x, orig_y)

        xs = torch.cat(xs, 0) # (nseq, w, h)
        ys = torch.cat(ys)
            
        xs = xs.view(length, -1) # (nseq, w*h)
        if hasattr(self, 'debug') and self.debug is True:
            print('states:', states)
        return xs, ys

########## making lstm fail #############
class TwoStateMNISTData(StateData):
    ''' two state transition model with an absorbing state'''
    def set_p(self, p):
        self.p = p
        
    def state_transition(self, s, t, x, y):
        p = self.p or 0.1
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
        
class AlternateStateMNISTData(StateData):
    '''alternate between models to use'''
    def state_transition(self, s, t, x, y):
        return 1 - s

    def set_target(self, y, s):
        if s == 0:
            return y
        else:
            return (10 - y) % 10

class ShiftStateMNISTData(StateData):
    '''all time steps have a differnet model'''
    def set_offset(self, offset):
        self.offset = offset # offset is list
        
    def state_transition(self, s, t, x, y):
        return t + 1

    def set_target(self, y, s):
        return (y + self.offset[s]) % 10

class StateMNISTData(StateData): 
    '''a generalization of shiftstatemnistdata'''
    def set_target_function(self, target_function):
        self.target_function = target_function
        
    def state_transition(self, s, t, x, y):
        return t + 1

    def set_target(self, y, s):
        return self.target_function[s][y.item()] % 10

class PartialSumStateMNISTData(StateData): 
    '''statemnistdata with memory, does the partial sum task with share cycle'''
    def set_share_cycle(self, n):
        self.share_cycle = n

    def set_target_function(self, target_function):
        self.target_function = target_function

    def init_state(self):
        return 0, 0 # (time, partial sum)
    
    def state_transition(self, s, t, x, y):
        # output next state
        time, ps = s
        next_ps = (ps + y.item()) % 10
        if hasattr(self, 'share_cycle'):
            if t % self.share_cycle == self.share_cycle - 1:
                next_ps = 0
        return t + 1,  next_ps

    def set_target(self, y, s):
        t, ps = s
        if hasattr(self, 'debug') and self.debug:
            return (ps + y) % 10        
        return (ps + self.target_function[t][y.item()]) % 10
    
''' examples of state mnist data
# 1. sharing 3 groups of data
min_length, max_length = 1, 9
target_function = [torch.randperm(n_categories) \
                   for _ in range(math.ceil(max_length / 3))]
target_function = [item for item in target_function for _ in range(3)]
'''
        
        
