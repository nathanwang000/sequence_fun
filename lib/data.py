import numpy as np
import random
import torch
from lib.utils import pad_timeseries_tensor

class Data(object):

    def __init__(self, dataset):
        self.dset = dataset
        self.order = np.arange(len(self.dset))
        self.current_index = 0

    def __len__(self):
        return len(self.dset)

    def random_shuffle(self, seed=None):
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.order)

    def get(self, index):
        return self.dset[self.order[index]]

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
        
    def random_batch(self, batch_size):
        chosen = np.random.choice(len(self), batch_size)
        xs, ys = self._get_batch(chosen)
        return xs, ys
    
    def next_batch(self, batch_size):
        chosen = [((self.current_index + i) % len(self)) for i in range(self.batch_size)]
        xs, ys = self._get_batch(chosen)
        self.current_index = (self.current_index + self.batch_size) % len(self)
        return xs, ys
        
        
class MNIST_add_data(Data):
    # add mnist 1-3 times
    def __init__(self, dataset, min_length=1, max_length=3):
        Data.__init__(self, dataset)
        self.min_length = min_length
        self.max_length = max_length
    
    def __len__(self):
        raise NotImplementedError()

    def next_batch(self):
        raise NotImplementedError()

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
            
        xs = xs.view(length, 1, -1) # (nseq, 1, w*h)
        return xs, ys.sum().unsqueeze(0) % 10

    def random_batch(self, batch_size):
        xs = []
        ys = []

        for _ in range(batch_size):
            x, y = self._random_one()
            xs.append(x)
            ys.append(y)

        # pad tensors
        sort_order = sorted(range(len(xs)), key=lambda i: len(xs[i]), reverse=True)
        xs = [xs[i] for i in sort_order]
        ys = [ys[i] for i in sort_order]
        x_lengths = [len(l) for l in xs]
        max_length = x_lengths[0]
        xs = torch.cat([pad_timeseries_tensor(x, max_length) for x in xs], 1)
        ys = torch.cat(ys)
        return xs, ys, x_lengths
                                
                                        
                                                            
    
