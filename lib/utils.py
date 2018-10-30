import time, math, torch, shutil
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class PrintTable(object):
    '''print tabular data in a nice format'''
    def __init__(self, nstr=15, nfloat=5):
        self.nfloat = nfloat
        self.nstr = nstr

    def _format(self, x):
        if type(x) is float:
            x_tr = ("%." + str(self.nfloat) + "f") % x
        else:
            x_tr = str(x)
        return ("%" + str(self.nstr) + "s") % x_tr

    def print(self, row):
        print( "|".join([self._format(x) for x in row]) )

def smooth(sequence, step=1):
    out = np.convolve(sequence, np.ones(step), 'valid') / step
    return out

def random_split_dataset(dataset, proportions):
    n = len(dataset)
    ns = [int(math.floor(p*n)) for p in proportions]
    ns[-1] += n - sum(ns)
    return torch.utils.data.random_split(dataset, ns)
    
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# debug function for MoW and MoE functions
def print_grad(net):
    print('\nmodels:')
    for i, m in enumerate(net.models):
        print(i)
        for p in m.parameters():
            if p.grad is not None:
                print(torch.sum(torch.abs(p.grad)))

    print('\nmeta models:')
    for i, m in enumerate(net.meta_models):
        print(i)
        for p in m.parameters():
            if p.grad is not None:            
                print(torch.sum(torch.abs(p.grad)))
    
