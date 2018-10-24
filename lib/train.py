import torch, time
from lib.utils import timeSince
import pprint

class Train(object):

    def __init__(self, net, optimizer, criterion, data,
                 batch_size=32, n_iters=None, print_every=None, plot_every=None):
        if n_iters is None:
            n_iters = int(100000 / batch_size)
        if print_every is None:
            print_every = int(n_iters / 100)
        if plot_every is None:
            plot_every = print_every
        
        self.net = net
        self.optimizer = optimizer
        self.criterion = criterion
        self.data = data
        
        self.batch_size = batch_size
        self.n_iters = n_iters
        self.print_every = print_every
        self.plot_every = plot_every

        # logging terms
        self.clear_logs()

    def __repr__(self):
        return pprint.pformat({'net': self.net,
                    'optimizer': self.optimizer,
                    'criterion': self.criterion,
                    'batch_size': self.batch_size,
                    'n_iters': self.n_iters})
        
    def clear_logs(self):
        self.all_losses = [] # todo: log this in some file 
        
    def train_step(self):
        raise NotImplementedError()

    def train(self):
        self.net.train()
        current_loss = 0
        start = time.time()

        for iter in range(1, self.n_iters + 1):
            # if in order, use next_batch, d is (x, y) for mlp, (x, y, x_lengths) for rnn
            d = self.data.random_batch(self.batch_size) 
            output, loss = self.train_step(*d)
            current_loss += loss

            # Print iter number, loss, name and guess
            if iter % self.print_every == 0:
                print('%d %d%% (%s) %.4f ' % (iter, iter / self.n_iters * 100,
                                              timeSince(start),
                                              current_loss / self.plot_every \
                                              / self.batch_size))

            # Add current loss avg to list of losses
            if iter % self.plot_every == 0:
                self.all_losses.append(current_loss / self.plot_every \
                                       /self.batch_size)
                current_loss = 0

class TrainMLP(Train):

    def train_step(self, x, y):

        self.optimizer.zero_grad()
        output = torch.nn.functional.log_softmax(self.net(x), dim=1)
        
        loss = self.criterion(output, y)
        loss.backward()
        
        self.optimizer.step()
        return output, loss.item()

class TrainRNN(Train):

    def train_step(self, x, y, x_lengths):
        hidden = self.net.initHidden(batch_size=self.batch_size)

        self.optimizer.zero_grad()
        output, hidden = self.net(x, hidden, x_lengths)

        loss = self.criterion(output, y)
        loss.backward()
                        
        self.optimizer.step()

        return output, loss.item()

class TrainMetaRNN(Train):

    def train_step(self, x, y, x_lengths):
        hidden = self.net.initHidden(batch_size=self.batch_size)

        self.optimizer.zero_grad()
        output, hidden = self.net(x, hidden, x_lengths)
        loss = self.criterion(output, y)
        loss.backward()
        self.net.after_backward()                      
        self.optimizer.step()

        return output, loss.item()

# debug function: todo: delete
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
    

