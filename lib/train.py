import torch, time
from lib.utils import timeSince, AverageMeter, PrintTable, smooth
import pprint
import numpy as np
import warnings
import shutil
from torch.optim import lr_scheduler

class Train(object):

    def __init__(self, net, optimizer, criterion, data,
                 batch_size=32, n_iters=None,
                 n_print=100, n_save=10,
                 n_plot=1000,
                 save_filename='models/checkpoint.pth.tar',
                 best_save_filename=None,
                 val_data=None,
                 n_iters_val=None,
                 lr_decay_factor=0.1,
                 use_gpu=False): # specifies which gpu to use, if False, don't use

        if optimizer is not None:
            self.lr_decay_factor = lr_decay_factor
            self.scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            'max', patience=5,
                                                            verbose=True,
                                                            factor=lr_decay_factor)
        
        self.start_iter = 1
        if n_iters is None:
            n_iters = int(100000 / batch_size)
        if n_iters_val is None:
            if val_data is None:
                n_iters_val = 100 * batch_size
            else:
                n_iters_val = len(val_data)

        self.n_iters = n_iters
        self.n_iters_val = n_iters_val
        self.batch_size = batch_size        
        self.print_every = int(n_iters / n_print)
        self.plot_every = int(n_iters / n_plot)
        self.save_every = int(n_iters / n_save)
        self.use_gpu = use_gpu
            
        self.net = net
        if self.use_gpu is not False:
            self.net.cuda()
        
        self.optimizer = optimizer
        self.criterion = criterion
        self.data = data
        self.val_data = val_data

        self.save_filename = save_filename
        if best_save_filename is None:
            split = self.save_filename.split('.')
            split[0] = split[0] + '_best'
            self.best_save_filename = ".".join(split)

        # logging terms
        self.clear_logs()

    def __repr__(self):
        return pprint.pformat({'net': self.net,
                    'optimizer': self.optimizer,
                    'criterion': self.criterion,
                    'batch_size': self.batch_size,
                    'n_iters': self.n_iters})
        
    def clear_logs(self):
        self.all_losses = []
        self.val_accs = []
        self.best_acc = None

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.save_filename)
        if is_best:
            shutil.copyfile(self.save_filename, self.best_save_filename)

    def load_checkpoint(self, load_filename):
        print("=> loading checkpoint '{}'".format(load_filename))
        checkpoint = torch.load(load_filename)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                        'max', patience=5,
                                                        factor=self.lr_decay_factor,
                                                        verbose=True)
        self.start_iter = checkpoint['niter']
        self.best_acc = checkpoint['best_acc']
        self.all_losses = checkpoint['train_losses']
        self.val_accs = checkpoint['val_accs']        
        print("=> loaded checkpoint '{}' (iteration {})"
              .format(load_filename, checkpoint['niter']))

    def smooth_loss(self, step=None):
        if step is None:
            step = self.print_every
        return smooth(self.all_losses, step=step)

    def smooth_valacc(self, step=None):
        if step is None:
            step = self.print_every
        return smooth(self.val_accs, step=step)
    
    def train_step(self):
        raise NotImplementedError()

    def val_acc(self):

        # return avg validation accuracy
        if self.val_data is None:
            return None

        print('==> evaluating validation accuracy')        
        acc = AverageMeter()
        for iter in range(self.n_iters_val):
            # d is (x, y) for mlp, (x, y, x_lengths) for rnn
            d = self.val_data.next_batch(1) # 1 so don't need to mask out
            d = list(d)            
            if self.use_gpu is not False:
                d[0], d[1] = d[0].cuda(), d[1].cuda()
            y = d[1] # (seq_len, bs)

            output = self.net.eval_forward(*d) # seq_len x bs x output_size
            if len(output.shape) < 3: # seq_len x bs, single output
                max_dim = 1
            else: # seq_len x bs x output_size, multi output
                max_dim = 2
            _, ans = torch.max(output, max_dim)

            n = len(y)
            for j in range(n):
                if y[j].item() == ans[j].item():
                    acc.update(1) # correct
                else:
                    acc.update(0) # incorrect
        print('==> validation accuracy is %d%%' % (acc.avg * 100))
        self.scheduler.step(acc.avg)
        self.net.train()
        return acc.avg
        
    def train(self):
        self.net.train()
        losses = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        table_printer = PrintTable()

        table_printer.print(['#iter', 'progress', 'total_time',
                             'batch_time', 'data_time', 'avg_loss'])
        
        start = time.time()
        end = time.time()
        for iter in range(self.start_iter, self.n_iters + 1):
            # measure data loading time
            data_time.update(time.time() - end)
            
            # d is (x, y) for mlp, (x, y, x_lengths) for rnn
            d = self.data.next_batch(self.batch_size)
            d = list(d)            
            if self.use_gpu is not False:
                d[0], d[1] = d[0].cuda(), d[1].cuda()
            output, loss = self.train_step(*d)

            # note: the second arg should really valid sizes, but here
            # the expectation should be correct
            losses.update(loss, self.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            #################### note keeping ###########################
            # Print iter number, loss, etc.
            if iter % self.print_every == 0:
                table_printer.print([iter,
                                     "%d%%" % (iter / self.n_iters * 100),
                                     timeSince(start),
                                     batch_time.avg,
                                     data_time.avg,
                                     losses.avg])
                #print('%d %d%% (%s %.3f %.3f) %.4f ' % ())

            if iter % self.plot_every == 0:
                # Add current loss avg to list of losses
                self.all_losses.append(losses.val)
                if len(self.val_accs) == 0: 
                    self.val_accs.append(self.val_acc())
                else:
                    self.val_accs.append(self.val_accs[-1])

            if iter % self.save_every == 0 or iter == self.n_iters:
                val_acc = self.val_acc()
                self.val_accs.append(val_acc)
                is_best = False
                if val_acc is not None:
                    if self.best_acc is None:
                        self.best_acc = val_acc
                        is_best = True
                    else:
                        is_best = self.best_acc < val_acc                        
                        self.best_acc = max(val_acc, self.best_acc)

                self.save_checkpoint({
                    'niter': iter + 1,
                    'arch': str(type(self.net)),
                    'state_dict': self.net.state_dict(),
                    'best_acc': self.best_acc,
                    'optimizer': self.optimizer.state_dict(),
                    'train_losses': self.all_losses,
                    'val_accs': self.val_accs
                }, is_best)
            #################### note keeping end ###########################
            
            end = time.time()                

######################################## derived models ################################
class TrainMLP(Train):

    def train_step(self, x, y):

        self.optimizer.zero_grad()
        output = torch.nn.functional.log_softmax(self.net(x), dim=1)
        
        loss = self.criterion(output, y.view(-1))
        loss.backward()
        
        self.optimizer.step()
        return output, loss.item()

class TrainSORNN(Train): # single output rnn

    def train_step(self, x, y, x_lengths):
        hidden = self.net.initHidden(batch_size=self.batch_size)


        self.optimizer.zero_grad()
        output, hidden = self.net(x, hidden, x_lengths)

        ################################################################        
        # mask for appropriate output
        bs = output.shape[1]
        output = output[list(map(lambda l: l-1, x_lengths)), range(bs)]        
        ################################################################

        loss = self.criterion(output, y.view(-1))        
        loss.backward()
        self.net.after_backward()         
        self.optimizer.step()

        return output, loss.item()

class TrainMORNN(Train): # multi output rnn

    def train_step(self, x, y, x_lengths):
        hidden = self.net.initHidden(batch_size=self.batch_size)

        self.optimizer.zero_grad()
        output, hidden = self.net(x, hidden, x_lengths)

        ###################################################################
        ## mask for appropriate output: (seq_len, bs, output_size)
        # flat labels and predictions
        y = y.view(-1) # seq_len * bs
        yhat = output.view(-1, self.net.output_size) # (seq_len * bs, output_size)

        # create mask
        masks = [torch.ones(l) for l in x_lengths]
        mask = torch.nn.utils.rnn.pad_sequence(masks, padding_value=0).view(-1)
        mask = mask.nonzero().squeeze()

        # select out rows of yhat and y that
        yhat = yhat[mask]
        y = y[mask]
        ###################################################################

        loss = self.criterion(yhat, y.view(-1))   
        loss.backward()
        self.net.after_backward()                              
        self.optimizer.step()

        return output, loss.item()

