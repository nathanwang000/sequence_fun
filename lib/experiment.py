from lib.data import StateMNISTData, PartialSumStateMNISTData
import torch.nn as nn
import math, torch, os
from lib.train import TrainMORNN
from lib.model import RNN_LSTM, RNN_MLP, RNN_LSTM_MoW, RNN_SLSTM, RNN_ILSTM, \
    RNN_IMLP, RNN_MLP_MoW, RNN_SMLP

# exact setting for each experiment ran
SYNTHETIC_EXPERIMENTS = {}

class Experiment(object):

    def __init__(self):
        self.share_cycle = 3
        self.max_length = 9
        self.min_length = 1
        self.shared_groups = [[0,1,2], [3, 4, 5]]

    def set_target_function(self):
        n_categories = 10
        target_function = [torch.randperm(n_categories) \
                           for _ in range(math.ceil(self.max_length / self.share_cycle))]
        target_function = [item for item in target_function for _ in
                           range(self.share_cycle)]
        self.target_function = target_function
    
    def gen_data(self, dataset, data_savename=None, data_length=None):
        raise NotImplementedError()

    def run(self, args, train_data, val_data):
        net = eval(args.arch)(784, args.nhidden, 10)

        print(args)
        if args.arch == 'RNN_LSTM':
            savename = 'lstm.pth.tar'
        elif args.arch == 'RNN_MLP':
            savename = 'mlp.pth.tar'        
        elif args.arch == 'RNN_LSTM_MoW':
            net.setKT(args.nshared, self.max_length)
            savename = 'lstm_mow_%d.pth.tar' % args.nshared
        elif args.arch == 'RNN_MLP_MoW':
            net.setKT(args.nshared, self.max_length)
            savename = 'mlp_mow_%d.pth.tar' % args.nshared
        elif args.arch == 'RNN_SLSTM':
            net.set_shared_groups(self.shared_groups)
            savename = 'lstm_shared.pth.tar'
        elif args.arch == 'RNN_SMLP':
            net.set_shared_groups(self.shared_groups)
            savename = 'mlp_shared.pth.tar'
        elif args.arch == 'RNN_ILSTM':
            net.set_max_length(self.max_length)
            savename = 'lstm_independent.pth.tar'
        elif args.arch == 'RNN_IMLP':
            net.set_max_length(self.max_length)
            savename = 'mlp_independent.pth.tar'
        
        savename = os.path.join(args.smdir, savename)
        optimizer = torch.optim.Adam(net.parameters())
        criterion = nn.NLLLoss()

        trainer = TrainMORNN(net, optimizer, criterion, train_data,
                             save_filename=savename, val_data=val_data,
                             use_gpu=args.use_gpu, n_iters=args.niters,
                             n_save=args.n_save_model,
                             batch_size=args.batch_size)
        if os.path.exists(savename):
            trainer.load_checkpoint(savename)

        trainer.train()

class Scarce(Experiment):

    def gen_data(self, dataset, data_savename=None, data_length=None):
        data = StateMNISTData(dataset)
        data.set_seq_length(min_length=self.min_length,
                            max_length=self.max_length)
        data.set_target_function(self.target_function)
        if data_savename is not None and data_length is not None:        
            data.save_data(data_savename, data_length)
        return data

class PartialSum(Experiment):

    def gen_data(self, dataset, data_savename=None, data_length=None):
        data = PartialSumStateMNISTData(dataset)
        data.set_seq_length(min_length=self.min_length,
                            max_length=self.max_length)
        data.set_target_function(self.target_function)
        data.set_share_cycle(self.share_cycle)

        if data_savename is not None and data_length is not None:
            data.save_data(data_savename, data_length)
        return data

exp = Scarce()
exp.set_target_function()
SYNTHETIC_EXPERIMENTS['scarce'] = exp

exp = PartialSum()
exp.set_target_function()
SYNTHETIC_EXPERIMENTS['partial_sum'] = exp

exp = PartialSum()
exp.share_cycle =  4
exp.shared_groups = [[0,1,2,3], [4,5,6,7], [8]]
exp.set_target_function()
SYNTHETIC_EXPERIMENTS['partial_sum2'] = exp

exp = PartialSum()
exp.min_length = 1
exp.max_length = 3
exp.share_cycle =  3
exp.shared_grouexp = [[0,1,2]]
exp.set_target_function()
SYNTHETIC_EXPERIMENTS['partial_sum3'] = exp
