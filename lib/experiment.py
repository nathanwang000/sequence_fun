from lib.data import StateMNISTData, PartialSumStateMNISTData
import torch.nn as nn
import math, torch, os
from lib.train import TrainMORNN
from lib.model import MODELS
import torchvision.transforms as transforms
import torchvision.datasets as dset
from lib.utils import random_split_dataset

# exact setting for each experiment ran
SYNTHETIC_EXPERIMENTS = {}

def register_exp(expname, exp):
    exp.set_target_function()
    SYNTHETIC_EXPERIMENTS[expname] = exp

class Experiment(object):

    def __init__(self):
        self.share_cycle = 3
        self.max_length = 9
        self.min_length = 1
        self.reset0 = True # partial sum reset to 0
        self.shared_groups = [[0,1,2], [3, 4, 5]]

    def get_train_val_test(self):
        root = './mnist_data'
        if not os.path.exists(root):
            os.mkdir(root)
    
        trans = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])
        # if not exist, download mnist dataset
        trainval_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
        train_proportion = 0.8
        train_set, val_set = random_split_dataset(trainval_set,
                                                  [train_proportion, 1-train_proportion])
        test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)
        return train_set, val_set, test_set
        
    def set_target_function(self):
        n_categories = 10
        target_function = [torch.randperm(n_categories) \
                           for _ in range(math.ceil(self.max_length / self.share_cycle))]
        target_function = [item for item in target_function for _ in
                           range(self.share_cycle)]
        self.target_function = target_function

    def gen_data_helper(self, dataset):
        raise NotImplementedError()
        
    def gen_data(self, dataset, override_data=False,
                 data_savename=None, data_length=None):
        data = self.gen_data_helper(dataset)
        if data_savename is not None and data_length is not None:        
            data.save_data(data_savename, data_length,
                           override_existing=override_data)
        return data

    def run(self, args, train_data, val_data):
        net = MODELS[args.arch](784, args.nhidden, 10, args.num_layers)

        print(args)
        if args.arch == 'RNN_LSTM':
            savename = 'lstm.pth.tar'
        elif args.arch == 'RNN_LSTM_2layers':
            savename = 'lstm2.pth.tar'
        elif args.arch == 'RNN_LSTM_MoO':
            net.setKT(args.nshared, self.max_length)
            savename = 'lstm_moo_%d.pth.tar' % args.nshared       
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
        if args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        elif args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                                        momentum=0.9,
                                        nesterov=True)
        criterion = nn.NLLLoss()

        trainer = TrainMORNN(net, optimizer, criterion, train_data,
                             save_filename=savename, val_data=val_data,
                             use_gpu=args.use_gpu, n_iters=args.niters,
                             n_save=args.n_save_model,
                             batch_size=args.batch_size)
        if os.path.exists(savename) and not args.override_model:
            trainer.load_checkpoint(savename)

        trainer.train()

class SameTask(Experiment):

    def gen_data_helper(self, dataset):
        data = StateMNISTData(dataset)
        data.set_seq_length(min_length=self.min_length,
                            max_length=self.max_length)
        data.set_target_function(self.target_function)
        return data

class Scarce(Experiment):

    def gen_data_helper(self, dataset):
        data = StateMNISTData(dataset)
        data.set_seq_length(min_length=self.min_length,
                            max_length=self.max_length)
        data.set_target_function(self.target_function)
        return data

class PartialSum(Experiment):

    def gen_data_helper(self, dataset):
        data = PartialSumStateMNISTData(dataset)
        data.set_seq_length(min_length=self.min_length,
                            max_length=self.max_length)
        data.set_target_function(self.target_function)
        data.set_share_cycle(self.share_cycle)
        data.set_reset0(self.reset0)
        
        return data

register_exp('partial_sum', PartialSum())

exp = PartialSum()
exp.share_cycle =  4
exp.shared_groups = [[0,1,2,3], [4,5,6,7], [8]]
register_exp('partial_sum2', exp)

exp = PartialSum()
exp.min_length = 1
exp.max_length = 3
exp.share_cycle =  3
exp.shared_grouexp = [[0,1,2]]
register_exp('partial_sum3', exp)

######## actually interesting settings ###########
########### no memory ##########
### all shared
exp = SameTask()
exp.min_length = 5
exp.max_length = 5
exp.share_cycle = 5
exp.shared_groups = [[0,1,2,3,4]]
register_exp('same_task', exp)    

### partially shared
register_exp('scarce', Scarce())

### independent
exp = SameTask()
exp.min_length = 5
exp.max_length = 5
exp.share_cycle = 1
exp.shared_groups = [[0,1,2],[3,4]]
register_exp('diff_task', exp)    

############ has memory ###########
### all shared
exp = PartialSum()
exp.min_length = 5
exp.max_length = 5
exp.share_cycle = 5
exp.shared_grouexp = [[0], [1,2,3,4]]
register_exp('partial_sum4', exp)

### partially shared
exp = PartialSum()
exp.min_length = 8
exp.max_length = 8
exp.share_cycle = 4
exp.shared_grouexp = [[0,1,2,3], [4,5,6,7]]
register_exp('partial_sum_share', exp)

### independent: don't reset0 in partial sum
exp = PartialSum()
exp.min_length = 5
exp.max_length = 5
exp.share_cycle = 1
exp.reset0 = False
exp.shared_grouexp = [[0,1,2], [3,4]]
register_exp('diff_memory', exp)
