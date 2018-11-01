import numpy as np
import os, copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import time
import math, tqdm
from lib.data import FlatData, MNIST_add_data
from lib.utils import timeSince
from lib.utils import random_split_dataset
import random
import torch.backends.cudnn as cudnn
import warnings, argparse
from lib.data import StateMNISTData
from lib.train import TrainMORNN
from lib.model import RNN_LSTM, RNN_MLP, RNN_LSTM_MoW, RNN_SLSTM, RNN_ILSTM, \
    RNN_IMLP, RNN_MLP_MoW, RNN_SMLP

model_names = ['RNN_LSTM', 'RNN_LSTM_MoW', 'RNN_SLSTM', 'RNN_ILSTM', \
               'RNN_MLP', 'RNN_IMLP', 'RNN_MLP_MoW', 'RNN_SMLP']

# parse arguments
parser = argparse.ArgumentParser(description='PyTorch RNN multi task training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='RNN_LSTM',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: RNN_LSTM)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--nhidden', default=30, type=int,
                    help='number of hidden units')
parser.add_argument('--use_gpu', action='store_true',
                    help='whether or not use gpu')
parser.add_argument('--smdir', default='models', type=str,
                    help='directory to save model')
parser.add_argument('--sddir', default='sequence_data', type=str,
                    help='directory to save data')
parser.add_argument('--nshared', default=2, type=int,
                    help='number of shared models')
parser.add_argument('--ntr', default=5000, type=int,
                    help='number of training data')
parser.add_argument('--niters', default=4000, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--batch_size', default=32, type=int, metavar='B',
                    help='batch_size')

args = parser.parse_args()

################################## setting ############################################
n_hidden = args.nhidden # 50, 300
n_categories = 10
os.system('mkdir -p %s' % args.smdir)
os.system('mkdir -p %s' % args.sddir)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True
    warnings.warn('You have chosen to seed training. '
                  'This will turn on the CUDNN deterministic setting, '
                  'which can slow down your training considerably! '
                  'You may see unexpected behavior when restarting '
                  'from checkpoints.')
    
'''StateMNISTData'''
min_length, max_length = 1, 9
target_function = [torch.randperm(n_categories) \
                   for _ in range(math.ceil(max_length / 3))]
# repeat 3 times
target_function = [item for item in target_function for _ in range(3)]

################################### get data ###########################################
root = './mnist_data'
use_gpu = args.use_gpu
if not os.path.exists(root):
    os.mkdir(root)
    
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
# if not exist, download mnist dataset
trainval_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
train_proportion = 0.8
train_set, val_set = random_split_dataset(trainval_set, [train_proportion, 1-train_proportion])
test_set = dset.MNIST(root=root, train=False, transform=trans, download=True)

savename_tr = os.path.join(args.sddir, 'train.pkl')
savename_val = os.path.join(args.sddir, 'val.pkl')
savename_te = os.path.join(args.sddir, 'test.pkl')
n_tr = args.ntr
n_val = 3000 # don't need to vary
n_te = 10000 # don't need to vary

train_data = StateMNISTData(train_set)
train_data.set_seq_length(min_length=min_length, max_length=max_length)
train_data.set_target_function(target_function)
train_data.save_data(savename_tr, n_tr)

val_data = StateMNISTData(val_set)
val_data.set_seq_length(min_length=min_length, max_length=max_length)
val_data.set_target_function(target_function)
val_data.save_data(savename_val, n_val)

test_data = StateMNISTData(test_set)
test_data.set_seq_length(min_length=min_length, max_length=max_length)
test_data.set_target_function(target_function)
test_data.save_data(savename_te, n_te)

######################################### run models ###################################
def run_models(args):
    net = eval(args.arch)(784, n_hidden, n_categories)

    print(args)
    if args.arch == 'RNN_LSTM':
        savename = 'lstm.pth.tar'
    elif args.arch == 'RNN_MLP':
        savename = 'mlp.pth.tar'        
    elif args.arch == 'RNN_LSTM_MoW':
        net.setKT(args.nshared, max_length)
        savename = 'lstm_mow.pth.tar'
    elif args.arch == 'RNN_MLP_MoW':
        net.setKT(args.nshared, max_length)
        savename = 'mlp_mow.pth.tar'
    elif args.arch == 'RNN_SLSTM':
        net.set_shared_groups([[0,1,2], [3, 4, 5]])
        savename = 'lstm_shared.pth.tar'
    elif args.arch == 'RNN_SMLP':
        net.set_shared_groups([[0,1,2], [3, 4, 5]])
        savename = 'mlp_shared.pth.tar'
    elif args.arch == 'RNN_ILSTM':
        net.set_max_length(max_length)
        savename = 'lstm_independent.pth.tar'
    elif args.arch == 'RNN_IMLP':
        net.set_max_length(max_length)
        savename = 'mlp_independent.pth.tar'
        
    savename = os.path.join(args.smdir, savename)
    optimizer = torch.optim.Adam(net.parameters())
    criterion = nn.NLLLoss()

    trainer = TrainMORNN(net, optimizer, criterion, train_data,
                         save_filename=savename, val_data=val_data,
                         use_gpu=use_gpu, n_iters=args.niters,
                         batch_size=args.batch_size)
    if os.path.exists(savename):
        trainer.load_checkpoint(savename)

    trainer.train()

run_models(args)