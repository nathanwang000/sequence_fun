import numpy as np
import os, copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time
import math, tqdm
from lib.utils import timeSince
from lib.model import MODELS
import random
import torch.backends.cudnn as cudnn
import warnings, argparse
from lib.experiment import SYNTHETIC_EXPERIMENTS
torch.set_num_threads(1)
model_names = MODELS.keys()
exp_names = SYNTHETIC_EXPERIMENTS.keys()

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
parser.add_argument('--override_data', action='store_true',
                    help='whether override data')
parser.add_argument('--override_model', action='store_true',
                    help='whether override model')
parser.add_argument('--smdir', default='models', type=str,
                    help='directory to save model')
parser.add_argument('--batch_size', default=32, type=int, metavar='B',
                    help='batch_size')
parser.add_argument('--niters', default=8000, type=int, metavar='N',
                    help='number of total iterations to run')
parser.add_argument('--n_save_model', default=10, type=int,
                    help='number of model save and validation eval in trainer')
parser.add_argument('--nshared', default=2, type=int,
                    help='number of shared models (for MoW)')
parser.add_argument('--num_layers', default=1, type=int,
                    help='number of layers for lstm')
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], 
                    help='which optimizer to use')
parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')


####################### data generate parameters ########################
parser.add_argument('--sddir', default='sequence_data', type=str,
                    help='directory to save data')
parser.add_argument('--ntr', default=5000, type=int,
                    help='number of training data')
parser.add_argument('--exp', default="scarce", type=str,
                    choices=exp_names,
                    help='experiment name to run')

args = parser.parse_args()

################################## setting ############################################
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
    
################################### get data ###########################################
savename_tr = os.path.join(args.sddir, 'train.pkl')
savename_val = os.path.join(args.sddir, 'val.pkl')
savename_te = os.path.join(args.sddir, 'test.pkl')
n_tr = args.ntr
n_val = 3000 # don't need to vary
n_te = 10000 # don't need to vary

experiment = SYNTHETIC_EXPERIMENTS[args.exp]
train_set, val_set, test_set = experiment.get_train_val_test()

train_data = experiment.gen_data(train_set, args.override_data, savename_tr, n_tr)
val_data = experiment.gen_data(val_set, args.override_data, savename_val, n_val)
test_data = experiment.gen_data(test_set, args.override_data, savename_te, n_te)

######################################### run models ###################################
experiment.run(args, train_data, val_data)
