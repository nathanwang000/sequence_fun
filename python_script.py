# a script to run main.py
import os
from lib.experiment import SYNTHETIC_EXPERIMENTS
import warnings, argparse

exp_names = SYNTHETIC_EXPERIMENTS.keys()

#### parse argument
parser = argparse.ArgumentParser(description='quick run of experiment for RNN-mtl')
parser.add_argument('--debug', action="store_true",
                    help='debug mode will just run one instance')
parser.add_argument('--exp', default="partial_sum", type=str,
                    choices=exp_names,
                    help='experiment name to run')
parser.add_argument('--nhidden', default=30, type=int,
                    help='number of hidden units')
parser.add_argument('--smdir', default='mo_models', type=str,
                    help='directory to save model')
parser.add_argument('--sddir', default='sequence_data', type=str,
                    help='directory to save data')
parser.add_argument('--override_data', action='store_true',
                    help='whether override data')
parser.add_argument('--override_model', action='store_true',
                    help='whether override model')
parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], 
                    help='which optimizer to use')
parser.add_argument('--num_layers', default=1, type=int,
                    help='number of layers for lstm')


args = parser.parse_args()

# naming convention: modelname_ntr/run_number
exp_name = args.exp # "partial_sum", "scarce"
nhidden = args.nhidden
model_dir_base = "%s/%s" % (args.smdir, exp_name)
data_dir_base = "%s/%s" % (args.sddir, exp_name)

use_gpu = True
niters = 10000
n_save_model = 10
if not args.debug:
    seeds = [42, 8, 3]
    ntrs  = [1000, 10000] #[1000, 500, 100, 5000, 10000]
    model_names = [
        'RNN_LSTM_2layers',
        'RNN_LSTM',
        'RNN_ILSTM',
        'RNN_LSTM_MoW',
        'RNN_SLSTM',
        'RNN_LSTM_MoO']
    #'RNN_MLP', 'RNN_IMLP', 'RNN_MLP_MoW', 'RNN_SMLP']
else: # debug run
    model_dir_base = model_dir_base + "_debug"
    n_save_model = 10
    args.override_model = True
    seeds = [42]
    ntrs = [100]
    model_names = ['RNN_LSTM_2layers'] #['RNN_LSTM_MoO'] #['RNN_LSTM'] 

for i, seed in enumerate(seeds):
    for ntr in ntrs:
        # todo: add search on hyperparameters: e.g. optmizer, weight_decay
        for model_name in model_names:
            command = []
            
            command.append("python main.py")
            command.append('-a %s' % model_name)
            command.append('--exp %s' % exp_name)
            command.append('--nhidden %d' % nhidden)
            command.append('--num_layers %d' % args.num_layers)            
            command.append('--optimizer %s' % args.optimizer)
            if args.override_data:
                command.append("--override_data")
            if args.override_model:
                command.append("--override_model")                
            if use_gpu:
                command.append("--use_gpu")
            command.append("--seed %d" % seed)
            command.append("--ntr %d" % ntr)
            command.append("--niters %d" % niters)
            command.append("--n_save_model %d" % n_save_model)            

            smdir = os.path.join(model_dir_base + "_%d" % ntr, str(i))
            sddir = os.path.join(data_dir_base + "_%d" % ntr) # same data across runs
            command.append('--smdir %s' % smdir)
            command.append('--sddir %s' % sddir)            
            
            command = " ".join(command)
            print(command)
            os.system(command)
    
