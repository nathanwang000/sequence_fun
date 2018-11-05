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
args = parser.parse_args()

# naming convention: modelname_1000/run_number
exp_name = args.exp # "partial_sum", "scarce"
model_dir_base = "mo_models/%s" % exp_name
data_dir_base = "sequence_data/%s" % exp_name

use_gpu = True
niters = 10000
n_save_model = 10
if not args.debug:
    seeds = [42, 8, 3]
    ntrs  = [100, 500, 1000, 5000, 10000]
    model_names = ['RNN_LSTM', 'RNN_LSTM_MoW', 'RNN_SLSTM', 'RNN_ILSTM', \
                   'RNN_MLP', 'RNN_IMLP', 'RNN_MLP_MoW', 'RNN_SMLP']
else: # debug run
    model_dir_base = model_dir_base + "_debug"
    n_save_model = 100
    seeds = [42]
    ntrs = [100]
    model_names = ['RNN_MLP_MoW']

for i, seed in enumerate(seeds):
    for ntr in ntrs:
        # todo: add search on hyperparameters: e.g. optmizer, weight_decay
        for model_name in model_names:
            command = []
            
            command.append("python main.py")
            command.append('-a %s' % model_name)
            command.append('--exp %s' % exp_name)
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
    
