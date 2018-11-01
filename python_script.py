# a script to run main.py
import os
# todo: add parsing arguments

seeds = [42, 8, 3, 0, 26]
ntrs  = [100, 500] #[1000, 5000, 10000]
use_gpu = True
model_names = ['RNN_LSTM', 'RNN_LSTM_MoW', 'RNN_SLSTM', 'RNN_ILSTM', \
               'RNN_MLP', 'RNN_IMLP', 'RNN_MLP_MoW', 'RNN_SMLP']

# naming convention: modelname_n1000/run_number
model_dir_base = "mo_models/scarce"
data_dir_base = "sequence_data/scarce"

for i, seed in enumerate(seeds):
    for ntr in ntrs:
        for model_name in model_names:
            command = []
            
            command.append("python main.py")
            command.append('-a %s' % model_name)
            if use_gpu:
                command.append("--use_gpu")
            command.append("--seed %d" % seed)
            command.append("--ntr %d" % ntr)

            smdir = os.path.join(model_dir_base + "_%d" % ntr, str(i))
            sddir = os.path.join(data_dir_base + "_%d" % ntr, str(i))
            command.append('--smdir %s' % smdir)
            command.append('--sddir %s' % sddir)            
            
            command = " ".join(command)
            print(command)
            os.system(command)
    
