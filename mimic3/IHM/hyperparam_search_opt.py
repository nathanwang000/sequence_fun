import os
import signal
import atexit
import subprocess
from itertools import product
import time
import argparse

parser = argparse.ArgumentParser(description="relaxed rnn")
parser.add_argument('-s', type=str,
                    help='save directory', default='train_loss')
parser.add_argument('-p', type=float, help='pct training data to use', default=1)
parser.add_argument('-c', type=int, help='n concurrent process', default=7)
args = parser.parse_args()
print(args)

procs = []
n_concurrent_process = args.c
save_dir = args.s + "_opt"

opts = ['optimizers.AdaBound', 'optimizers.CrossBound', 'optimizers.CrossAdaBound',
        'optimizers.Swats', 'torch.optim.SGD', 'torch.optim.Adam', 'optimizers.Diff']
models = ['RNN_LSTM']
lrs = [1e-3, 1e-4, 100, 10, 1, 0.1, 0.01]
dropouts = [0.3]
nhiddens = [32]
nlayers = [2]
batch_sizes = [100]

for lr, d, bs, nhidden, nlayer, m, opt in product(lrs, dropouts, batch_sizes,
                                                  nhiddens, nlayers, models, opts):
    commands = ["python", "main.py", "-lr", str(lr), "-d", str(d), "-bs", str(bs),
                "-nhidden", str(nhidden), "-nlayer", str(nlayer), "-m", m,
                "-s", save_dir, "-p", str(args.p), "-o", opt,
                "-name", opt.split('.')[-1]]
    procs.append(subprocess.Popen(commands))

    while True:
        new_procs = []
        while len(procs) > 0:
            p = procs.pop()
            if p.poll() == None: # active
                new_procs.append(p)

        procs = new_procs
        #print("len process", len(procs), len(new_procs))
        if len(procs) >= n_concurrent_process:
            time.sleep(3)
        else:
            break # fetch next
                
for p in procs:
    p.wait()
    
def kill_procs():
    for p in procs:
        if p.pid is None:
            pass
        else:
            os.kill(p.pid, signal.SIGTERM)
        
atexit.register(kill_procs)

