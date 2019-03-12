import os
import signal
import atexit
import subprocess
from itertools import product
import time

procs = []
n_concurrent_process = 5
nruns = 10

models = ['RNN_LSTM','RNN_LSTM_MoO','RNN_LSTM_MoW','RNN_LSTM_MoO_time','RNN_ILSTM']
lrs = [0.001]
dropouts = [0.3]
nhiddens = [32, 64, 8, 16]
nlayers = [2, 4, 1, 8]
batch_sizes = [100]
# 16 settings for 5 models, then 160 min per model, total of 800 / 60 ~= 13 hours
# at most 16 hours: bottle neck is RNN_MoW

for i, m in product(range(nruns), models):
    if m == 'RNN_LSTM':
        commands = ["python", "main.py", "-s", "repeat", "-lr", "0.001",
                    "-d", '0.3', "-bs", '100',
                    "-nhidden", "64", "-nlayer", '2', "-m", m]
    elif m == 'RNN_LSTM_MoO'

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

