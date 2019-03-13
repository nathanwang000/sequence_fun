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
parser.add_argument('-c', type=int, help='n concurrent process', default=20)
args = parser.parse_args()
print(args)

procs = []
n_concurrent_process = args.c
save_dir = args.s + "_{:.1f}".format(args.p) + "_test"

models = ['RNN_LSTM','RNN_LSTM_MoO','RNN_LSTM_MoW','RNN_LSTM_MoO_time','RNN_ILSTM']
for i, m in product(range(10), models):
    # lstm: mimic3/train_loss/RNN_LSTM-0.001-0.3-16-1-100^-1.00^2O8NM
    # lstm-mow: mimic3/train_loss/RNN_LSTM_MoW-0.001-0.3-64-2-100^-1.00^3S3OD
    # lstm-moot: mimic3/train_loss/RNN_LSTM_MoO_time-0.001-0.3-64-2-100^-1.00^K8LVS
    # lstm-moo: mimic3/train_loss/RNN_LSTM_MoO-0.001-0.3-32-4-100^-1.00^6XU80
    # lstm-ind: mimic3/train_loss/RNN_ILSTM-0.001-0.3-32-2-100^-1.00^FXNAZ
    commands = ["python", "main.py", "-lr", '0.001', "-d", '0.3', "-bs", '100',
                "-m", m, "-s", save_dir, "-p", str(args.p), "-test"]
    
    if m == 'RNN_LSTM':
        commands.extend(["-nhidden", '16', "-nlayer", '1'])        
    elif m == 'RNN_LSTM_MoO':
        commands.extend(["-nhidden", '32', "-nlayer", '4'])
    elif m == 'RNN_LSTM_MoW':
        commands.extend(["-nhidden", '64', "-nlayer", '2'])        
    elif m == 'RNN_LSTM_MoO_time':
        commands.extend(["-nhidden", '64', "-nlayer", '2'])                
    elif m == 'RNN_ILSTM':
        commands.extend(["-nhidden", '32', "-nlayer", '2'])                        

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

