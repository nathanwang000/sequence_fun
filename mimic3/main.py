# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import TensorDataset
from MTL_RNN.lib.model import RNN_LSTM, RNN_LSTM_MoW, RNN_LSTM_MoO, RNN_LSTM_MoO_time
from MTL_RNN.lib.model import RNN_ILSTM
from sklearn.metrics import roc_auc_score
import random, string, os, tqdm
import MTL_OPT.lib.optimizer as optimizers
from MTL_OPT.lib.utils import AverageMeter, OptRecorder, random_string
from MTL_OPT.lib.utils import random_split_dataset
import argparse
from sklearn.externals import joblib
parser = argparse.ArgumentParser(description="opt")
parser.add_argument('-o', type=str,
                    help='optimizer', default='torch.optim.Adam')
parser.add_argument('-m', type=str,
                    help='model name', default='RNN_LSTM')
parser.add_argument('-seed', type=int,
                    help='random seed', default=42)
parser.add_argument('-epoch', type=int,
                    help='#epoch to run', default=30)
parser.add_argument('-lr', type=float,
                    help='learning rate', default=0.001)
parser.add_argument('-s', type=str,
                    help='save directory', default='train_loss')
parser.add_argument('-b', action='store_true', help='bidirectional')
parser.add_argument('-d', type=float, help='dropout', default=0)
parser.add_argument('-nhidden', type=int, help='hidden size', default=16)
parser.add_argument('-nlayer', type=int, help='layers for lstm', default=2)
parser.add_argument('-bs', type=int, help='batch size', default=100)

args = parser.parse_args()
print(args)
os.system('mkdir -p {}'.format(args.s))
run_id = random_string()

train_losses = []
train_aucs = []
val_aucs = []
test_aucs = []
val_best = 0
torch.set_num_threads(1)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

def eval_loader(model, loader):
    model.eval()

    loss_meter = AverageMeter()    
    y_true, y_score = [], []
    with torch.no_grad():
        for inputs, targets in tqdm.tqdm(loader):
            # Get mini-batch inputs and targets
            # x: (bs, seq_len, d) => (seq_len, bs, d)
            inputs = inputs.permute(1,0,2)
            inputs = inputs.to(device)
            targets = targets.to(device)
            bs = inputs.size(1)
            input_lengths = [seq_length] * bs            
            
            # Set initial hidden and cell states
            states = model.initHidden(batch_size=bs)
            
            # Forward pass
            outputs, states = model(inputs, states, input_lengths)
            outputs = outputs[-1] # last step (bs, 2)
            loss = criterion(outputs, targets.reshape(-1))

            y_true.extend([t.item() for t in targets])
            y_score.extend([t.item() for t in nn.functional.softmax(outputs, 1)[:,1]]) 
            loss_meter.update(loss.item(), bs)

    auc = roc_auc_score(y_true, y_score)
    model.train()
    return loss_meter.avg, auc

name =  "{}/{}-{}-{}-{}-{}-{}^{:.2f}^{}".format(args.s,
                                                args.m,
                                                args.lr,
                                                args.d,
                                                args.nhidden,
                                                args.nlayer,
                                                args.bs,
                                                -1, # placeholder
                                                run_id)
print(name)
def save():
    # Save the model checkpoint
    if os.path.exists('{}.train_losses'.format(name)):
        os.system('rm {}.train_losses'.format(name))
        os.system('rm {}.train_errors'.format(name))
        os.system('rm {}.val_errors'.format(name))
        os.system('rm {}.test_errors'.format(name))
        # os.system('rm {}.opt_track'.format(name))
        os.system('rm {}.ckpt'.format(name))

    joblib.dump(train_losses, name + ".train_losses")
    joblib.dump(train_aucs, name + ".train_errors")
    joblib.dump(val_aucs, name + ".val_errors")
    joblib.dump(test_aucs, name + ".test_errors")
    # joblib.dump(opt_recorder.tracker, name + ".opt_track")
    torch.save(model.state_dict(), name + '.ckpt')

# Load mimic3 dataset
def load_data(path):
    data = np.load(path)
    x = torch.from_numpy(data['data']).float()
    # x: (N, seq_len, d)
    y = torch.from_numpy(data['labels'])
    return TensorDataset(x, y)

train_dataset = load_data('/data1/mimic/jeeheh_IHMnpz/IHM_train.npz')
val_dataset = load_data('/data1/mimic/jeeheh_IHMnpz/IHM_val.npz')
test_dataset = load_data('/data1/mimic/jeeheh_IHMnpz/IHM_test.npz')

# Hyper-parameters
embed_size = 76
hidden_size = args.nhidden # 16
output_size = 2 # binary classification
num_layers = args.nlayer
num_epochs = args.epoch
batch_size = args.bs
seq_length = 48
learning_rate = args.lr
bidirectional = args.b
num_directions = 2 if bidirectional else 1

# loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=batch_size, 
                                         shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)

# model
model = eval(args.m)(embed_size, hidden_size, output_size,
                     num_layers, num_directions, args.d)
if args.m in ('RNN_LSTM_MoW', 'RNN_LSTM_MoO', 'RNN_LSTM_MoO_time'):
    model.setKT(2, seq_length)
elif args.m == 'RNN_ILSTM':
    model.set_max_length(seq_length)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
if '(' in args.o:
    opt = args.o
    alpha_index = opt.find('(')
    alphas = eval(opt[alpha_index:])
    optimizer = eval(opt[:alpha_index])(model.parameters(), lr=learning_rate,
                                        alphas=alphas)
else:
    optimizer = eval(args.o)(model.parameters(), lr=learning_rate)
# opt_recorder = OptRecorder(optimizer)    

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Train the model
for epoch in range(num_epochs):
    
    for step, (inputs, targets) in enumerate(train_loader):
        # Get mini-batch inputs and targets
        # x: (bs, seq_len, d) => (seq_len, bs, d)
        inputs = inputs.permute(1,0,2)
        inputs = inputs.to(device)
        targets = targets.to(device)
        bs = inputs.size(1)
        input_lengths = [seq_length] * bs
        
        # Set initial hidden and cell states
        states = model.initHidden(batch_size=bs)
        
        # Forward pass
        outputs, states = model(inputs, states, input_lengths)
        outputs = outputs[-1] # last step
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        model.after_backward()
        optimizer.step()

        if step in [0, len(train_loader) // 2]:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, step, len(train_loader), loss.item()))
            tr_loss, tr_auc = eval_loader(model, train_loader)
            _, val_auc = eval_loader(model, val_loader)
            _, test_auc = eval_loader(model, test_loader)

            print('(tr auc, val auc, test auc): ({:.3f}, {:.3f}, {:.3f})'.format(
                tr_auc, val_auc, test_auc
            ))

            train_losses.append(tr_loss)
            train_aucs.append(tr_auc)            
            val_aucs.append(-val_auc) # negative b/c it is the error criteria
            if val_best < val_auc:
                val_best = val_auc
                torch.save(model.state_dict(), name + '.ckpt_best_{}'.format(epoch))
            test_aucs.append(test_auc)
            # opt_recorder.record()
            save()

# Save the model checkpoints
save()
