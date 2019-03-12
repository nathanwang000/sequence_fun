# Some part of the code was referenced from below.
# https://github.com/pytorch/examples/tree/master/word_language_model 
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_

from torch.utils.data import TensorDataset
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
parser.add_argument('-seed', type=int,
                    help='random seed', default=42)
parser.add_argument('-epoch', type=int,
                    help='#epoch to run', default=10)
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
train_errors = []
val_errors = []
test_errors = []
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
            inputs = inputs.to(device)
            targets = targets.to(device)
            bs = inputs.size(0)
            
            # Set initial hidden and cell states
            states = (torch.zeros(num_layers * num_directions, bs,
                                  hidden_size).to(device),
                      torch.zeros(num_layers * num_directions, bs,
                                  hidden_size).to(device))

            # Forward pass
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1))

            y_true.extend([t.item() for t in targets])
            y_score.extend([t.item() for t in nn.functional.softmax(outputs, 1)[:,1]]) 
            loss_meter.update(loss.item(), bs)

    auc = roc_auc_score(y_true, y_score)
    model.train()
    return loss_meter.avg, auc

name =  "{}/{}-{}-{}-{}-{}-{}^{:.2f}^{}".format(args.s,
                                                'lstm',
                                                args.lr,
                                                args.d,
                                                args.nhidden,
                                                args.nlayer,
                                                args.bs,
                                                -1, # placeholder
                                                run_id)

def save():
    # Save the model checkpoint
    if os.path.exists('{}.train_losses'.format(name)):
        os.system('rm {}.train_losses'.format(name))
        os.system('rm {}.train_errors'.format(name))
        os.system('rm {}.val_errors'.format(name))
        os.system('rm {}.test_errors'.format(name))
        os.system('rm {}.opt_track'.format(name))
        os.system('rm {}.ckpt'.format(name))

    joblib.dump(train_losses, name + ".train_losses")
    joblib.dump(train_errors, name + ".train_errors")
    joblib.dump(val_errors, name + ".val_errors")
    joblib.dump(test_errors, name + ".test_errors")
    joblib.dump(opt_recorder.tracker, name + ".opt_track")
    torch.save(model.state_dict(), name + '.ckpt')

# Load mimic3 dataset
def load_data(path):
    data = np.load(path)
    x = torch.from_numpy(data['data']).float()
    y = torch.from_numpy(data['labels'])
    return TensorDataset(x, y)

train_dataset = load_data('/data1/mimic/jeeheh_IHMnpz/IHM_train.npz')
val_dataset = load_data('/data1/mimic/jeeheh_IHMnpz/IHM_val.npz')
test_dataset = load_data('/data1/mimic/jeeheh_IHMnpz/IHM_test.npz')

# Hyper-parameters
embed_size = 76
hidden_size = args.nhidden
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

# RNN based language model
class RNN(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, num_layers,
                 bidirectional=False, dropout=0):
        super(RNN, self).__init__()
        self.dropout = dropout
        if dropout != 0:
            self.drop = nn.Dropout(p=dropout)

        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            input_size = hidden_size * 2
        else:
            input_size = hidden_size
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, x, h):
        # Forward propagate LSTM
        out, (h, c) = self.lstm(x, h)
        
        # last time hidden
        out = out[:, -1, :]
        #out = out.reshape(out.size(0)*out.size(1), out.size(2))

        if self.dropout != 0:
            out = self.drop(out)
            
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

model = RNN(output_size, embed_size, hidden_size, num_layers,
            bidirectional).to(device)

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
opt_recorder = OptRecorder(optimizer)    

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states] 

# Train the model
for epoch in range(num_epochs):
    
    for step, (inputs, targets) in enumerate(train_loader):
        # Get mini-batch inputs and targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        bs = inputs.size(0)
        
        # Set initial hidden and cell states
        states = (torch.zeros(num_layers * num_directions, bs, hidden_size).to(device),
                  torch.zeros(num_layers * num_directions, bs, hidden_size).to(device))
        
        # Forward pass
        outputs, states = model(inputs, states)
        loss = criterion(outputs, targets.reshape(-1))
        
        # Backward and optimize
        model.zero_grad()
        loss.backward()
        optimizer.step()

        if step in [0, len(train_loader) // 2]:
            print ('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, step, len(train_loader), loss.item()))
            tr_loss, tr_error = eval_loader(model, train_loader)
            _, val_error = eval_loader(model, val_loader)
            _, test_error = eval_loader(model, test_loader)

            print('(tr auc, val auc, test auc): ({:.3f}, {:.3f}, {:.3f})'.format(
                tr_error, val_error, test_error
            ))

            train_losses.append(tr_loss)
            train_errors.append(tr_error)            
            val_errors.append(-val_error) # best one need to be negative
            test_errors.append(test_error)
            opt_recorder.record()
            save()

# Save the model checkpoints
save()
