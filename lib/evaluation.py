import torch
import numpy as np
import glob
from lib.train import Train

class Evaluation(object):

    def __init__(self, net, data, use_gpu=False):
        self.net = net
        self.data = data
        self.use_gpu = use_gpu

    def confusion(self, n_categories, n_confusion=10000):
        # only for classification problem
        confusion = torch.zeros(n_categories, n_categories)

        # Go through a bunch of examples
        for i in range(n_confusion):

            d = self.data.next_batch(1)
            d = list(d)            
            if self.use_gpu is not False:
                d[0], d[1] = d[0].cuda(self.use_gpu), d[1].cuda(self.use_gpu)

            y = d[1] # seq_len x bs which is seq_len x 1
            output = self.net.eval_forward(*d) # seq_len x bs x output_size
            if len(output.shape) < 3: # seq_len x bs, single output
                max_dim = 1
            else: # seq_len x bs x output_size, multi output
                max_dim = 2
            _, ans = torch.max(output, max_dim)
            for j in range(len(y)):
                confusion[y[j].item()][ans[j].item()] += 1
        return confusion

def plot_confusion(confusion):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    n_confusion = confusion.sum()
    n_categories = confusion.shape[0]
    
    # print accuracy
    n_correct = sum([confusion[i][i] for i in range(n_categories)])
    print('accuracy is %.2f%%' % (n_correct / n_confusion * 100))
    
    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    all_categories = [str(i) for i in range(n_categories)]
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.ylabel('true label')
    plt.xlabel('predicted')

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
        
def plot_train_val(patterns, fontsize=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    dummy_trainer = Train(None, None, None, None)
    for pattern in patterns:
        for cpt_fn in glob.glob(pattern):
            cpt = torch.load(cpt_fn)
            name = cpt_fn.split('.')[0].split('/')[-1]
            l = cpt['train_losses']
            dummy_trainer.all_losses = l
            plt.plot(dummy_trainer.smooth_loss(), label=name)
            
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('training loss', fontsize=fontsize)
    plt.grid()
    plt.show()
    
    for pattern in patterns:
        for cpt_fn in glob.glob(pattern):
            cpt = torch.load(cpt_fn)
            name = cpt_fn.split('.')[0].split('/')[-1]
            l = cpt['val_accs']
            dummy_trainer.val_accs = l
            plt.plot(dummy_trainer.smooth_valacc(), label=name)

    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('validation acc', fontsize=fontsize)
    plt.grid()
    plt.show()
    
def plot_fill(lines, x=None, color='b', label='default'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    for l in lines:
        if x is not None:
            plt.plot(x, l, color=color, alpha=0.2)
        else:
            plt.plot(l, color=color, alpha=0.2)
    
    # lines may not have the same length
    max_length = max([len(l) for l in lines])
    middle_line = np.zeros(max_length)    
    for i in range(max_length):
        middle_line[i] = np.percentile([l[i] for l in lines if len(l) > i], 50)
        
    if x is not None:
        plt.plot(x, middle_line, color=color, label=label)
    else:
        plt.plot(middle_line, color=color, label=label)
    
def get_train_val_curves(pattern):
    dummy_trainer = Train(None, None, None, None)
    tr_curves = []
    val_curves = []
    name = ""
    for cpt_fn in glob.glob(pattern):
        cpt = torch.load(cpt_fn)
        name = cpt_fn.split('.')[0].split('/')[-1]
        dummy_trainer.all_losses = cpt['train_losses']
        dummy_trainer.val_accs = cpt['val_accs']
        
        tr_curves.append(dummy_trainer.smooth_loss())
        val_curves.append(dummy_trainer.smooth_valacc())
    return tr_curves, val_curves, name

def plot_train_val_multiple(patterns, colors=['blue', 'orange', 'green', 'red', 
                                              'purple', 'brown', 'pink', 'gray'], 
                            fontsize=15):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker
    
    for i, pattern in enumerate(patterns):
        tr_curves, val_curves, name = get_train_val_curves(pattern)
        if name is not "":
            plot_fill(tr_curves, label=name, color=colors[i])
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('training loss', fontsize=fontsize)
    plt.grid()
    plt.show()
    
    for i, pattern in enumerate(patterns):
        tr_curves, val_curves, name = get_train_val_curves(pattern)
        if name is not "":
            plot_fill(val_curves, label=name, color=colors[i])
    plt.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    plt.title('validation acc', fontsize=fontsize)
    plt.grid()
    plt.show()

        
                                
        


