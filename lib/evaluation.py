import torch
import numpy as np

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
        

        
                                
        


