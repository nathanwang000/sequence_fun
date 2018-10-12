import torch
import numpy as np

class Evaluation(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data

    def confusion(self, n_categories, n_confusion=10000):
        # only for classification problem
        confusion = torch.zeros(n_categories, n_categories)

        # Go through a bunch of examples and record which are correctly guessed
        for i in range(n_confusion):
            d = self.data.random_batch(1)
            y = d[1]
            output = self.net.eval_forward(*d)
            _, ans = torch.max(output, 1)
            confusion[y.item()][ans.item()] += 1
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
        

        
                                
        


