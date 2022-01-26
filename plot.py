
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

EPOCH = 100
def save_fig(images,labels,nt_1 ,data,targets, nt_2):
    images = images.to('cpu').detach().numpy().copy()
    data = data.to('cpu').detach().numpy().copy()
    #targets = targets.to('cpu').detach().copy()
    A =images
    X = data
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    sns.scatterplot(A[:, 0], A[:, 1], hue=labels, ax=ax1)
    ax1.set_title(nt_1)
    sns.scatterplot(X[:, 0], X[:, 1], hue=targets, ax=ax2)
    ax2.set_title(nt_2)
    plt.savefig(
        "nonlinear_{a}_and_{b}.png".format(a=nt_1, b=nt_2))
    
