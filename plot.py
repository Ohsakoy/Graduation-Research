
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
#from tsnecuda import TSNE


EPOCH = 100
def save_fig(images,labels,data,targets):
    images = images.to('cpu').detach().numpy().copy()
    data = data.to('cpu').detach().numpy().copy()
    #targets = targets.to('cpu').detach().copy()
    A =images
    X = data
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    sns.scatterplot(A[:, 0], A[:, 1], hue=labels, ax=ax1)
    #ax1.set_title(nt_1)
    sns.scatterplot(X[:, 0], X[:, 1], hue=targets, ax=ax2)
    #ax2.set_title(nt_2)
    plt.savefig(
        "nonlinear.png")
    
def t_sne(images,labels):
    images = images.to('cpu').detach().numpy().copy()
    #labels = labels.to('cpu').detach().numpy().copy()
    images2d = TSNE(n_components=2).fit_transform(images.reshape(60000,-1))
    # f, ax = plt.subplots(1, 1, figsize=(10, 10))
    # for i in range(10):
    #     target = images2d[labels == i]
    #     ax.scatter(x=target[:, 0], y=target[:, 1], label=str(i), alpha=0.5)
    # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    # Create the figure
    fig = plt.figure( figsize=(8,8) )
    ax = fig.add_subplot(1, 1, 1, title='TSNE' )

    # Create the scatter
    ax.scatter(
        x=images2d[:,0],
        y=images2d[:,1],
        c=labels,
        cmap=plt.cm.get_cmap('Paired'),
        alpha=0.4,
        s=0.5)
    
    plt.savefig('mnist_0.png')
    
def plot_loss(train1, train2,train3, epoch):
    f, (ax1,ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
    ax1.plot(range(epoch), train1)
    ax1.set_title("CE")
    ax2.plot(range(epoch), train2)
    ax2.set_title("CDR")
    ax3.plot(range(epoch), train3)
    ax3.set_title('Trimming_minibatch')
    
    plt.grid()
    plt.savefig('train_val_loss_CDR.png')