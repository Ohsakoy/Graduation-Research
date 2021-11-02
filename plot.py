import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import title


EPOCH = 100

class Plot_fig():
    def save_fig(avg, std, train, loss):
        if train == 1 and loss == 1:
            title = 'TrainLoss'
            label = 'train_loss'
            col = 'r-'
            y_label = 'loss'
        elif train == 1 and loss == 0:
            title = 'TrainAccuracy'
            label = 'train_accuracy'
            col = 'r-'
            y_label = 'accuracy'
        elif train == 0 and loss == 1:
            title = 'TestLoss'
            label = 'test_loss'
            col = 'b-'
            y_label = 'loss'
        elif train == 0 and loss == 0:
            title = 'TestAccuracy'
            label = 'test_accuracy'
            col = 'b-'
            y_label = 'accuracy'

        fig1 = plt.figure()
        fig1.suptitle(title)
        plt.plot(range(EPOCH), avg, col, label=label)
        plt.fill_between(range(EPOCH), avg-std, avg+std, alpha=0.2)
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel(y_label)
        fig1.savefig("AvgStdplot/{}_ms.png".format(label))
