import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import title

#noise_type = 'symmetric'
#noise_type = 'asymmetric'
noise_type = 'instance'
MAX_NUM = 10
EXP_NUM = 10
EPOCH = 100


def save_fig(x, y, train, loss):
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

    label1 = 'noise_0.4'
    label2 = 'noise_0.2'
    fig1 = plt.figure()
    fig1.suptitle(title)
    plt.plot(range(EPOCH), x, 'r-', label=label1)
    plt.plot(range(EPOCH), y, 'b-', label=label2)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel(y_label)
    fig1.savefig("Instance/{}.png".format(label))


model = np.load('MNIST_Result/{}_noise_result.npz'.format(noise_type))

TrainLoss = model['train_loss_result']
TrainAccuracy = model['train_acc_result']
TestLoss = model['test_loss_result']
TestAccuracy = model['test_acc_result']

save_fig(TrainLoss[0], TrainLoss[1], 1, 1)
save_fig(TrainAccuracy[0], TrainAccuracy[1], 1, 0)
save_fig(TestLoss[0], TestLoss[1],0, 1)
save_fig(TestAccuracy[0], TestAccuracy[1],0, 0)
