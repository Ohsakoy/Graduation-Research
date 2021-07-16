import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import title
from plot import Plot_fig

MAX_NUM = 10
EXP_NUM = 10
EPOCH = 100


def avg_std_method(x):
    avg = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return avg, std


model = np.load('F_MNIST_model_1-10/model_result.npz')
print(model.files)

TrainLoss = model['train_loss_result']
TrainAccuracy = model['train_acc_result']
TestLoss = model['test_loss_result']
TestAccuracy = model['test_acc_result']

avgTrainLoss, stdTrainLoss = avg_std_method(TrainLoss)
avgTrainAccuracy, stdTrainAccuracy = avg_std_method(TrainAccuracy)
avgTestLoss, stdTestLoss = avg_std_method(TestLoss)
avgTestAccuracy, stdTestAccuracy = avg_std_method(TestAccuracy)

plot_fig = Plot_fig()

plot_fig.save_fig(avgTrainLoss, stdTrainLoss, 1, 1)
plot_fig.save_fig(avgTrainAccuracy, stdTrainAccuracy, 1, 0)
plot_fig.save_fig(avgTestLoss, stdTestLoss, 0, 1)
plot_fig.save_fig(avgTestAccuracy, stdTestAccuracy, 0, 0)
