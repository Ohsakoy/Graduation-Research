import matplotlib.pyplot as plt
import numpy as np
from numpy.core.defchararray import title

noise_type = 'symmetric'
#noise_type = 'asymmetric'
#noise_type = 'instance'
MAX_NUM = 2
EXP_NUM = 10
EPOCH = 100


model = np.load('MY_REL_Result/{}_noise_result.npz'.format(noise_type))


TrainLoss = model['train_loss_result']
TrainAccuracy = model['train_acc_result']
TestLoss = model['test_loss_result']
TestAccuracy = model['test_acc_result']
ValidationLoss = model['val_loss_result']

for num in range(MAX_NUM):
    id = np.argmin(ValidationLoss[num])
    print(id)
    test_acc_max = TestAccuracy[num][id]

    print('Best Accuracy', test_acc_max)
