
import numpy as np
from numpy.core.defchararray import title
import argparse


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.4)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-3)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
args = parser.parse_args()


EPOCH = 100


model = np.load('CIFAR10_Result/{me}_{nt}_{nr}_result.npz'.format(
    me=args.method, nt=args.noise_type, nr=args.noise_rate),)


TrainLoss = model['train_loss_result']
TrainAccuracy = model['train_acc_result']
TestLoss = model['test_loss_result']
TestAccuracy = model['test_acc_result']
ValidationLoss = model['val_loss_result']


id = np.argmin(ValidationLoss)
test_acc_max = TestAccuracy[id]
print('Best Accuracy', test_acc_max)
