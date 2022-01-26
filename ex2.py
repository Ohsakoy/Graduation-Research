import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
import numpy as np
import time
from noisy_dataset import NoisyDataset
from add_noise_split_crust import Noise_dataset_split_crust
import tools
from REL_model import LeNet
from REL_model import Logistic
from REL_model import NeuralNetwork
from REL_model import NeuralNetLinear
from cdr_method import CDR
from crust_method import CRUST
from ce_method import CE
from trimming_method import Trimming
import eval_on_holdout
import plot
from lazyGreedy import lazy_greedy_heap
from fl_mnist import FacilityLocationMNIST
import REL_model
import models
from covid_ct_dataset import CovidCTDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=3000)
parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--weight_decay', type=float, help='l2', default=0.01)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
parser.add_argument('--outlier_ratio', type=float, default=0.2)
parser.add_argument('--optim', type=str, default='sgd')

args = parser.parse_args()

device = torch.device("cpu")

results = np.load('{d}_result/{model}_{optim}_result.npz'.format(
    d=args.dataset, model=args.model_name, optim=args.optim))

TrainLoss = results['train_loss_result']
TrainAccuracy = results['train_acc_result']
TestLoss = results['test_loss_result']
TestAccuracy = results['test_acc_result']
ValidationLoss = results['val_loss_result']

best_acc = TestAccuracy[np.argmin(ValidationLoss)]
best_num =np.argmin(ValidationLoss)

print('epoch %d, train_loss: %.8f train_acc: %f val_loss: %f test_loss: %f test_acc: %f' %
      (args.epoch, TrainLoss[args.epoch-1],TrainAccuracy[args.epoch-1],ValidationLoss[args.epoch-1],TestLoss[args.epoch-1],TestAccuracy[args.epoch-1]))

print('epoch %d, train_loss: %.8f train_acc: %f val_loss: %f test_loss: %f test_acc: %f' %
      (best_num+1, TrainLoss[best_num], TrainAccuracy[best_num], ValidationLoss[best_num], TestLoss[best_num], TestAccuracy[best_num]))

print(best_acc)