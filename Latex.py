
import argparse
import torch
import numpy as np
import models

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
                    help='overall corruption rate, should be less than 1', default=0.5)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.1)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
args = parser.parse_args()

dataset = ['mnist', 'fmnist', 'cifar10', 'cifar100']
methods = ['CE', 'CDR', 'Trimming']#,'Trimming_minibatch']
noise_rate =[0.01,0.05, 0.1,0.2]

for _ , method in enumerate (methods):
    print(method, end = " ")
    for i, noise in enumerate(noise_rate):
        best_acc = np.load('best_acc_{d}/{nt}_{nr}_{m}_result.npy'.format(
            d=args.dataset,nt=args.noise_type, m=method, nr=noise))
        avg = np.mean(best_acc, axis=0)
        std = np.std(best_acc, axis=0)
        print('& $' , end=' ')
        print('%.4f \\pm %.4f' % (avg, std),  end=' ')
        print('$', end=' ')
    print('\\\\')

# print('Trimming_change', end=" ")
# for i, noise in enumerate(noise_rate):
#     best_acc = np.load('best_acc_trimming_change/{nt}_{nr}_{m}_result.npy'.format(
#         d=args.dataset, nt=args.noise_type, m='Trimming', nr=noise))
#     avg = np.mean(best_acc, axis=0)
#     std = np.std(best_acc, axis=0)
#     print('& $', end=' ')
#     print('%.4f \\pm %.4f' % (avg, std),  end=' ')
#     print('$', end=' ')
# print('\\\\')

print('Trimming_minibatch', end=" ")
for i, noise in enumerate(noise_rate):
    best_acc = np.load('best_acc_{d}/{nt}_{nr}_{m}_result.npy'.format(
        d=args.dataset, nt=args.noise_type, m='Trimming_minibatch', nr=noise))
    avg = np.mean(best_acc, axis=0)
    std = np.std(best_acc, axis=0)
    print('& $', end=' ')
    print('%.4f \\pm %.4f' % (avg, std),  end=' ')
    print('$', end=' ')
print('\\\\')

# noise_rate = [0.1]
# for i, noise in enumerate(noise_rate):
#     best_acc = np.load('best_acc_{d}/{nt}_{nr}_{m}_result.npy'.format(
#         d=args.dataset, nt=args.noise_type, m='CE', nr=noise))
#     avg = np.mean(best_acc, axis=0)
#     std = np.std(best_acc, axis=0)
#     print('& $', end=' ')
#     print('%.4f \\pm %.4f' % (avg, std),  end=' ')
#     print('$', end=' ')
# print('\\\\')
