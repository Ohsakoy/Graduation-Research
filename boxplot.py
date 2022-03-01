import argparse
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='PyTorch Training')
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


trimming_loss = np.zeros(10)
ce_with_random = np.zeros(10)
ce_with_outlier = np.zeros(10)
ce_with_true = np.zeros(10)

for i in range(10):
    results = np.load('{d}_result/{me}_{nt}_{nr}_result_{n}.npz'.format(d=args.dataset, me='Trimming',
            nt=args.noise_type, nr=args.noise_rate,n=i))
    results_ce = np.load('{d}_analysis/random_{me}_{nt}_{nr}_result_{num}.npz'.format(d=args.dataset, me='CE',
        nt=args.noise_type, nr=args.noise_rate,num=i))
    results_cwo = np.load('{d}_analysis/{me}_{nt}_{nr}_result_{num}.npz'.format(d=args.dataset, me='CE',
        nt=args.noise_type, nr=args.noise_rate,num=i))
    results_cwt = np.load('{d}_analysis/true_{me}_{nt}_{nr}_result_{num}.npz'.format(d=args.dataset, me='CE',
        nt=args.noise_type, nr=args.noise_rate,num=i))
    TrainLoss = results['train_loss_result']
    TrainLoss_ce = results_ce['train_loss_result']
    TrainLoss_cwo = results_cwo['train_loss_result']
    TrainLoss_cwt = results_cwt['train_loss_result']
    trimming_loss[i] = TrainLoss[np.argmin(TrainLoss)]
    ce_with_random[i] = TrainLoss_ce[np.argmin(TrainLoss_ce)]
    ce_with_outlier[i] =TrainLoss_cwo[np.argmin(TrainLoss_cwo)]
    ce_with_true[i] =TrainLoss_cwt[np.argmin(TrainLoss_cwt)]

p = (trimming_loss, ce_with_random ,ce_with_outlier, ce_with_true)
fig, ax = plt.subplots(figsize=(12.0, 8.0))

bp = ax.boxplot(p)
ax.set_xticklabels(['Trimming','CE with random inlier', 'CE with inlier detected by Trimming', 'CE with true inlier'])

plt.title('{nt}'.format( nt=args.noise_type))
plt.xlabel('methods')
plt.ylabel('train_loss')
# Y軸のメモリのrange
#plt.ylim([0,100])
plt.grid()
plt.savefig("train_loss_{nt}.png".format( nt=args.noise_type))

# avg_loss = np.mean(best_loss)
# std_loss = np.std(best_loss)
#print(np.std(best_loss))
# print('& $' , end=' ')
# print('%.8f \\pm %.8f' % (avg_loss, std_loss),  end=' ')
# print('$', end=' ')

best_acc = np.load('best_acc_{d}/{nt}_{nr}_{m}_result.npy'.format(
            d=args.dataset,nt=args.noise_type,m='Trimming', nr=args.noise_rate))
best_acc_ce=np.load('analysis_{d}/random_{nt}_{nr}_{m}_result.npy'.format(
            d=args.dataset, nt=args.noise_type, m='CE', nr=args.noise_rate))
best_acc_cwo=np.load('analysis_{d}/{nt}_{nr}_{m}_result.npy'.format(
            d=args.dataset, nt=args.noise_type, m='CE', nr=args.noise_rate))
best_acc_cwt=np.load('analysis_{d}/true_{nt}_{nr}_{m}_result.npy'.format(
            d=args.dataset, nt=args.noise_type, m='CE', nr=args.noise_rate))
pp = (best_acc,best_acc_ce, best_acc_cwo, best_acc_cwt)
fig, ax = plt.subplots(figsize=(12.0, 8.0))

bp = ax.boxplot(pp)
ax.set_xticklabels(['Trimming','CE with random inlier', 'CE with inlier detected by Trimming', 'CE with true inlier'])

plt.title('{nt}'.format( nt=args.noise_type))
plt.xlabel('methods')
plt.ylabel('R-precision')
# Y軸のメモリのrange
#plt.ylim([0,100])
plt.grid()
plt.savefig("outlier_acc_{nt}.png".format( nt=args.noise_type))

# avg = np.mean(best_acc, axis=0)
# std = np.std(best_acc, axis=0)
# print('& $' , end=' ')
# print('%.4f \\pm %.4f' % (avg, std),  end=' ')
# print('$')

