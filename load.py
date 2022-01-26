import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import time
from REL_model import LeNet
from REL_model import Logistic
from REL_model import NeuralNetwork
from REL_model import NeuralNetLinear
import eval_on_holdout
import REL_model
import models
import plot
import seaborn as sns
import matplotlib.pyplot as plt

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
parser.add_argument('--nt', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--num', type=int, default=100)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=0.01)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
parser.add_argument('--inlier_ratio', type=float, default=0.8)
args = parser.parse_args()

#device = torch.device("cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_images = torch.load('{d}_data/{d}_{nt}_{nr}_train_images.pt'.format(
    d='nonlinear', nt=args.noise_type, nr=args.noise_rate))
val_images = torch.load('{d}_data/{d}_{nt}_{nr}_val_images.pt'.format(
    d='nonlinear', nt=args.noise_type, nr=args.noise_rate))
train_and_val_images = torch.load('{d}_data/{d}_{nt}_{nr}_train_and_val_images.pt'.format(
    d='nonlinear', nt=args.noise_type, nr=args.noise_rate))

labels = np.load('{d}_data/{d}_{nt}_{nr}_labels.npz'.format(
    d='nonlinear', nt=args.noise_type, nr=args.noise_rate))
train_labels = labels['train_labels']
val_labels = labels['val_labels']
train_and_val_labels = labels['train_and_val_labels']
train_and_val_labels_without_noise = labels['train_and_val_labels_without_noise']


train_data = torch.load('{d}_data/{d}_{nt}_{nr}_train_images.pt'.format(
    d='nonlinear', nt='symmetric', nr=args.noise_rate))
val_data = torch.load('{d}_data/{d}_{nt}_{nr}_val_images.pt'.format(
    d='nonlinear', nt='symmetric', nr=args.noise_rate))
train_and_val_data = torch.load('{d}_data/{d}_{nt}_{nr}_train_and_val_images.pt'.format(
    d='nonlinear', nt='symmetric', nr=args.noise_rate))

targets = np.load('{d}_data/{d}_{nt}_{nr}_labels.npz'.format(
    d='nonlinear', nt='symmetric', nr=args.noise_rate))
train_targets = targets['train_labels']
val_targets = targets['val_labels']
train_and_val_targets = targets['train_and_val_labels']
train_and_val_targets_without_noise = targets['train_and_val_labels_without_noise']

plot.save_fig(train_and_val_images, train_and_val_labels, args.noise_type,
            train_and_val_data, train_and_val_targets, args.nt)

'''

n=10
for num in range(190,200):
    model.load_state_dict(torch.load(
        "island_models/model_{e}.pth".format(nt=args.noise_type, e=num)), strict=False)
    model.train()
    for item in noise_train_loader:
        images = item[0]
        labels = item[1]
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        outputs, _ = model.forward(images)
        _, pred = torch.max(outputs, 1)
        optimizer.step()
        
    images = images.to('cpu').detach().numpy().copy()
    labels = labels.to('cpu').detach().numpy().copy()
    pred = pred.to('cpu').detach().numpy().copy()
    X=images

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    sns.scatterplot(X[:, 0], X[:, 1], hue=labels, ax=ax1)
    
    model.eval()
    for item in test_loader:
        data = item[0]
        targets = item[1]
        data = data.to(device)
        targets = targets.to(device)
        outputs, _ = model(data)
        _, pred = torch.max(outputs, 1)
        

    data = data.to('cpu').detach().numpy().copy()
    targets = targets.to('cpu').detach().numpy().copy()
    pred = pred.to('cpu').detach().numpy().copy()
    
    X = data
    
    #f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
    #sns.scatterplot(X[:, 0], X[:, 1], hue=targets, ax=ax1)
    sns.scatterplot(X[:, 0], X[:, 1], hue=pred, ax=ax2)
    plt.savefig("Plot_island/{nt}_label_{n}.png".format(nt=args.noise_type, n=num))
'''
