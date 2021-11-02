import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split

import numpy as np
import time
import datetime

from add_noise_split import Noise_dataset_split
from add_noise_split_crust import Noise_dataset_split_crust
import tools
from REL_model import LeNet
from eval import Evaluation_Method
from cdr_method import CDR
from crust_method import CRUST
from lazyGreedy import lazy_greedy_heap
from fl_mnist import FacilityLocationMNIST
import REL_model
import models
from torch.autograd import grad

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')

parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.5)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=120)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=5e-4)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.1)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
args = parser.parse_args()

#device = torch.device("cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, )), ])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'mnist':
    num_gradual_cdr = 10
    num_classes = 10
    batch_size = 32
    transform_train = transform
    transform_test = transform
    train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True,
                                            transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True,
                                            transform=transform_test)

elif args.dataset == 'fmnist':
    num_gradual_cdr = 10
    num_classes = 10
    batch_size = 32
    transform_train = transform
    transform_test = transform
    train_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=True, download=True,
                                                    transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=False, download=True,
                                                    transform=transform_test)
elif args.dataset == 'cifar10':
    if args.method == 'CRUST':
        batch_size = 128
    else:
        batch_size = 64
    num_gradual_cdr = 20
    num_classes = 10
    train_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=True, download=True,
                                                transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, download=True,
                                                transform=transform_test)
elif args.dataset == 'cifar100':
    if args.method == 'CRUST':
        batch_size = 128
    else:
        batch_size = 64
    num_gradual_cdr = 20
    num_classes = 100
    train_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=True, download=True,
                                                transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=False, download=True,
                                                transform=transform_test)


if args.method == 'CDR':
    SPLIT_TRAIN_VAL_RATIO = 0.9
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=True)
    for images, labels in train_loader:
        images_train = images.to(device)
        labels_train = labels

    noise_train_dataset = Noise_dataset_split(images, labels_train, train=True,
                                            transform=transform_train, target_transform=tools.transform_target,
                                            noise_type=args.noise_type, noise_rate=args.noise_rate, dataset=args.dataset,
                                            split_per=0.9, random_seed=1, num_class=10)
    noise_val_dataset = Noise_dataset_split(images, labels_train, train=False,
                                            transform=transform_test, target_transform=tools.transform_target,
                                            noise_type=args.noise_type, noise_rate=args.noise_rate, dataset=args.dataset,
                                            split_per=0.9, random_seed=1, num_class=10)
elif args.method == 'CRUST':
    SPLIT_TRAIN_VAL_RATIO = 1.0
    r_crust = 2.0
    fl_ratio_crust = 0.5
    noise_train_dataset = Noise_dataset_split_crust(train_dataset.data, train_dataset.targets, train=True,
                                                    transform=transform_train, target_transform=tools.transform_target,
                                                    noise_type=args.noise_type, noise_rate=args.noise_rate, dataset=args.dataset,
                                                    split_per=SPLIT_TRAIN_VAL_RATIO, random_seed=1, num_class=num_classes)
    trainval_loader = torch.utils.data.DataLoader(
        noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)


noise_train_loader = torch.utils.data.DataLoader(
    noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)

noise_val_loader = torch.utils.data.DataLoader(
    noise_val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size)


if args.dataset == 'mnist':
    clf = LeNet()
elif args.dataset == 'fmnist':
    clf = REL_model.ResNet50(input_channel=1, num_classes=10)
elif args.dataset == 'cifar10':
    if args.method == 'CRUST':
        clf = models.__dict__[args.arch](num_classes=num_classes)
    else:
        clf = REL_model.ResNet50(input_channel=3, num_classes=10)
elif args.dataset == 'cifar100':
    clf = REL_model.ResNet50(input_channel=3, num_classes=100)

clf = clf.to(device)
optimizer = torch.optim.SGD(
    clf.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()


def eval_on_holdout_data(loader):
    clf.eval()
    eval_method = Evaluation_Method()
    with torch.no_grad():
        for item in loader:
            images = item[0]
            labels = item[1]
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = clf(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
    loss_final, acc_final = eval_method.get_result()
    return loss_final, acc_final


TrainLoss = np.empty(args.epoch)
TrainAccuracy = np.empty(args.epoch)
TestLoss = np.empty(args.epoch)
TestAccuracy = np.empty(args.epoch)
ValidationLoss = np.empty(args.epoch)


if args.method == 'CDR':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 80], gamma=0.1)
elif args.method == 'CRUST':
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 100], last_epoch=args.start_epoch - 1)
    weights = [1] * len(noise_train_dataset)
    weights = torch.FloatTensor(weights)

TrainLoss = np.empty(args.epoch)
TrainAccuracy = np.empty(args.epoch)
TestLoss = np.empty(args.epoch)
TestAccuracy = np.empty(args.epoch)
ValidationLoss = np.empty(args.epoch)

for epoch in range(args.start_epoch, args.epoch):
    start = time.time()
    if args.method == 'CDR':
        cdr = CDR()
        train_loss, train_acc = cdr.train_rel(clf, device, noise_train_loader, epoch, args.noise_rate,
                                              num_gradual_cdr, criterion, optimizer)
    elif args.method == 'CRUST':
        crust = CRUST()
        if epoch >= 5:
            noise_train_dataset.switch_data()
            grads_all, labels = crust.estimate_grads(
                trainval_loader, clf, device, criterion)
            ssets = []
            weights = []
            for c in range(num_classes):

                sample_ids = np.where((labels == c) == True)[0]
                grads = grads_all[sample_ids]

                dists = pairwise_distances(grads)
                weight = np.sum(dists < r_crust, axis=1)
                V = range(len(grads))
                F = FacilityLocationMNIST(V, D=dists)
                B = int(fl_ratio_crust * len(grads))
                sset, vals = lazy_greedy_heap(F, V, B)

                #if len(sset) > 0:
                weights.extend(weight[sset].tolist())
                sset = sample_ids[np.array(sset)]
                ssets += list(sset)
            #assert(len(ssets) == len(weights))
            weights = torch.FloatTensor(weights)
            #weightsprint(weights.shape)
            noise_train_dataset.adjust_base_indx_tmp(ssets)
            #print("train_dataset_len ", len(noise_train_dataset))

        if epoch > 5:
            train_loss, train_acc = crust.train_crust(
                noise_train_loader, clf, device, criterion, weights, optimizer, fetch=True)
        else:
            train_loss, train_acc = crust.train_crust(
                noise_train_loader, clf, device, criterion, weights, optimizer, fetch=False)
    elif args.method == 'CE':
        train_loss, train_acc = train(noise_train_loader)

    if SPLIT_TRAIN_VAL_RATIO == 1.0:
        val_loss = np.NAN
    else:
        val_loss, _ = eval_on_holdout_data(noise_val_loader)
    test_loss, test_acc = eval_on_holdout_data(test_loader)

    print('epoch %d, train_loss: %f train_acc: %f test_loss: %f test_acc: %f' %
          (epoch+1, train_loss, train_acc, test_loss, test_acc))
    TrainLoss[epoch] = train_loss
    TrainAccuracy[epoch] = train_acc
    TestLoss[epoch] = test_loss
    TestAccuracy[epoch] = test_acc
    ValidationLoss[epoch] = val_loss
    lr_scheduler.step()
    end = time.time() - start
    print(end)


if SPLIT_TRAIN_VAL_RATIO == 1.0:
    test_acc_max = TestAccuracy[args.epoch-1]
else:
    test_acc_max = TestAccuracy[np.argmin(ValidationLoss)]
print('Best Accuracy', test_acc_max)


np.savez('{d}_result/{me}_{nt}_{nr}_result'.format(d=args.dataset, me=args.method,
                                                   nt=args.noise_type, nr=args.noise_rate), train_loss_result=TrainLoss,
         train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
         test_acc_result=TestAccuracy, val_loss_result=ValidationLoss)
