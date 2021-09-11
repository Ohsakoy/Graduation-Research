import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
import argparse
import numpy as np
import time
import datetime
import data_load
import add_noise_split
import tools
from REL_model import LeNet
from eval import Evaluation_Method
from cde_method import CDE
from crust_method import CRUST
from lazyGreedy import lazy_greedy_heap
from fl_mnist import FacilityLocationMNIST
import REL_model

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.4)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=1e-3)
parser.add_argument('--momentum', type=int, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDE')
args = parser.parse_args()

num = 0
MAX_NUM = 2
num_classes = 10
EPOCH = 100
batch_size = 32
learning_rate = 0.01
weight_decay = 1e-3
num_gradual = 10
n_iter = 10
n_samples = 6
early_exit = False
r = 2.0
fl_ratio = 0.5
use_crust = True
constants = torch.FloatTensor
noise_type = 'symmetric'
#noise_type = 'asymmetric'
#noise_type = 'instance'
TrainLoss = np.empty(100)
TrainAccuracy = np.empty(100)
TestLoss = np.empty(100)
TestAccuracy = np.empty(100)
ValidationLoss = np.empty(100)

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
    transform_train = transform
    transform_test = transform
    train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True,
                                            transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True,
                                            transform=transform_test)
    
elif args.dataset == 'fmnist':
    transform_train = transform
    transform_test = transform
    train_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=True, download=True,
                                                    transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=False, download=True,
                                                    transform=transform_test)
elif args.dataset == 'cifar10':
    batch_size = 64
    num_gradual = 20
    train_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=True, download=True,
                                                transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, download=True,
                                                transform=transform_test)
elif args.dataset == 'cifar100':
    batch_size = 64
    num_gradual = 20
    train_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=True, download=True,
                                                transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=False, download=True,
                                                transform=transform_test)


#data loader)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size)


#device = torch.device("cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if args.dataset == 'mnist':
    clf = LeNet()
elif args.dataset == 'fmnist':
    clf = REL_model.ResNet50(input_channel=1, num_classes=10)
elif args.dataset == 'cifar10':
    clf = REL_model.ResNet50(input_channel=3, num_classes=10)
elif args.dataset == 'cifar100':
    clf = REL_model.ResNet50(input_channel=3, num_classes=100)

clf = clf.to(device)
optimizer = torch.optim.SGD(
    clf.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
criterion = nn.CrossEntropyLoss()

for images, labels in train_loader:
    images = images.to(device)

noise_train_dataset = add_noise_split.noise_dataset_split(images, labels, train=True,
                                                        transform=transform_train, target_transform=tools.transform_target,
                                                        noise_type=args.noise_type, noise_rate=args.noise_rate, dataset=args.dataset,
                                                        split_per=0.9, random_seed=1, num_class=10)

#val_noise
noise_val_dataset = add_noise_split.noise_dataset_split(images, labels, train=False,
                                                        transform=transform_test, target_transform=tools.transform_target,
                                                        noise_type=args.noise_type, noise_rate=args.noise_rate, dataset=args.dataset,
                                                        split_per=0.9, random_seed=1, num_class=10)


noise_train_loader = torch.utils.data.DataLoader(
    noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

#CRUST
trainval_loader = torch.utils.data.DataLoader(
    noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

noise_val_loader = torch.utils.data.DataLoader(
    noise_val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)


def train(noise_train_loader):
    clf.train()
    eval_method = Evaluation_Method()
    for images, labels, _ in noise_train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs, _ = clf.forward(images)
        loss = criterion(outputs, labels)
        eval_method.loss(loss)
        eval_method.acc(labels, outputs)
        loss.backward()
        optimizer.step()
    train_loss, train_acc = eval_method.get_result()
    return train_loss, train_acc


def test(test_loader):
    clf.eval()
    eval_method = Evaluation_Method()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = clf(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
    test_loss, test_acc = eval_method.get_result()
    return test_loss, test_acc


def val(val_loader):
    clf.eval()
    eval_method = Evaluation_Method()
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = clf(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
    val_loss, val_acc = eval_method.get_result()
    return val_loss


weights = [1] * len(noise_train_dataset)
weights = torch.FloatTensor(weights)


for epoch in range(EPOCH):
    start = time.time()
    if args.method == 'CDE':
        cde = CDE()
        train_loss, train_acc = cde.train_rel(clf, device, noise_train_loader, epoch, args.noise_rate,
                                                num_gradual, criterion, optimizer)
    elif args.method == 'CRUST':
        crust = CRUST()
        if use_crust and epoch >= 5:
            grads_all, labels = crust.estimate_grads(
                trainval_loader, clf, criterion)

            ssets = []
            weights = []
            for c in range(num_classes):
                sample_ids = np.where((labels == c) == True)[0]
                grads = grads_all[sample_ids]

                dists = pairwise_distances(grads)
                weight = np.sum(dists < r, axis=1)
                V = range(len(grads))
                F = FacilityLocationMNIST(V, D=dists)
                B = int(fl_ratio * len(grads))
                sset, vals = lazy_greedy_heap(F, V, B)

                if len(sset) > 0:
                    weights.extend(weight[sset].tolist())
                    sset = sample_ids[np.array(sset)]
                    ssets += list(sset)

            assert(len(ssets) == len(weights))
            weights = torch.FloatTensor(weights)
            print('weights', weights.shape)
            noise_train_dataset.adjust_base_indx_tmp(ssets)
            print(len(noise_train_dataset))
            #noise_train_loader = torch.utils.data.DataLoader(
            #   noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        print('weights', weights.shape)
        if use_crust and epoch > 5:
            train_loss, train_acc, early_exit = crust.train_crust(
                noise_train_loader, clf, criterion, weights, optimizer, fetch=True)
        else:
            train_loss, train_acc, early_exit = crust.train_crust(
                noise_train_loader, clf, criterion, weights, optimizer, fetch=False)
    elif args.method == 'CE':
        train_loss, train_acc = train(noise_train_loader)

    if early_exit:
        break

    val_loss = val(noise_val_loader)
    test_loss, test_acc = test(test_loader)

    print('epoch %d, train_loss: %f train_acc: %f test_loss: %f test_acc: %f' %
            (epoch+1, train_loss, train_acc, test_loss, test_acc))
    TrainLoss[epoch] = train_loss
    TrainAccuracy[epoch] = train_acc
    TestLoss[epoch] = test_loss
    TestAccuracy[epoch] = test_acc
    ValidationLoss[epoch] = val_loss
    end = time.time() - start

id = np.argmin(ValidationLoss)
test_acc_max = TestAccuracy[id]
print('Best Accuracy', test_acc_max)


np.savez('FMNIST_Result/{me}_{nt}_{nr}_result'.format(me=args.method, nt=args.noise_type, nr=args.noise_rate), train_loss_result=TrainLoss,
        train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
        test_acc_result=TestAccuracy, val_loss_result=ValidationLoss)
