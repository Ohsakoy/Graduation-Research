import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
import torch.optim as optim
import argparse
import sys
import numpy as np
import time
import data_load
import tools
from REL_model import LeNet
from eval import Evaluation_Method
from rel_method import CDE
from rel_method import S2E
from rel_method import CRUST
from lazyGreedy import lazy_greedy_heap 
from fl_mnist import FacilityLocationMNIST


MAX_NUM = 2
num_classes = 10
EPOCH = 100
batch_size = 32
learning_rate = 0.01
weight_decay = 1e-3
num_gradual = 10
n_iter = 10
n_samples = 6
use_crust = True
r = 2.0
fl_ratio = 0.5
constants = torch.FloatTensor
noise_type = 'symmetric'
#noise_type = 'asymmetric'
#noise_type = 'instance'
NOISE_RATES = np.array([0.2, 0.4])
TrainLoss = np.empty((2, 100))
TrainAccuracy = np.empty((2, 100))
TestLoss = np.empty((2, 100))
TestAccuracy = np.empty((2, 100))
ValidationLoss = np.empty((2, 100))

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, )), ])


train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True,
                                        transform=transform, target_transform=tools.transform_target)


test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True,
                                        transform=transform)

#data loader)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size)


for num in range(MAX_NUM):
    device = torch.device("cpu")
    clf = LeNet()
    clf = clf.to(device)
    optimizer = torch.optim.SGD(
        clf.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for images, labels in train_loader:
        images = images.to(device)

    #train_noise
    noise_train_dataset = data_load.mnist_dataset(images, labels, train=True,
                                                noise_type=noise_type, noise_rate=NOISE_RATES[num],
                                                split_per=0.9, random_seed=1, num_class=10)

#    noise_or_not = noise_train_dataset.noise_or_not

    noise_train_loader = torch.utils.data.DataLoader(
        noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    trainval_loader = torch.utils.data.DataLoader(
        noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
    #val_noise
    noise_val_dataset = data_load.mnist_dataset(images, labels, train=False,
                                                noise_type=noise_type, noise_rate=NOISE_RATES[num],
                                                split_per=0.9, random_seed=1, num_class=10)

    noise_val_loader = torch.utils.data.DataLoader(
        noise_val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    def test(test_loader):
        clf.eval()
        eval_method = Evaluation_Method()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = clf(images)
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
                outputs = clf(images)
                loss = criterion(outputs, labels)
                eval_method.loss(loss)
                eval_method.acc(labels, outputs)
        val_loss, val_acc = eval_method.get_result()
        return val_loss

    start = time.time()
    weights = [1] * len(noise_train_dataset)
    weights = torch.FloatTensor(weights)
    for epoch in range(EPOCH):
        
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
                weights.extend(weight[sset].tolist())
                sset = sample_ids[np.array(sset)]
                ssets += list(sset)
            weights = torch.FloatTensor(weights)
        if use_crust and epoch > 5:
            fetch = True
            train_loss, train_acc = crust.train_crust(
                noise_train_loader, clf, criterion, weights, optimizer, fetch)
        else:
            fetch = False
            train_loss, train_acc = crust.train_crust(
                noise_train_loader, clf, criterion, weights, optimizer, fetch)
        
        val_loss = val(noise_val_loader)
        test_loss, test_acc = test(test_loader)

        print('epoch %d, train_loss: %f train_acc: %f test_loss: %f test_acc: %f' %
            (epoch+1, train_loss, train_acc, test_loss, test_acc))
        TrainLoss[num][epoch] = train_loss
        TrainAccuracy[num][epoch] = train_acc
        TestLoss[num][epoch] = test_loss
        TestAccuracy[num][epoch] = test_acc
        ValidationLoss[num][epoch] = val_loss

    id = np.argmin(ValidationLoss[num])
    test_acc_max = TestAccuracy[num][id]
    print('Best Accuracy', test_acc_max)
    end = time.time() - start
    print(end)


np.savez('CRUST_MNIST_Result/{}_noise_result'.format(noise_type), train_loss_result=TrainLoss,
        train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
        test_acc_result=TestAccuracy, val_loss_result=ValidationLoss)
