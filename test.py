import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import numpy as np

from Nuralnet import Net
from eval import Evaluation_Method


MAX_NUM = 1
EPOCH = 1
TrainLoss = np.empty((10, 100))
TrainAccuracy = np.empty((10, 100))
TestLoss = np.empty((10, 100))
TestAccuracy = np.empty((10, 100))
time_list = np.empty(10)




#データセット
fashion_mnist_train = FashionMNIST(
    "FashionMNIST", train=True, download=True, transform=transforms.ToTensor())

fashion_mnist_test = FashionMNIST(
    "FashionMNIST", train=False, download=True, transform=transforms.ToTensor())

BATCH_SIZE = 64
train_loader = DataLoader(
    fashion_mnist_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(
    fashion_mnist_test, batch_size=BATCH_SIZE, shuffle=True)


for num in range(MAX_NUM):
    print("{x}回目".format(x=num+1))
    device = torch.device("cpu")
    net = Net()
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())

    def train(train_loader):
        net.train()
        eval_method = Evaluation_Method()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net.forward(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = eval_method.get_result()
        return train_loss, train_acc

    def test(test_loader):
        net.eval()
        eval_method = Evaluation_Method()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                eval_method.loss(loss)
                eval_method.acc(labels, outputs)
        test_loss, test_acc = eval_method.get_result()
        return test_loss, test_acc

    start = time.time()

    for epoch in range(EPOCH):
        train_loss, train_acc = train(train_loader)
        test_loss, test_acc = test(test_loader)
        print('epoch %d, train_loss: %.4f train_acc: %.4f test_loss: %.4f test_acc: %.4f' %
              (epoch+1, train_loss, train_acc, test_loss, test_acc))

    end = time.time() - start
    print(end)

