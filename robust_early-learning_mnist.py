import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
from torchvision import datasets
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 32
EPOCH = 100
TrainLoss = np.empty((10, 100))
TrainAccuracy = np.empty((10, 100))
TestLoss = np.empty((10, 100))
TestAccuracy = np.empty((10, 100))
time_list = np.empty(10)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307, ), (0.3081, )), ])

trainval_dataset = datasets.MNIST(
    'mnist', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(
    'mnist', train=False, transform=transform, download=True)

train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,[50000,10000])

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False)


