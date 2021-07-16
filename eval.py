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


class Evaluation_Method():
    def __init__(self):
        self.total_loss = 0.0
        self.correct_predicted_num = 0.0
        self.data_num = 0.0
        
    def loss(self, loss):
        self.total_loss += loss.item()
    
    def acc(self, labels, outputs):
        predicted = outputs.max(1, keepdim=True)[1]
        labels = labels.view_as(predicted)
        self.correct_predicted_num += predicted.eq(labels).sum().item()
        self.data_num += labels.size(0)
    
    def get_result(self):
        avg_loss = self.total_loss / self.data_num
        avg_accuracy = self.correct_predicted_num / self.data_num
        return avg_loss, avg_accuracy


