import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import numpy as np

from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
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
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=0.01)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
parser.add_argument('--outlier_ratio', type=float, default=0.2)
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

batchsize = 10


def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data


class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c]
                        for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
            
        label = int(self.img_list[idx][1])
        
        return (image, label)
        # sample = {'img': image,
        #         'label': int(self.img_list[idx][1])}
        
        #return sample


train_dataset = CovidCTDataset(root_dir='new_data/CT_image',
                        txt_COVID='new_data/Covid_txt/trainCT_COVID.txt',
                        txt_NonCOVID='new_data/NonCovid_txt/trainCT_NonCOVID.txt',
                        transform=train_transformer)
val_dataset = CovidCTDataset(root_dir='new_data/CT_image',
                        txt_COVID='new_data/Covid_txt/valCT_COVID.txt',
                        txt_NonCOVID='new_data/NonCovid_txt/valCT_NonCOVID.txt',
                        transform=val_transformer)
train_and_val_dataset = CovidCTDataset(root_dir='new_data/CT_image',
                                txt_COVID='new_data/Covid_txt/train_and_valCT_COVID.txt',
                                txt_NonCOVID='new_data/NonCovid_txt/train_and_valCT_NonCOVID.txt',
                                transform=train_transformer)
test_dataset = CovidCTDataset(root_dir='new_data/CT_image',
                        txt_COVID='new_data/Covid_txt/testCT_COVID.txt',
                        txt_NonCOVID='new_data/NonCovid_txt/testCT_NonCOVID.txt',
                        transform=val_transformer)

train_loader = DataLoader(
    train_dataset, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batchsize,
                        drop_last=False, shuffle=False)
train_and_val_loader = DataLoader(
    train_and_val_dataset, batch_size=batchsize, drop_last=False, shuffle=True)

test_loader = DataLoader(
    test_dataset, batch_size=batchsize, drop_last=False, shuffle=False)


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()  # b, 3, 32, 32
        layer1 = torch.nn.Sequential()
        layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))

        #b, 32, 32, 32
        layer1.add_module('relu1', torch.nn.ReLU(True))
        layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2)
                          )  # b, 32, 16, 16 //池化为16*16
        self.layer1 = layer1
        layer4 = torch.nn.Sequential()
        layer4.add_module('fc1', torch.nn.Linear(401408, 2))
        self.layer4 = layer4

    def forward(self, x):
        conv1 = self.layer1(x)
        fc_input = conv1.view(conv1.size(0), -1)
        fc_out = self.layer4(fc_input)


model = SimpleCNN()

model = model.to(device)
optimizer = torch.optim.SGD(
    model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()


for item in (train_loader):

    print(item[0].shape)
    print(item[1].shape)
    # print(batch_samples)
    # data = batch_samples['img'].to(device)
    # target = batch_samples['label'].to(device)
    # print(data.shape)
    # print(target.shape)
    # print(type(data))
    # print(type(target))
    assert False
        
        
        
