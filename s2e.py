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
from scipy.special import psi, polygamma
from numpy.linalg.linalg import inv

MAX_NUM = 2
num_classes = 10
EPOCH = 200
batch_size = 128
learning_rate = 0.001
weight_decay = 1e-3
num_gradual = 10
n_iter = 10
n_samples = 6
delta = 1
seed = 1
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
    
    def black_box_function(opt_param):

        cnn1 = clf
        optimizer1 = optimizer
        
        cnn2 = clf
        optimizer2 = optimizer
        
        # rate_schedule=opt_param[0]*(1-np.exp(-opt_param[2]*np.power(np.arange(s.EPOCH,dtype=float),opt_param[1])))+(1-opt_param[0])*(1-1/np.power((opt_param[4]*np.arange(s.EPOCH,dtype=float)+1),opt_param[3]))-np.power(np.arange(EPOCH,dtype=float)/EPOCH,opt_param[5])*opt_param[6]
        rate_schedule=opt_param[0]*(1-np.exp(-opt_param[2]*np.power(np.arange(EPOCH,dtype=float),opt_param[1])))\
    +(1-opt_param[0])*opt_param[7]*(1-1/np.power((opt_param[4]*np.arange(EPOCH,dtype=float)+1),opt_param[3]))\
    +(1-opt_param[0])*(1-opt_param[7])*(1-np.log(1+opt_param[8])/np.log(1+opt_param[8]+opt_param[9]*np.arange(EPOCH,dtype=float)))\
    -np.power(np.arange(EPOCH,dtype=float)/EPOCH,opt_param[5])*opt_param[6]\
    -np.log(1+np.power(np.arange(EPOCH,dtype=float),opt_param[11]))/np.log(1+np.power(EPOCH,opt_param[11]))*opt_param[10]
        #print('Schedule:',rate_schedule,opt_param)
        
        
        for epoch in range(EPOCH):
            # train models
            cnn1.train()
            cnn2.train()
            s2e = S2E()
            train_loss, train_acc = s2e.train(noise_train_loader, epoch, cnn1, optimizer1, cnn2, optimizer2, rate_schedule,device)
            val_loss = val(noise_val_loader)
            test_loss, test_acc = test(test_loader)
            print('epoch %d, train_loss: %f train_acc: %f test_loss: %f test_acc: %f' %
                (epoch+1, train_loss, train_acc, test_loss, test_acc))
            TrainLoss[num][epoch] = train_loss
            TrainAccuracy[num][epoch] = train_acc
            TestLoss[num][epoch] = test_loss
            TestAccuracy[num][epoch] = test_acc
            ValidationLoss[num][epoch] = val_loss
        
        return test_acc
            
    def main():
        np.random.seed(seed)
        cur_acc = np.zeros(n_samples)
        idx = np.zeros(n_samples)
        num_param = 12
        max_pt = np.zeros(num_param)
        hyphyp = np.ones(num_param*2)
        hypgrad = np.zeros((num_param*2, 1))
        hessian = np.zeros((num_param*2, num_param*2))
        for iii in range(n_iter):
            print('Distribution:', hyphyp)
            cur_param = np.zeros((n_samples, num_param))
            loggrad = np.zeros((n_samples, num_param*2, 1))
            loghess = np.zeros((n_samples, num_param*2, num_param*2))
            for jjj in range(n_samples):
                for kkk in range(num_param):
                    cur_param[jjj][kkk] = np.random.beta(
                        hyphyp[2*kkk], hyphyp[2*kkk+1])
                    cur_param[jjj][kkk] = np.random.beta(
                        hyphyp[2*kkk], hyphyp[2*kkk+1])
                    cur_param[jjj][kkk] = np.random.beta(
                        hyphyp[2*kkk], hyphyp[2*kkk+1])
                    loggrad[jjj][2*kkk][0] = np.log(cur_param[jjj][kkk])+psi(
                        hyphyp[2*kkk]+hyphyp[2*kkk+1])-psi(hyphyp[2*kkk])
                    loggrad[jjj][2*kkk+1][0] = np.log(1-cur_param[jjj][kkk])+psi(
                        hyphyp[2*kkk]+hyphyp[2*kkk+1])-psi(hyphyp[2*kkk+1])
                    loghess[jjj][2*kkk][2*kkk] = polygamma(
                        1, hyphyp[2*kkk]+hyphyp[2*kkk+1])-polygamma(1, hyphyp[2*kkk])
                    loghess[jjj][2*kkk][2*kkk +
                                        1] = polygamma(1, hyphyp[2*kkk]+hyphyp[2*kkk+1])
                    loghess[jjj][2*kkk+1][2 *
                                        kkk] = polygamma(1, hyphyp[2*kkk]+hyphyp[2*kkk+1])
                    loghess[jjj][2*kkk+1][2*kkk+1] = polygamma(
                        1, hyphyp[2*kkk]+hyphyp[2*kkk+1])-polygamma(1, hyphyp[2*kkk+1])
                cur_param[jjj][2] *= 0.5
                cur_param[jjj][4] *= 0.5
                cur_param[jjj][9] *= 0.5
                cur_param[jjj][5] /= 0.5
                cur_param[jjj][6] *= 0.5
                cur_param[jjj][11] /= 0.5
                cur_param[jjj][10] *= 0.5
                cur_acc[jjj] = black_box_function(cur_param[jjj])

            idx = np.argsort(cur_acc)
            hypgrad = loggrad[idx[-1]]
            hessian = loggrad[idx[-1]]*loggrad[idx[-1]].T+loghess[idx[-1]]
            hypgrad = hypgrad/n_samples
            hessian = hessian/n_samples
            u, s, vh = np.linalg.svd(hessian, full_matrices=False)
            print(u, s, vh)
            s = np.maximum(s, 1e-5)
            hessian = np.dot(np.dot(u, np.diag(s)), vh)
            hessian = inv(hessian)
            hypgrad = delta*hessian*hypgrad
            hyphyp = hyphyp+hypgrad[:, 0]
            hyphyp = np.maximum(hyphyp, 1)

    main()
