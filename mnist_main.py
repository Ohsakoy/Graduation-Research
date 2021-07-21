import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torch.optim as optim
import argparse
import sys
import numpy as np
import time
import datetime
import data_load
import tools
from REL_model import LeNet
from eval import Evaluation_Method

MAX_NUM = 2
num_classes = 10
EPOCH = 100
batch_size = 32
learning_rate = 0.01
weight_decay = 1e-3
num_gradual = 10
constants = torch.FloatTensor
#noise_type = 'symmetric'
#noise_type = 'asymmetric'
noise_type = 'instance'
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

    noise_train_loader = torch.utils.data.DataLoader(
        noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

    #val_noise 
    noise_val_dataset = data_load.mnist_dataset(images, labels, train=False,
                                                noise_type=noise_type, noise_rate=NOISE_RATES[num],
                                                split_per=0.9, random_seed=1, num_class=10)

    noise_val_loader = torch.utils.data.DataLoader(
        noise_val_dataset, batch_size=batch_size, drop_last=False, shuffle=False)

    def train(noise_train_loader):
        clf.train()
        eval_method = Evaluation_Method()
        for images, labels,index in noise_train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = clf.forward(images)
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
        val_loss,val_acc = eval_method.get_result()
        return val_loss

    def accuracy(logits,target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        output = F.softmax(logits, dim=1)
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def train_one_step(images, labels, nonzero_ratio, clip):
        clf.train()
        pred = clf(images)
        loss = criterion(pred, labels)
        loss.backward()
        
        to_concat_g = []
        to_concat_v = []
        for name, param in clf.named_parameters():
            if param.dim() in [2, 4]:
                to_concat_g.append(param.grad.data.view(-1))
                to_concat_v.append(param.data.view(-1))
        all_g = torch.cat(to_concat_g)
        all_v = torch.cat(to_concat_v)
        metric = torch.abs(all_g * all_v)
        num_params = all_v.size(0)
        nz = int(nonzero_ratio * num_params)
        top_values, _ = torch.topk(metric, nz)
        thresh = top_values[-1]

        for name, param in clf.named_parameters():
            if param.dim() in [2, 4]:
                mask = (torch.abs(param.data * param.grad.data)
                        >= thresh).type(constants)
                mask = mask * clip
                param.grad.data = mask * param.grad.data

        optimizer.step()
        optimizer.zero_grad()
        
        acc = accuracy(pred, labels, topk=(1,))

        return float(acc[0]), loss

    def train_rel(train_loader,epoch):
        clf.train()
        train_total = 0
        total = 0
        train_correct = 0
        total_loss = 0
        clip_narry = np.linspace(1-NOISE_RATES[num], 1, num=num_gradual)
        clip_narry = clip_narry[::-1]
        if epoch < num_gradual:
            clip = clip_narry[epoch]
        
        clip = (1 - NOISE_RATES[num])
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Loss transfer
            prec, loss = train_one_step(images, labels, clip, clip)
            total_loss += loss.item()
            train_total += 1
            total += labels.size(0)
            train_correct += prec
        
        train_loss =  float(total_loss)/float(total)   
        train_acc = float(train_correct)/float(train_total)
        return train_loss, train_acc

    def evaluate(test_loader):
        clf.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = clf(images)
                #loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                #acc
                outputs1 = F.softmax(logits, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total += labels.size(0)
                correct += (pred1.cpu() == labels.long()).sum()

        test_loss = total_loss / total
        test_acc = 100 * float(correct) / float(total)
        
        return test_loss, test_acc
    
    def val_rel(val_loader):
        clf.eval()
        correct = 0
        total = 0
        total_loss = 0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = clf(images)
                #loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                #acc
                outputs1 = F.softmax(logits, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total += labels.size(0)
                correct += (pred1.cpu() == labels.long()).sum()

        val_loss = total_loss / total
        val_acc = 100 * float(correct) / float(total)

        return val_loss, val_acc

    start = time.time()
    for epoch in range(EPOCH):
        #train_loss, train_acc = train(noise_train_loader)
        #val_loss = val(noise_val_loader)
        #test_loss, test_acc = test(test_loader)
        
        train_loss, train_acc = train_rel(noise_train_loader,epoch)
        val_loss, val_acc = val_rel(noise_val_loader)
        test_loss, test_acc = evaluate(test_loader)
        
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

'''
np.savez('MNIST_Result/{}_noise_result'.format(noise_type), train_loss_result=TrainLoss,
        train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
        test_acc_result=TestAccuracy, val_loss_result=ValidationLoss)
'''

np.savez('MY_REL_Result/{}_noise_result'.format(noise_type), train_loss_result=TrainLoss,
        train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
        test_acc_result=TestAccuracy,val_loss_result=ValidationLoss)


