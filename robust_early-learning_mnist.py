import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import torchvision.models as tv_models
import torch.optim as optim
import argparse
import sys
import numpy as np
import datetime
import Data_load
import Tools
from REL_model import LeNet
import time

num = 0
MAX_NUM = 2
num_classes = 10
EPOCH = 100
batch_size = 32
learning_rate = 0.01
weight_decay = 1e-3
seed = 1
split_percentage = 0.9
num_gradual = 10
num_workers = 4
#noise_type = 'symmetric'
noise_type = 'asymmetric'
#noise_type = 'instance'
NOISE_RATES = np.array([0.2, 0.4])
TrainLoss = np.empty((2, 100))
TrainAccuracy = np.empty((2, 100))
TestLoss = np.empty((2, 100))
TestAccuracy = np.empty((2, 100))
ValidationLoss = np.empty((2, 100))
ValidationAccuracy = np.empty((2, 100))

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, )), ])

for num in range(MAX_NUM):
    train_dataset = Data_load.mnist_dataset(True,transform=transform,
                                            target_transform=Tools.transform_target,
                                            noise_type=noise_type,
                                            noise_rate=NOISE_RATES[num],
                                            split_per=split_percentage,
                                            random_seed=seed)

    val_dataset = Data_load.mnist_dataset(False,transform=transform,
                                        target_transform=Tools.transform_target,
                                        noise_type=noise_type,
                                        noise_rate=NOISE_RATES[num],
                                        split_per=split_percentage,
                                        random_seed=seed)

    test_dataset = Data_load.mnist_test_dataset(transform=transform,
                target_transform=Tools.transform_target)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            drop_last=False,
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            drop_last=False,
                                            shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            drop_last=False,
                                            shuffle=False)



    def accuracy(logit, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        output = F.softmax(logit, dim=1)
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


    def train_one_step(net, data, label, optimizer, criterion, nonzero_ratio, clip):
        net.train()
        pred = net(data)
        loss = criterion(pred, label)
        loss.backward()

        to_concat_g = []
        to_concat_v = []
        for name, param in net.named_parameters():
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

        for name, param in net.named_parameters():
            if param.dim() in [2, 4]:
                mask = (torch.abs(param.data * param.grad.data)
                        >= thresh).type(torch.FloatTensor)
                mask = mask * clip
                param.grad.data = mask * param.grad.data

        optimizer.step()
        optimizer.zero_grad()
        acc = accuracy(pred, label, topk=(1,))

        return float(acc[0]), loss


    def train(train_loader, epoch, model1, optimizer1, ):
        model1.train()
        train_total = 0
        total = 0
        train_correct = 0
        total_loss = 0
        clip_narry = np.linspace(1-NOISE_RATES[num], 1, num=num_gradual)
        clip_narry = clip_narry[::-1]
        if epoch < num_gradual:
            clip = clip_narry[epoch]

        clip = (1 - NOISE_RATES[num])
        for i, (data, labels, indexes) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            logits1 = model1(data)
            prec1,  = accuracy(logits1, labels, topk=(1, ))
            train_total += 1
            total += labels.size(0)
            train_correct += prec1
            # Loss transfer
            prec1, loss = train_one_step(
                model1, data, labels, optimizer1, nn.CrossEntropyLoss(), clip, clip)
            total_loss += loss.item()

        train_loss = float(total_loss)/ float(total)
        train_acc1 = float(train_correct)/float(train_total)
        return train_loss, train_acc1


    # Evaluate the Model
    def evaluate(test_loader, model1):
        model1.eval()  # Change model to 'eval' mode.
        correct1 = 0
        total1 = 0
        total_loss = 0
        with torch.no_grad():
            for data, labels, _ in test_loader:
                data = data.to(device)
                labels = labels.to(device)
                #data = data.cuda()
                logits1 = model1(data)
                loss = criterion(logits1, labels)
                total_loss += loss.item()
                outputs1 = F.softmax(logits1, dim=1)
                _, pred1 = torch.max(outputs1.data, 1)
                total1 += labels.size(0)
                correct1 += (pred1.cpu() == labels.long()).sum()

            acc1 = 100.0 * float(correct1) / float(total1)
            test_loss = float(total_loss) / float(total1)

        return test_loss, acc1


    device = torch.device("cpu")
    clf1 = LeNet()
    clf1 = clf1.to(device)
    optimizer1 = torch.optim.SGD(clf1.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler1 = MultiStepLR(optimizer1, milestones=[10, 20], gamma=0.1)

    start = time.time()
    for epoch in range(EPOCH):
        #scheduler1.step()
        #print(optimizer1.state_dict()['param_groups'][0]['lr'])
        clf1.train()

        train_loss,train_acc = train(train_loader, epoch, clf1, optimizer1)
        val_loss, val_acc = evaluate(val_loader, clf1)
        test_loss,test_acc = evaluate(test_loader, clf1)
        
        print('epoch %d, train_loss: %f train_acc: %f test_loss: %f test_acc: %.4f' %
            (epoch+1, train_loss, train_acc, test_loss, test_acc))
        TrainLoss[num][epoch] = train_loss
        TrainAccuracy[num][epoch] = train_acc
        TestLoss[num][epoch] = test_loss
        TestAccuracy[num][epoch] = test_acc
        ValidationAccuracy[num][epoch] = val_acc
        
    end = time.time() - start
    print(end)


np.savez('REL_Result/{}_noise_result'.format(noise_type), train_loss_result=TrainLoss,
        train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
        test_acc_result=TestAccuracy, val_acc_result=ValidationAccuracy)
