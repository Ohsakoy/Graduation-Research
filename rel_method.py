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
from loss import loss_coteaching
from torch.autograd import grad

class CDE():
    def accuracy(self,logits, target, topk=(1,)):
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

    def train_one_step(self,clf, images, labels, nonzero_ratio, clip, criterion, optimizer):
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
                        >= thresh).type(torch.FloatTensor)
                mask = mask * clip
                param.grad.data = mask * param.grad.data

        optimizer.step()
        optimizer.zero_grad()
        
        acc = self.accuracy(pred, labels, topk=(1,))

        return float(acc[0]), loss

    def train_rel(self, clf, device, train_loader, epoch, noise_rate, num_gradual, criterion, optimizer):
        clf.train()
        train_total = 0
        total = 0
        train_correct = 0
        total_loss = 0
        clip_narry = np.linspace(1-noise_rate, 1, num=num_gradual)
        clip_narry = clip_narry[::-1]
        if epoch < num_gradual:
            clip = clip_narry[epoch]
        
        clip = (1 - noise_rate)
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Loss transfer
            prec, loss = self.train_one_step(clf, images, labels, clip, clip, criterion, optimizer)
            total_loss += loss.item()
            train_total += 1
            total += labels.size(0)
            train_correct += prec
        
        train_loss =  float(total_loss)/float(total)   
        train_acc = float(train_correct)/float(train_total)
        return train_loss, train_acc


    '''
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
    '''



class S2E():
    def accuracy(self,logit, target, topk=(1,)):
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
    
    def train(self, train_loader, epoch, model1, optimizer1, model2, optimizer2, rate_schedule, device):
        train_total = 0
        train_correct = 0
        train_total2 = 0
        train_correct2 = 0
        total_loss1 = 0
        total_loss2 = 0
        total = 0

        for images, labels, index in (train_loader):
            
            images = Variable(images)
            labels = Variable(labels)
            
            images = images.to(device)
            labels = labels.to(device)
        
            # Forward + Backward + Optimize
            logits1 = model1(images)
            prec1= self.accuracy(logits1, labels, topk=(1,))
            prec1 = float(prec1[0])
            train_total += 1
            train_correct += prec1

            logits2 = model2(images)
            prec2= self.accuracy(logits2, labels, topk=(1, ))
            prec2 = float(prec2[0])
            train_total2 += 1
            train_correct2 += prec2
            
            rate_schedule[epoch] = min(rate_schedule[epoch], 0.99)
            loss_1, loss_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch])
            total_loss1 += loss_1.item()
            total_loss2 += loss_2.item()
            total += labels.size(0)

            optimizer1.zero_grad()
            loss_1.backward()
            optimizer1.step()
            optimizer2.zero_grad()
            loss_2.backward()
            optimizer2.step()
            
        train_loss1 = float(total_loss1)/(total)
        train_loss2 = float(total_loss2)/(total)
        train_loss = (train_loss1 + train_loss2) / 2
        train_acc1 = float(train_correct)/float(train_total)
        train_acc2 = float(train_correct2)/float(train_total2)
        train_acc = (train_acc1+train_acc2) / 2
        return train_loss, train_acc

class CRUST():
    def accuracy(self, logits, target, topk=(1,)):
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

    def train_crust(self, train_loader, model, criterion, weights, optimizer, fetch):
        # switch to train mode
        model.train()
        train_total = 0
        train_correct = 0
        total_loss = 0
        total = 0

        for input, target, index in (train_loader):
            if fetch:
                input_b = train_loader.dataset.fetch(target)
                lam = np.random.beta(1, 0.1)
                input = lam * input + (1 - lam) * input_b
            c_weights = weights[index]
            c_weights = c_weights.type(torch.FloatTensor)
            c_weights = c_weights / c_weights.sum()
            
            #input = input.type(torch.FloatTensor)         
            
            output = model(input)
            loss = criterion(output, target)
            loss = (loss * c_weights).sum()
            total_loss += loss.item()
            
            prec1 = self.accuracy(output, target, topk=(1,))
            train_total += 1
            total += target.size(0)
            train_correct += prec1[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_acc = float(train_correct)/float(train_total)
        train_loss = float(total_loss)/float(total)
        
        return train_loss, train_acc
    
    def estimate_grads(self, trainval_loader, model, criterion):
        # switch to train mode
        model.train()
        all_grads = []
        all_targets = []
        
        for input, target, _ in (trainval_loader):
            all_targets.append(target)
            
            # compute output
            output, feat = model(input)
            
            loss = criterion(output, target).mean()
            est_grad = grad(loss, feat)
            all_grads.append(est_grad[0].detach().cpu().numpy())
            
        all_grads = np.vstack(all_grads)
        all_targets = np.hstack(all_targets)
        
        
        return all_grads, all_targets

