import torch.optim as optim
import torchvision.models as tv_models
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np

class CDR():
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
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

    def train_one_step(self, clf, images, labels, nonzero_ratio, clip, criterion, optimizer):
        clf.train()
        pred, _ = clf(images)
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
                        >= thresh)#.type(torch.FloatTensor)
                mask = mask * clip
                '''
                print("param.data = ",param.data.get_device())
                print("mask_device =",mask.get_device())
                print("param.grad = ", param.grad.data.get_device())
                '''
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
            prec, loss = self.train_one_step(
                clf, images, labels, clip, clip, criterion, optimizer)
            total_loss += loss.item()
            train_total += 1
            total += labels.size(0)
            train_correct += prec

        train_loss = float(total_loss)/float(total)
        train_acc = float(train_correct)/float(train_total)
        return train_loss, train_acc
