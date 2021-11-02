import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad


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
            res.append(correct_k.mul_(1.0 / batch_size))
        return res

    def train_crust(self, train_loader, model, device, criterion, weights, optimizer, fetch):
        # switch to train mode
        model.train()
        train_total = 0
        train_correct = 0
        total_loss = 0
        total = 0

        for i, (input, target, index) in enumerate(train_loader):
            if fetch:
                input_b = train_loader.dataset.fetch(target)
                lam = np.random.beta(1, 0.1)
                input = lam * input + (1 - lam) * input_b
                
            c_weights = weights[index]
            c_weights = c_weights.type(torch.FloatTensor)
            c_weights = c_weights / c_weights.sum()

            #input = input.type(torch.FloatTensor)
            input = input.to(device)
            target = target.to(device)
            c_weights = c_weights.to(device)
            # if i == 0:
            #     print(input.shape)
            #     print(target.shape)
            
            
            output, _ = model(input)
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
        

    def estimate_grads(self, trainval_loader, model, device, criterion):
        # switch to train mode
        model.train()
        all_grads = []
        all_targets = []
        
        for i, bat in enumerate(trainval_loader):
            input, target, _ = bat
            all_targets.append(target)
            input = input.to(device)
            target = target.to(device)
            # if i==0:
            #     print(input.shape)
            #     print(target.shape)

            # compute output
            output, feat = model(input)

            loss = criterion(output, target).mean()
            est_grad = grad(loss, feat)
            all_grads.append(est_grad[0].detach().cpu().numpy())

        all_grads = np.vstack(all_grads)
        all_targets = np.hstack(all_targets)

        return all_grads, all_targets
