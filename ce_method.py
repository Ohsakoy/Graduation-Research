import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from eval import Evaluation_Method

class CE():
    def train(self,train_loader, clf, device, criterion, optimizer):
        clf.train()
        eval_method = Evaluation_Method()
        for item in train_loader:
            images = item[0]
            labels = item[1]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs, _ = clf.forward(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = eval_method.get_result()
        return train_loss, train_acc
