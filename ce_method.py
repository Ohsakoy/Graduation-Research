import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from eval import Evaluation
import plot
import seaborn as sns
import matplotlib.pyplot as plt
class CE():
    def train(self,train_loader, clf, device, criterion, optimizer):
        clf.train()
        
        eval_method = Evaluation()
        for item in train_loader:
            images = item[0]
            labels = item[1]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = clf.forward(images)
            #outputs, _ = clf.forward(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = eval_method.get_result()
        
        # images = images.to('cpu').detach().numpy().copy()
        # labels = labels.to('cpu').detach().numpy().copy()
        

        # X = images
        # f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        # sns.scatterplot(X[:, 0], X[:, 1], hue=labels, ax=ax1)

        # plt.savefig("Plot/CE_label.png")
        # assert(False)
        return train_loss, train_acc

