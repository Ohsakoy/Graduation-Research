from eval import Evaluation
from torch.autograd import grad
import numpy as np
import torch.nn.functional as F
import torch
import seaborn as sns
import matplotlib.pyplot as plt


def eval_on_holdout_data(loader, model, device, criterion):
    model.eval()
    eval_method = Evaluation()
    with torch.no_grad():
        for item in loader:
            images = item[0]
            labels = item[1]
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            #outputs, _ = model(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
    loss_final, acc_final = eval_method.get_result()
    return loss_final, acc_final


def eval_outlier_detection(train_and_val_images, train_and_val_labels, train_and_val_labels_without_noise, model, device):
    
    outlier_id = np.where(train_and_val_labels !=
                        train_and_val_labels_without_noise)[0]
    num_outliers = len(outlier_id)
    #print(outlier_id)
    num = 0
    model.eval()
    with torch.no_grad():
        images = train_and_val_images.to(device)
        labels = torch.LongTensor(train_and_val_labels).to(device)
        outputs = model.forward(images)
        #outputs, _ = model.forward(images)
        _, pred = torch.max(outputs, 1)

        n_batch = labels.size(0)
        #負の対数尤度
        negLogProbs = - outputs[torch.arange(n_batch), labels] + torch.logsumexp(outputs, dim=1)
        _, idx = torch.sort(negLogProbs, descending=True)
        pred_outlier = idx[0: int(num_outliers)].to('cpu').detach().numpy()
        #print(pre_outlier)
        for i in pred_outlier:
            if np.any(outlier_id == i) :
                num = num + 1
                
        if num == 0:
            outlier_detection_accuracy = 0.0
        else:
            outlier_detection_accuracy = float(num) / float(num_outliers)
        

            
        
        # images = images.to('cpu').detach().numpy().copy()
        # labels = labels.to('cpu').detach().numpy().copy()
        # pred = pred.to('cpu').detach().numpy().copy()
        # ''''
        # ire = torch.where(negLogProbs == 0.0)[0]
        # ire = ire.to('cpu').detach().numpy()
        # pred[ire] = 2
        # '''
        # pred[pre_outlier] = 2
        
        # X = images

        # f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))
        # sns.scatterplot(X[:, 0], X[:, 1], hue=labels, ax=ax1)
        # sns.scatterplot(X[:, 0], X[:, 1], hue=pred, ax=ax2)
        # plt.savefig("exp.png")

        
        return outlier_detection_accuracy, pred_outlier
