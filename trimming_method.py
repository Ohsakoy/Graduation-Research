from tracemalloc import stop
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import grad
from eval import Evaluation

import seaborn as sns
import matplotlib.pyplot as plt


class Trimming():
    def train_batch(self, train_and_val_loader, model, device,  optimizer, outlier_ratio):
        model.train()
        eval_method = Evaluation()
        for i, batch in enumerate(train_and_val_loader):
            images = batch[0]
            labels = batch[1]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            #print(images.shape)
            
            outputs = model.forward(images)
            n_batch = labels.size(0)
            
            #負の対数尤度
            negLogProbs = - outputs[torch.arange(n_batch), labels] + torch.logsumexp(outputs, dim=1) 
            
            sorted_negLogProbs, idx = torch.sort(negLogProbs, descending = False) 
            #ソートしたものの尤度が低い80%
            trimmed = sorted_negLogProbs[0: -int(n_batch * outlier_ratio)]

            #trim_id = idx[0: int(n_batch * outlier_ratio)]
            #trim_id = idx[0: -int(n_batch * outlier_ratio)]
            loss = torch.mean(trimmed)
            
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = eval_method.get_result()
        
        return train_loss, train_acc

    def train_batch_debug(self, train_and_val_loader, model, device,  optimizer, outlier_ratio):
        model.train()
        eval_method = Evaluation()
        for i, batch in enumerate(train_and_val_loader):
            images = batch[0]
            labels = batch[1]
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            outputs = model.forward(images)
            n_batch = labels.size(0)

            #負の対数尤度
            negLogProbs = - outputs[torch.arange(n_batch), labels] + torch.logsumexp(outputs, dim=1)
            sorted_negLogProbs, idx = torch.sort(negLogProbs, descending=False)
            #ソートしたものの尤度が低い80%
            trimmed = sorted_negLogProbs[0: -int(n_batch * outlier_ratio)]

            #trim_id = idx[0: int(n_batch * outlier_ratio)]
            #trim_id = idx[0: -int(n_batch * outlier_ratio)]
            loss = torch.mean(trimmed)

            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
            loss.backward()
            optimizer.step()
        train_loss, train_acc = eval_method.get_result()

        return train_loss, train_acc
    
    
    @staticmethod
    def trim_outliers(negLogProbs, n, outlier_ratio):
        sorted_negLogProbs, idx = torch.sort(negLogProbs, descending=False)

        trim_id = idx[0:-int(n * outlier_ratio)]
        r = torch.randperm(trim_id.size(0))
        trim_id = trim_id[r]
        return trim_id, idx
    
    def train_minibatch_covid(self, images, labels, model, device, optimizer, outlier_ratio, batch_size):
        images = images.to(device)
        labels = torch.LongTensor(labels).to(device)
        model.eval()
        eval_method = Evaluation()
        all_outputs =  torch.zeros((images.size(0),1000),requires_grad=False)
        for i in range(0, images.size(0), batch_size):
            output = model.forward(images[i:(i+batch_size)])
            output = output.detach()
            all_outputs[i:(i+batch_size)] = output
        
        n = labels.size(0)
        negLogProbs = - all_outputs[torch.arange(n), labels] + torch.logsumexp(all_outputs, dim=1)
        
        trim_id, idx = Trimming.trim_outliers(negLogProbs, n, outlier_ratio)
        # _, idx = torch.sort(negLogProbs, descending=False)
        # trim_id = idx[0: -int(n * outlier_ratio)]
        # r = torch.randperm(trim_id.size(0))
        # trim_id = trim_id[r]
        
        model.train()
        for i in range(batch_size, n-int(n * outlier_ratio), batch_size):
            images_batch = images[trim_id[0:batch_size],:]
            labels_batch = labels[trim_id[0:batch_size]]
            optimizer.zero_grad()
            outputs_batch = model.forward(images_batch)
            negLogProbs_batch = -outputs_batch[torch.arange(batch_size), labels_batch] + torch.logsumexp(outputs_batch, dim=1)
            loss_batch = torch.mean(negLogProbs_batch)
            
            eval_method.loss(loss_batch)
            eval_method.acc(labels_batch, outputs_batch)
            loss_batch.backward()
            optimizer.step()
            
            In = trim_id[batch_size:]
            Out = idx[-int(n * outlier_ratio):]
            
            id = torch.cat((In,Out))
            #print("id: ",id.shape)
            # rand = torch.randperm(id.size(0))
            # id=id[rand]
            all_outputs =  torch.zeros((images[id].size(0),1000),requires_grad=False)
            for i in range(0, images.size(0), batch_size):
                num = id[i:(i+batch_size)]
                output = model.forward(images[num])
                output = output.detach()
                all_outputs[i:(i+batch_size)] = output
                
            #outputs = model.forward(images[id])
            #print(all_outputs.shape)
            
            negLogProbs = -all_outputs[torch.arange(labels[id].size(0)), labels[id]] + torch.logsumexp(all_outputs, dim=1)
                
            trim_id, idx = Trimming.trim_outliers(negLogProbs, n, outlier_ratio)
            trim_id = trim_id.detach()
            idx = idx.detach()
            # sorted_negLogProbs, idx = torch.sort(negLogProbs, descending=False)
            # trim_id = idx[0:-int(n * outlier_ratio)]
            # r = torch.randperm(trim_id.size(0))
            # trim_id = trim_id[r]
        
        train_loss, train_acc = eval_method.get_result()
        #assert False

        return train_loss, train_acc
    
    def train_minibatch(self, images, labels, model, device, optimizer, outlier_ratio, batch_size):
        images = images.to(device)
        labels = torch.LongTensor(labels).to(device)
        model.train()
        eval_method = Evaluation()
        outputs = model.forward(images)
        n = labels.size(0)
        negLogProbs = - outputs[torch.arange(n), labels] + torch.logsumexp(outputs, dim=1)
        
        trim_id, idx = Trimming.trim_outliers(negLogProbs, n, outlier_ratio)
        # _, idx = torch.sort(negLogProbs, descending=False)
        # trim_id = idx[0: -int(n * outlier_ratio)]
        # r = torch.randperm(trim_id.size(0))
        # trim_id = trim_id[r]
        
        for i in range(batch_size, n-int(n * outlier_ratio), batch_size):
            images_batch = images[trim_id[0:batch_size],:]
            labels_batch = labels[trim_id[0:batch_size]]
            optimizer.zero_grad()
            outputs_batch = model.forward(images_batch)
            negLogProbs_batch = -outputs_batch[torch.arange(batch_size), labels_batch] + torch.logsumexp(outputs_batch, dim=1)
            loss_batch = torch.mean(negLogProbs_batch)
            
            eval_method.loss(loss_batch)
            eval_method.acc(labels_batch, outputs_batch)
            loss_batch.backward()
            optimizer.step()
            
            In = trim_id[batch_size:]
            Out = idx[-int(n * outlier_ratio):]
            
            id = torch.cat((In,Out))
            #print("id: ",id.shape)
            # rand = torch.randperm(id.size(0))
            # id=id[rand]
            
            outputs = model.forward(images[id])
            negLogProbs = - \
                outputs[torch.arange(labels[id].size(0)), labels[id]] + \
                torch.logsumexp(outputs, dim=1)
                
            trim_id, idx = Trimming.trim_outliers(negLogProbs, n, outlier_ratio)
            # sorted_negLogProbs, idx = torch.sort(negLogProbs, descending=False)
            # trim_id = idx[0:-int(n * outlier_ratio)]
            # r = torch.randperm(trim_id.size(0))
            # trim_id = trim_id[r]
        
        train_loss, train_acc = eval_method.get_result()
        #assert False

        return train_loss, train_acc
