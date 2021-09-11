import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
from random import choice
import random
import copy

class mnist_dataset(Data.Dataset):
    def __init__(self, images, labels, train=True, transform=None, target_transform=None, dataset='mnist', noise_type=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(images,labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)
        self.whole_data = np.array(self.train_data).copy()
        self.whole_targets = copy.deepcopy(np.array(self.train_labels).tolist())
    
    def adjust_base_indx_tmp(self, idx):
        new_data = self.whole_data[idx, ...]
        targets_np = np.array(self.whole_targets)
        new_targets = targets_np[idx].tolist()
        self.train_data = new_data
        self.train_labels = new_targets
        
    
    #crust data_load
    def fetch(self, targets):
        self.whole_data = np.array(self.train_data).copy()
        self.whole_targets = copy.deepcopy(
            np.array(self.train_labels).tolist())
        whole_targets_np = np.array(self.whole_targets)
        uniq_targets = np.unique(whole_targets_np)
        idx_dict = {}
        for uniq_target in uniq_targets:
            idx_dict[uniq_target] = np.where(
                whole_targets_np == uniq_target)[0]

        idx_list = []
        for target in targets:
            idx_list.append(np.random.choice(idx_dict[target.item()], 1))
        idx_list = np.array(idx_list).flatten()
        '''
        imgs = []
        for idx in idx_list:
            img = self.whole_data[idx]
            #img = Image.fromarray(img)
            img = self.transform(img)
            #img = self.target_transform(img)
            imgs.append(img[None, ...])
        train_data = torch.cat(imgs, dim=0)
        
        return train_data
        '''
        return self.whole_data[idx_list]
    
    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label,index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class fmnist_dataset(Data.Dataset):
    def __init__(self, images, labels, train=True, transform=None, target_transform=None, dataset='fmnist', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        

        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(images,
                                                                                                labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]
        else:
            img, label = self.val_data[index], self.val_labels[index]
        #img = torch.from_numpy(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class cifar10_dataset(Data.Dataset):
    def __init__(self,images, labels, train=True, transform=None, target_transform=None, dataset='cifar10', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(images,
                                                                                                labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)

        self.train_data = self.train_data.numpy()
        self.val_data = self.val_data.numpy()
        
        
        if self.train:
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        
        if self.train:
            img, label = self.train_data[index], self.train_labels[index]

        else:
            img, label = self.val_data[index], self.val_labels[index]

        # if self.transform is not None:
        #     img = self.transform(img)
            
        if self.target_transform is not None:
            label = self.target_transform(label)
            
            
        
        return img, label, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)


class cifar100_dataset(Data.Dataset):
    def __init__(self, images, labels ,train=True, transform=None, target_transform=None, dataset='cifar100', noise_type='symmetric', noise_rate=0.5, split_per=0.9, random_seed=1, num_class=100):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(images,
                                                                                                labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)

        if self.train:
            self.train_data = self.train_data.reshape((-1, 3, 32, 32))
            self.train_data = self.train_data.transpose((0, 2, 3, 1))

        else:
            self.val_data = self.val_data.reshape((-1, 3, 32, 32))
            self.val_data = self.val_data.transpose((0, 2, 3, 1))

    def __getitem__(self, index):

        if self.train:
            img, label = self.train_data[index], self.train_labels[index]

        else:
            img, label = self.val_data[index], self.val_labels[index]

        #img = Image.fromarray(img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, index

    def __len__(self):

        if self.train:
            return len(self.train_data)

        else:
            return len(self.val_data)
