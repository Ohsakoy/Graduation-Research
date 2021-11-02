import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
from random import choice
import random
import copy


class Noise_dataset_split(Data.Dataset):
    def __init__(self, images, labels, train=True, transform=None, target_transform=None, dataset=None, noise_type=None, noise_rate=None, split_per=0.9, random_seed=1, num_class=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        
        # clean images and noisy labels (training and validation)
        
        #labels = labels.cpu()
        
        self.data, self.val_data, self.targets, self.val_targets = tools.dataset_split(
            images, labels, dataset, noise_type, noise_rate, split_per, random_seed, num_class)
        

    def __getitem__(self, index):

        if self.train:
            img, label = self.data[index], self.targets[index]
        else:
            img, label = self.val_data[index], self.val_targets[index]

        if self.target_transform is not None:
            label = self.target_transform(label)

        return (img, label, index)

    def __len__(self):

        if self.train:
            return len(self.data)

        else:
            return len(self.val_data)
