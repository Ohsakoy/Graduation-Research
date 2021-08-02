import numpy as np
import torch.utils.data as Data
from PIL import Image
import tools
import torch
from random import choice
import random
import copy

class mnist_dataset(Data.Dataset):
    def __init__(self, images, labels, train=True, transform=None, target_transform=None, noise_type=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_class=10):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        # clean images and noisy labels (training and validation)
        self.train_data, self.val_data, self.train_labels, self.val_labels = tools.dataset_split(images,labels, noise_type, noise_rate, split_per, random_seed, num_class)
        
    def fetch(self, targets):
        self.whole_targets = copy.deepcopy(self.targets)
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
        imgs = []
        for idx in idx_list:
            img = self.whole_data[idx]
            img = Image.fromarray(img)
            img = self.transform(img)
            imgs.append(img[None, ...])
        train_data = torch.cat(imgs, dim=0)
        return train_data
        
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

