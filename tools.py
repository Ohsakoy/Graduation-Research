import numpy as np
import utils
import os
import numpy as np
import torch
import torchvision
from math import inf
from scipy import stats
from torchvision.transforms import transforms
import torch.nn.functional as F
import torch.nn as nn
import torch

# train set and val set split
def dataset_split(train_images, train_labels, noise_type=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):

    clean_train_labels = train_labels[:, np.newaxis]
    if noise_type == 'symmetric':
        noisy_labels = utils.noisify_multiclass_symmetric(
            clean_train_labels, noise_rate, random_seed, num_classes)
    elif noise_type == 'asymmetric':
        noisy_labels = utils.noisify_multiclass_asymmetric(
            clean_train_labels, noise_rate, random_seed, num_classes)
    else :
        dataset = zip(train_images, train_labels)
        noisy_labels = utils.get_instance_noisy_label(
            n=noise_rate, dataset=dataset, labels = train_labels, num_classes=10, feature_size=784, norm_std=0.1, seed=random_seed)
    
    noisy_labels = noisy_labels.squeeze()
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(
        num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(train_images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = train_images[train_set_index,:], train_images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]

    return train_set, val_set, train_labels, val_labels

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target
