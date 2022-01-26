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


def dataset_split(images, labels, dataset='mnist', noise_type=None, noise_rate=0.5, split_per=0.9, random_seed=1, num_classes=10):

    clean_labels = labels.numpy().copy()
    labels_with_newaxis = labels[:, np.newaxis]
    
    if noise_type == 'symmetric':
        noisy_labels = utils.noisify_multiclass_symmetric(
            labels_with_newaxis, noise_rate, random_seed, num_classes)
    elif noise_type == 'island':
        if dataset == 'linear':
            noisy_labels = utils.noise_island_linear(images, labels_with_newaxis, noise_rate, -2.6, -0.2)
        elif dataset == 'nonlinear':
            noisy_labels, noise_point = utils.noise_island_circle(images, labels_with_newaxis, noise_rate)
    elif noise_type == 'asymmetric':
        if dataset == 'mnist':
            noisy_labels = utils.noisify_multiclass_asymmetric(
                labels_with_newaxis, noise_rate, random_seed, num_classes)
        elif dataset == 'fmnist':
            noisy_labels = utils.noisify_multiclass_asymmetric_fmnist(
                labels_with_newaxis, noise_rate, random_seed, num_classes)
        elif dataset == 'cifar10':
            noisy_labels = utils.noisify_multiclass_asymmetric_cifar10(
                labels_with_newaxis, noise_rate, random_seed, num_classes)
        elif dataset == 'cifar100':
            noisy_labels = utils.noisify_multiclass_asymmetric_cifar100(
                labels_with_newaxis, noise_rate, random_seed, num_classes)
        elif dataset == 'linear':
            noisy_labels = utils.noise_asymmetric_linear(labels_with_newaxis, noise_rate)
        elif dataset == 'nonlinear':
            noisy_labels = utils.noise_asymmetric_nonlinear(
                labels_with_newaxis, noise_rate)
    elif noise_type == 'instance':
        if dataset == 'mnist':
            dataset = zip(images, labels)
            noisy_labels = utils.get_instance_noisy_label(
                noise_rate = noise_rate, dataset=dataset, labels = labels, num_classes=10, feature_size=784, norm_std=0.1, seed=random_seed)
        elif dataset == 'fmnist':
            dataset = zip(images, labels)
            noisy_labels = utils.get_instance_noisy_label(
                noise_rate = noise_rate, dataset=dataset, labels=labels, num_classes=10, feature_size=784, norm_std=0.1, seed=random_seed)
        elif dataset == 'cifar10':
            dataset = zip(images, labels)
            noisy_labels = utils.get_instance_noisy_label(
                noise_rate = noise_rate, dataset=dataset, labels=labels, num_classes=10, feature_size=3072, norm_std=0.1, seed=random_seed)
        elif dataset == 'cifar100':
            dataset = zip(images, labels)
            noisy_labels = utils.get_instance_noisy_label(
                noise_rate = noise_rate, dataset=dataset, labels=labels, num_classes=100, feature_size=3072, norm_std=0.1, seed=random_seed)
        elif dataset == 'covid_ct':
            dataset = zip(images, labels)
            noisy_labels = utils.get_instance_noisy_label(
                noise_rate = noise_rate, dataset=dataset, labels=labels, num_classes=2, feature_size=3*224*224, norm_std=0.1, seed=random_seed)
        elif dataset == 'linear' or dataset == 'nonlinear':
            dataset = zip(images, labels)
            noisy_labels = utils.get_instance_noisy_label(
                noise_rate = noise_rate, dataset=dataset, labels=labels, num_classes=2, feature_size=2, norm_std=0.1, seed=random_seed)
            
    
    noisy_labels = noisy_labels.squeeze() 
    
    num_samples = int(noisy_labels.shape[0])
    np.random.seed(random_seed)
    train_set_index = np.random.choice(
        num_samples, int(num_samples*split_per), replace=False)
    index = np.arange(images.shape[0])
    val_set_index = np.delete(index, train_set_index)

    train_set, val_set = images[train_set_index,:], images[val_set_index, :]
    train_labels, val_labels = noisy_labels[train_set_index], noisy_labels[val_set_index]
    
    
    return train_set, val_set, train_labels, val_labels, images, noisy_labels, clean_labels

def transform_target(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target
