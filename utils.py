import torch.nn as nn
import os
import os.path
import torch
import torchvision
from math import inf
from scipy import stats
import copy
import hashlib
import errno
import numpy as np
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal


def multiclass_noisify(y, P, random_state=1):
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y



def noisify_multiclass_symmetric(y_train, noise, random_state, nb_classes):
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes-1):
            P[i, i] = 1. - n
        P[nb_classes-1, nb_classes-1] = 1. - n

        y_train_noisy = multiclass_noisify(
            np.array(y_train), P=P, random_state=random_state)
        actual_noise = (y_train_noisy != np.array(y_train)).mean()
        assert actual_noise > 0.0
        return y_train_noisy
    else:
        return y_train
    


def noisify_multiclass_asymmetric(y_train, noise, random_state=None, nb_classes=10):
    P = np.eye(10)
    n = noise

    # 2 -> 7
    P[2, 2], P[2, 7] = 1. - n, n

    # 5 <-> 6
    P[5, 5], P[5, 6] = 1. - n, n
    P[6, 6], P[6, 5] = 1. - n, n

    # 3 -> 8
    P[3, 3], P[3, 8] = 1. - n, n

    y_train_noisy = multiclass_noisify(
        np.array(y_train), P=P, random_state=random_state)
    actual_noise = (y_train_noisy != np.array(y_train)).mean()
    assert actual_noise > 0.0

    y_train = y_train_noisy

    return y_train


def get_instance_noisy_label(n, dataset, labels, num_classes, feature_size, norm_std, seed):
    
    label_num = num_classes
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    #torch.cuda.manual_seed(int(seed))

    P = []
    flip_distribution = stats.truncnorm(
        (0 - n) / norm_std, (1 - n) / norm_std, loc=n, scale=norm_std)
    flip_rate = flip_distribution.rvs(labels.shape[0])

    if isinstance(labels, list):
        labels = torch.FloatTensor(labels)
    #labels = labels.cuda()

    W = np.random.randn(label_num, feature_size, label_num)

    W = torch.FloatTensor(W)
    for i, (x, y) in enumerate(dataset):
        A = x.view(1, -1).mm(W[y]).squeeze(0)
        A[y] = -inf
        A = flip_rate[i] * F.softmax(A, dim=0)
        A[y] += 1 - flip_rate[i]
        P.append(A)
    P = torch.stack(P, 0).numpy()
    #np.save("transition_matrix.npy", P)
    
    l = [i for i in range(label_num)]
    new_label = [np.random.choice(l, p=P[i]) for i in range(labels.shape[0])]
    record = [[0 for _ in range(label_num)] for i in range(label_num)]

    for a, b in zip(labels, new_label):
        a, b = int(a), int(b)
        record[a][b] += 1
        
    pidx = np.random.choice(range(P.shape[0]), 1000)
    cnt = 0
    for i in range(1000):
        if labels[pidx[i]] == 0:
            a = P[pidx[i], :]
            cnt += 1
        if cnt >= 10:
            break
    return np.array(new_label)
