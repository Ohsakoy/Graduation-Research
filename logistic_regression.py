import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split

from REL_model import Logistic
from eval import Evaluation_Method
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import scipy.stats
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# import seaborn as sns
# import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.5)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=0.01)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if args.dataset == 'linear':
    X, y = make_classification(
        n_samples=10000, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=2, class_sep=2, random_state=17)
elif args.dataset == 'circle':
    X, y = make_gaussian_quantiles(
        cov=3., n_samples=10000, n_features=2,n_classes=2, random_state=1)

if args.dataset == 'linear' or args.dataset == 'circle':
    batch_size = 10
    SPLIT_TRAIN_VAL_RATIO = 0.9
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=0)
    train_X, val_X, train_Z, val_Z = train_test_split(
        train_X, train_y, train_size=SPLIT_TRAIN_VAL_RATIO, random_state=0)
    train_X = torch.Tensor(train_X)
    test_X = torch.Tensor(test_X)
    val_X = torch.Tensor(val_X)
    train_Z = torch.LongTensor(train_Z)
    test_y = torch.LongTensor(test_y)
    val_Z = torch.LongTensor(val_Z)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_Z)
    val_dataset = torch.utils.data.TensorDataset(val_X, val_Z)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size)



if args.dataset == 'linear' or args.dataset == 'circle':
    clf = Logistic()

clf = clf.to(device)
optimizer = torch.optim.SGD(
    clf.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
criterion = nn.CrossEntropyLoss()


def train(train_loader):
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


def eval_on_holdout_data(loader):
    clf.eval()
    eval_method = Evaluation_Method()
    with torch.no_grad():
        for item in loader:
            images = item[0]
            labels = item[1]
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = clf(images)
            loss = criterion(outputs, labels)
            eval_method.loss(loss)
            eval_method.acc(labels, outputs)
    loss_final, acc_final = eval_method.get_result()
    return loss_final, acc_final


TrainLoss = np.empty(args.epoch)
TrainAccuracy = np.empty(args.epoch)
TestLoss = np.empty(args.epoch)
TestAccuracy = np.empty(args.epoch)
ValidationLoss = np.empty(args.epoch)

for epoch in range(args.start_epoch, args.epoch):
    if args.method == 'CE':
        train_loss, train_acc = train(train_loader)

    if SPLIT_TRAIN_VAL_RATIO == 1.0:
        val_loss = np.NAN
    else:
        val_loss, _ = eval_on_holdout_data(val_loader)
    test_loss, test_acc = eval_on_holdout_data(test_loader)
    print('epoch %d, train_loss: %f train_acc: %f test_loss: %f test_acc: %f' %
        (epoch+1, train_loss, train_acc, test_loss, test_acc))
    TrainLoss[epoch] = train_loss
    TrainAccuracy[epoch] = train_acc
    TestLoss[epoch] = test_loss
    TestAccuracy[epoch] = test_acc
    ValidationLoss[epoch] = val_loss
    

if SPLIT_TRAIN_VAL_RATIO == 1.0:
    test_acc_max = TestAccuracy[args.epoch-1]
else:
    test_acc_max = TestAccuracy[np.argmin(ValidationLoss)]
print('Best Accuracy', test_acc_max)
