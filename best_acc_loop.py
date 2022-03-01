import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import pairwise_distances #crust
import numpy as np
import time
from noisy_dataset import NoisyDataset
from add_noise_split_crust import Noise_dataset_split_crust
import tools
from REL_model import LeNet
from REL_model import NeuralNetwork
from REL_model import NeuralNetLinear
from cdr_method import CDR
from crust_method import CRUST
from ce_method import CE
from trimming_method import Trimming
import eval_on_holdout
import plot
from lazyGreedy import lazy_greedy_heap #crust
from fl_mnist import FacilityLocationMNIST#crust
import REL_model
import models #crust
from covid_ct_dataset import CovidCTDataset

np.random.seed(3523421)
torch.manual_seed(3523421)

model_names = sorted(name for name in models.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--model_name', type=str, default='resnet50')
parser.add_argument('--weight_decay', type=float, help='l2', default=0.01)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
parser.add_argument('--outlier_ratio', type=float, default=0.2)
parser.add_argument('--optim', type=str, default='sgd')

args = parser.parse_args()

#device = torch.device("cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, )), ])
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


test_transformer_covid = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

best_acc = np.zeros(3)
for i in range(3):
    train_images = torch.load('{d}_data/{d}_{nt}_{nr}_train_images.pt'.format(
        d=args.dataset, nt=args.noise_type, nr=args.noise_rate))
    val_images = torch.load('{d}_data/{d}_{nt}_{nr}_val_images.pt'.format(
        d=args.dataset, nt=args.noise_type, nr=args.noise_rate))
    train_and_val_images = torch.load('{d}_data/{d}_{nt}_{nr}_train_and_val_images.pt'.format(
        d=args.dataset, nt=args.noise_type, nr=args.noise_rate))
    labels = np.load('{d}_data/{d}_{nt}_{nr}_labels.npz'.format(
        d=args.dataset, nt=args.noise_type, nr=args.noise_rate))
    train_labels = labels['train_labels']
    val_labels = labels['val_labels']
    train_and_val_labels = labels['train_and_val_labels']
    train_and_val_labels_without_noise = labels['train_and_val_labels_without_noise']


    noise_train_dataset = NoisyDataset(
        train_images, train_labels, target_transform=tools.transform_target)
    noise_val_dataset = NoisyDataset(
        val_images, val_labels,  target_transform=tools.transform_target)
    train_and_val_dataset = NoisyDataset(
        train_and_val_images, train_and_val_labels,  target_transform=tools.transform_target)


    if args.dataset == 'mnist':
        num_gradual_cdr = 10
        num_classes = 10
        batch_size = 32
        #batch_size = len(train_dataset)
        test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True,
                                                transform=transform)
    elif args.dataset == 'fmnist':
        num_gradual_cdr = 10
        num_classes = 10
        batch_size = 32
        test_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=False, download=True,
                                                        transform=transform)
    elif args.dataset == 'cifar10':
        if args.method == 'CRUST':
            batch_size = 128
        else:
            batch_size = 64
        num_gradual_cdr = 20
        num_classes = 10
        test_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, download=True,
                                                    transform=transform_test)
    elif args.dataset == 'cifar100':
        if args.method == 'CRUST':
            batch_size = 128
        else:
            batch_size = 64
        num_gradual_cdr = 20
        num_classes = 100
        test_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=False, download=True,
                                                    transform=transform_test)
    elif args.dataset == 'linear' or args.dataset == 'nonlinear':
        num_classes = 2
        num_gradual_cdr = 2
        batch_size = len(noise_train_dataset)
        test_images = torch.load('{d}_data/test_images.pt'.format(d=args.dataset))
        test_labels = torch.load('{d}_data/test_labels.pt'.format(d=args.dataset))
        test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    elif args.dataset == 'covid_ct':
        num_classes = 2
        batch_size = 16
        num_gradual_cdr = 2
        test_dataset = CovidCTDataset(root_dir='new_data/CT_image',
                                        txt_COVID='new_data/Covid_txt/testCT_COVID.txt',
                                        txt_NonCOVID='new_data/NonCovid_txt/testCT_NonCOVID.txt',
                                        transform=test_transformer_covid)


    noise_train_loader = torch.utils.data.DataLoader(
        noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)
    noise_val_loader = torch.utils.data.DataLoader(
        noise_val_dataset, batch_size=len(noise_val_dataset), drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_and_val_loader = torch.utils.data.DataLoader(
        train_and_val_dataset, batch_size=len(train_and_val_dataset), drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset))


    if args.dataset == 'mnist':
        model = LeNet()
    elif args.dataset == 'fmnist':
        model = REL_model.ResNet50(input_channel=1, num_classes=10)
    elif args.dataset == 'cifar10':
        if args.method == 'CRUST':
            model = models.__dict__[args.arch](num_classes=num_classes)
        else:
            model = REL_model.ResNet50(input_channel=3, num_classes=10)
    elif args.dataset == 'cifar100':
        if args.method == 'CRUST':
            model = models.__dict__[args.arch](num_classes=num_classes)
        else:
            model = REL_model.ResNet50(input_channel=3, num_classes=100)
    elif args.dataset == 'linear':
        #model = Logistic()
        model = NeuralNetLinear()
    elif args.dataset == 'nonlinear':
        model = NeuralNetwork()
        #model = Logistic()
    elif args.dataset == 'covid_ct':
        model = torchvision.models.resnet50(pretrained=True)#models.resnet50(pretrained=True)

    model = model.to(device)
    #optimizer 
    optimizer = torch.optim.SGD(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
    
    criterion = nn.CrossEntropyLoss()


    if args.method == 'CDR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 80], gamma=0.1)
    elif args.method == 'CRUST':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[80, 100], last_epoch=args.start_epoch - 1)
        weights = [1] * len(noise_train_dataset)
        weights = torch.FloatTensor(weights)
    elif args.method == 'Trimming':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    TrainLoss = np.zeros(args.epoch)
    TrainAccuracy = np.zeros(args.epoch)
    TestLoss = np.zeros(args.epoch)
    TestAccuracy = np.zeros(args.epoch)
    ValidationLoss = np.ones(args.epoch)  # np.empty(args.epoch)
    OutlierDetectionAccuracy = np.zeros(args.epoch)

    start = time.time()
    for epoch in range(args.start_epoch, args.epoch):
        st = time.time()
        if args.method == 'CDR':
            cdr = CDR()
            train_loss, train_acc = cdr.train_rel(model, device, noise_train_loader, epoch, args.noise_rate,
                                                num_gradual_cdr, criterion, optimizer)
        elif args.method == 'CE':
            ce = CE()
            train_loss, train_acc = ce.train(
                noise_train_loader, model, device, criterion, optimizer)
        elif args.method == 'CRUST':
            crust = CRUST()
            if epoch >= 5:
                noise_train_dataset.switch_data()
                grads_all, labels = crust.estimate_grads(
                    trainval_loader, model, device, criterion)
                ssets = []
                weights = []
                for c in range(num_classes):

                    sample_ids = np.where((labels == c) == True)[0]
                    grads = grads_all[sample_ids]

                    dists = pairwise_distances(grads)
                    weight = np.sum(dists < r_crust, axis=1)
                    V = range(len(grads))
                    F = FacilityLocationMNIST(V, D=dists)
                    B = int(fl_ratio_crust * len(grads))
                    sset, vals = lazy_greedy_heap(F, V, B)

                    #if len(sset) > 0:
                    weights.extend(weight[sset].tolist())
                    sset = sample_ids[np.array(sset)]
                    ssets += list(sset)

                weights = torch.FloatTensor(weights)
                noise_train_dataset.adjust_base_indx_tmp(ssets)

            if epoch > 5:
                train_loss, train_acc = crust.train_crust(
                    noise_train_loader, model, device, criterion, weights, optimizer, fetch=True)
            else:
                train_loss, train_acc = crust.train_crust(
                    noise_train_loader, model, device, criterion, weights, optimizer, fetch=False)

        elif args.method == 'Trimming':
            trimming = Trimming()
            train_loss, train_acc = trimming.train_batch(
                    train_and_val_loader, model, device, optimizer, args.outlier_ratio)
        elif args.method == 'Trimming_change':
            trimming = Trimming()
            if epoch >= 300:
                args.outlier_ratio = 0.2
            elif epoch >= 200:
                args.outlier_ratio = 0.1
            elif epoch >= 100:
                args.outlier_ratio = 0.05
            else:
                args.outlier_ratio = 0.01 
            print('outlier:',args.outlier_ratio)
            train_loss, train_acc = trimming.train_batch(
                train_and_val_loader, model, device, optimizer, args.outlier_ratio)
        elif args.method == 'Trimming_minibatch':
            trimming = Trimming()
            if args.dataset == 'covid_ct':
                train_loss, train_acc = trimming.train_minibatch_covid(
                    train_and_val_images, train_and_val_labels, model, device, optimizer, args.outlier_ratio, 16)
            else:
                train_loss, train_acc = trimming.train_minibatch(
                    train_and_val_images, train_and_val_labels, model, device, optimizer, args.outlier_ratio, 32)

        if args.method == 'Trimming' or args.method == 'Trimming_minibatch'or args.method == 'Trimming_change':
            val_loss = np.NAN
        else:
            val_loss, val_acc = eval_on_holdout.eval_on_holdout_data(
                noise_val_loader, model, device, criterion)
        test_loss, test_acc = eval_on_holdout.eval_on_holdout_data(
            test_loader, model, device, criterion)

        outlier_detection_accuracy, _ = eval_on_holdout.eval_outlier_detection(
            train_and_val_images, train_and_val_labels, train_and_val_labels_without_noise, model, device)
        if args.method == 'Trimming' or args.method == 'Trimming_minibatch'or args.method == 'Trimming_change':
            print('epoch %d, train_loss: %.8f train_acc: %f test_loss: %f test_acc: %f' %
                (epoch+1, train_loss, train_acc, test_loss, test_acc))
        else:
            print('epoch %d, train_loss: %f train_acc: %f val_loss: %f val_acc: %f test_loss: %f test_acc: %f' %
                (epoch+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
        print("outlier_detection_accuracy: %.4f" % (outlier_detection_accuracy))
        TrainLoss[epoch] = train_loss
        TrainAccuracy[epoch] = train_acc
        TestLoss[epoch] = test_loss
        TestAccuracy[epoch] = test_acc
        ValidationLoss[epoch] = val_loss
        OutlierDetectionAccuracy[epoch] = outlier_detection_accuracy
            
        if args.method == 'CDR' or args.method == 'CRUST':
            lr_scheduler.step()
        e = time.time() - st
        print(e)
    end = time.time() - start


    if args.method == 'Trimming' or args.method == 'Trimming_minibatch'or args.method == 'Trimming_change':
        test_acc_max = OutlierDetectionAccuracy[np.argmin(TrainLoss)]
        #test_acc_max = test_acc
    else:
        #test_acc_max = TestAccuracy[np.argmin(ValidationLoss)]
        test_acc_max = OutlierDetectionAccuracy[np.argmin(ValidationLoss)]
    print('Best Accuracy', test_acc_max)
    print(end)
    best_acc[i] = test_acc_max

    np.savez('{d}_result/{me}_{nt}_{nr}_result_{n}'.format(d=args.dataset, me=args.method,
                                                    nt=args.noise_type, nr=args.noise_rate,n=i), train_loss_result=TrainLoss,
            train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
            test_acc_result=TestAccuracy, val_loss_result=ValidationLoss,
            outlier_detection_accuracy=OutlierDetectionAccuracy)

np.save('best_acc_{d}/{nt}_{nr}_{m}_result'.format(
    d=args.dataset, nt=args.noise_type, m=args.method, nr=args.noise_rate), best_acc)
