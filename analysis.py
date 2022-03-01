import argparse
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import pairwise_distances
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
from lazyGreedy import lazy_greedy_heap
from fl_mnist import FacilityLocationMNIST
import REL_model
from covid_ct_dataset import CovidCTDataset

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.1)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=400)
parser.add_argument('--model', type=str, default='LeNet')
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

device = torch.device("cpu")

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

best_acc = np.zeros(10)
for i in range(10):
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

    if args.dataset == 'linear' or args.dataset == 'nonlinear':
        num_classes = 2
        num_gradual_cdr = 2
        batch_size = len(noise_train_dataset)
        test_images = torch.load('{d}_data/test_images.pt'.format(d=args.dataset))
        test_labels = torch.load('{d}_data/test_labels.pt'.format(d=args.dataset))
        test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)

    noise_train_loader = torch.utils.data.DataLoader(
        noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)
    noise_val_loader = torch.utils.data.DataLoader(
        noise_val_dataset, batch_size=len(noise_val_dataset), drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)
    train_and_val_loader = torch.utils.data.DataLoader(
        train_and_val_dataset, batch_size=len(train_and_val_dataset), drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=len(test_dataset))

    # load = np.load('best_pred_outlier_{d}_Trimming_{nt}_{nr}.npy'.format(d=args.dataset, 
    #                                                 nt=args.noise_type, nr=args.noise_rate))
    # pred_outliers = load.astype(np.int)
    # new_images = np.delete(train_and_val_images, pred_outliers, axis=0)
    # new_labels = np.delete(train_and_val_labels, pred_outliers, axis=0)

    # new_dataset = NoisyDataset(
    #     new_images, new_labels, target_transform=tools.transform_target)
    # new_loader = torch.utils.data.DataLoader(
    #     new_dataset, batch_size=len(new_dataset), drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)

    n = len(train_and_val_labels) * args.noise_rate
    outliers = np.random.choice(len(train_and_val_labels) ,int(n),replace=False)
    new_images = np.delete(train_and_val_images, outliers, axis=0)
    new_labels = np.delete(train_and_val_labels, outliers, axis=0)
    
    plot.save_fig(train_and_val_images, train_and_val_labels, new_images,new_labels)
    print(new_images.shape, new_labels.shape)
    #assert False

    new_dataset = NoisyDataset(
        new_images, new_labels, target_transform=tools.transform_target)
    new_loader = torch.utils.data.DataLoader(
        new_dataset, batch_size=len(new_dataset), drop_last=False, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    if args.dataset == 'linear':
        #model = Logistic()
        model = NeuralNetLinear()
    elif args.dataset == 'nonlinear':
        model = NeuralNetwork()
        

    model = model.to(device)
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

    outlier_id = np.where(train_and_val_labels !=
                            train_and_val_labels_without_noise)[0]
    num_outliers = len(outlier_id)

    TrainLoss = np.zeros(args.epoch)
    TrainAccuracy = np.zeros(args.epoch)
    TestLoss = np.zeros(args.epoch)
    TestAccuracy = np.zeros(args.epoch)
    ValidationLoss = np.ones(args.epoch)  # np.empty(args.epoch)
    OutlierDetectionAccuracy = np.zeros(args.epoch)


    for epoch in range(args.start_epoch, args.epoch):
        start = time.time()
        if args.method == 'CE':
            ce = CE()
            train_loss, train_acc = ce.train(
                new_loader, model, device, criterion, optimizer)

        test_loss, test_acc = eval_on_holdout.eval_on_holdout_data(
            test_loader, model, device, criterion)

        outlier_detection_accuracy, _ = eval_on_holdout.eval_outlier_detection(
            train_and_val_images, train_and_val_labels, train_and_val_labels_without_noise, model, device)
        
        print('epoch %d, train_loss: %.8f train_acc: %f test_loss: %f test_acc: %f' %
                (epoch+1, train_loss, train_acc, test_loss, test_acc))
        print("outlier_detection_accuracy: %.4f" % (outlier_detection_accuracy))
        TrainLoss[epoch] = train_loss
        TrainAccuracy[epoch] = train_acc
        TestLoss[epoch] = test_loss
        TestAccuracy[epoch] = test_acc
        #ValidationLoss[epoch] = val_loss
        OutlierDetectionAccuracy[epoch] = outlier_detection_accuracy
        #PredOutliers[epoch] = pred_outliers

        end = time.time() - start
        #print(end)


    if args.method == 'CRUST':
        test_acc_max = TestAccuracy[args.epoch-1]
    elif args.method == 'Trimming' or args.method == 'Trimming_minibatch':
        test_acc_max = OutlierDetectionAccuracy[np.argmin(TrainLoss)]
        #test_acc_max = test_acc
    else:
        #test_acc_max = TestAccuracy[np.argmin(ValidationLoss)]
        test_acc_max = OutlierDetectionAccuracy[np.argmin(TrainLoss)]
    print('Best Accuracy', test_acc_max)
    best_acc[i] = test_acc_max



    np.savez('{d}_analysis/random_{me}_{nt}_{nr}_result_{num}'.format(d=args.dataset, me=args.method,
            nt=args.noise_type, nr=args.noise_rate,num=i), train_loss_result=TrainLoss,
            train_acc_result=TrainAccuracy, test_loss_result=TestLoss,
            test_acc_result=TestAccuracy, val_loss_result=ValidationLoss,
            outlier_detection_accuracy=OutlierDetectionAccuracy)

np.save('analysis_{d}/random_{nt}_{nr}_{m}_result'.format(
        d=args.dataset, nt=args.noise_type, m=args.method, nr=args.noise_rate), best_acc)
