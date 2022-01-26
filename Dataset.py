import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import make_classification
from sklearn.datasets import make_gaussian_quantiles
from sklearn.model_selection import train_test_split
import numpy as np
from add_noise_split_crust import Noise_dataset_split_crust
import tools
import utils
import models
import plot
from covid_ct_dataset import CovidCTDataset


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
                    help='overall corruption rate, should be less than 1', default=0.5)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model', type=str, default='LeNet')
parser.add_argument('--weight_decay', type=float, help='l2', default=0.01)
parser.add_argument('--learning_rate', type=float,
                    help='momentum', default=0.01)
parser.add_argument('--momentum', type=float, help='momentum', default=0.9)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
parser.add_argument('--inlier_ratio', type=float, default=0.8)
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
train_transformer_covid = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if args.dataset == 'mnist':
    num_gradual_cdr = 10
    num_classes = 10

    transform_train = transform
    transform_test = transform
    train_dataset = torchvision.datasets.MNIST("MNIST", train=True, download=True,
                                               transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.MNIST("MNIST", train=False, download=True,
                                              transform=transform_test)
    batch_size = 32
    #batch_size = len(train_dataset)
elif args.dataset == 'fmnist':
    num_gradual_cdr = 10
    num_classes = 10
    batch_size = 32
    transform_train = transform
    transform_test = transform
    train_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=True, download=True,
                                                      transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.FashionMNIST("FashionMNIST", train=False, download=True,
                                                     transform=transform_test)
elif args.dataset == 'cifar10':
    if args.method == 'CRUST':
        batch_size = 128
    else:
        batch_size = 64
    num_gradual_cdr = 20
    num_classes = 10
    train_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=True, download=True,
                                                 transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.CIFAR10("CIFAR10", train=False, download=True,
                                                transform=transform_test)
elif args.dataset == 'cifar100':
    if args.method == 'CRUST':
        batch_size = 128
    else:
        batch_size = 64
    num_gradual_cdr = 20
    num_classes = 100
    train_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=True, download=True,
                                                  transform=transform_train, target_transform=tools.transform_target)
    test_dataset = torchvision.datasets.CIFAR100("CIFAR100", train=False, download=True,
                                                 transform=transform_test)
elif args.dataset == 'linear':
    X, y = make_classification(n_samples=3000, n_features=2, n_informative=1, n_redundant=0,
                            n_classes=2, n_clusters_per_class=1, class_sep=3, random_state=12, flip_y=0)
elif args.dataset == 'nonlinear':
    X, y = make_gaussian_quantiles(
        cov=3., n_samples=3000, n_features=2, n_classes=2, random_state=1)
    point = np.where((X[:, 0] < 2) & (X[:, 1] > -2))[0]
    y[point] = y[point] & 0
elif args.dataset == 'covid_ct':
    num_classes = 2
    train_dataset = CovidCTDataset(root_dir='new_data/CT_image',
                                txt_COVID='new_data/Covid_txt/train_and_valCT_COVID.txt',
                                txt_NonCOVID='new_data/NonCovid_txt/train_and_valCT_NonCOVID.txt',
                                transform=train_transformer_covid)

if args.dataset == 'linear' or args.dataset == 'nonlinear':
    num_classes = 2
    num_gradual_cdr = 2
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.5, random_state=0)
    train_X = torch.Tensor(train_X)
    test_X = torch.Tensor(test_X)
    train_y = torch.LongTensor(train_y)
    test_y = torch.LongTensor(test_y)
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    batch_size = len(train_dataset)
    #torch.save(test_X, '{d}_data/test_images.pt'.format(d=args.dataset))
    #torch.save(test_y, '{d}_data/test_labels.pt'.format(d=args.dataset))
    #batch_size = 32


SPLIT_TRAIN_VAL_RATIO = 0.9
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=True)

for images, labels in train_loader:
    images_train = images  # .to(device)
    labels_train = labels
    #print(images.shape)

train_images, val_images, train_labels, val_labels, train_and_val_images, train_and_val_labels, train_and_val_labels_without_noise = tools.dataset_split(
    images_train, labels_train, args.dataset, args.noise_type, args.noise_rate, SPLIT_TRAIN_VAL_RATIO, random_seed=1, num_classes=num_classes)

# plot.save_fig(train_and_val_images, train_and_val_labels,
#             train_and_val_images, train_and_val_labels_without_noise)

#data loader)
# if args.method != 'CRUST':
#     SPLIT_TRAIN_VAL_RATIO = 0.9
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=len(train_dataset), drop_last=False, shuffle=True)
#     for images, labels in train_loader:
#         images_train = images  # .to(device)
#         labels_train = labels
    
#     train_images, val_images, train_labels, val_labels, train_and_val_images, train_and_val_labels, train_and_val_labels_without_noise = tools.dataset_split(
#         images_train, labels_train, args.dataset, args.noise_type, args.noise_rate, SPLIT_TRAIN_VAL_RATIO, random_seed=1, num_classes=num_classes)
    
#     # plot.save_fig(train_and_val_images, train_and_val_labels,
#     #               train_and_val_images, train_and_val_labels_without_noise)

# elif args.method == 'CRUST':
#     SPLIT_TRAIN_VAL_RATIO = 1.0
#     r_crust = 2.0
#     fl_ratio_crust = 0.5
#     noise_train_dataset = Noise_dataset_split_crust(train_dataset.data, train_dataset.targets, train=True,
#                                                     transform=transform_train, target_transform=tools.transform_target,
#                                                     noise_type=args.noise_type, noise_rate=args.noise_rate, dataset=args.dataset,
#                                                     split_per=SPLIT_TRAIN_VAL_RATIO, random_seed=1, num_class=num_classes)
#     trainval_loader = torch.utils.data.DataLoader(
#         noise_train_dataset, batch_size=batch_size, drop_last=False, shuffle=False, num_workers=args.workers, pin_memory=True)


#image 保存　tensor
torch.save(train_images,'{d}_data/{d}_{nt}_{nr}_train_images.pt'.format(
    d=args.dataset,nt=args.noise_type, nr=args.noise_rate))
torch.save(val_images, '{d}_data/{d}_{nt}_{nr}_val_images.pt'.format(
    d=args.dataset, nt=args.noise_type, nr=args.noise_rate))
torch.save(train_and_val_images, '{d}_data/{d}_{nt}_{nr}_train_and_val_images.pt'.format(
    d=args.dataset, nt=args.noise_type, nr=args.noise_rate))

#label 保存 numpy
np.savez('{d}_data/{d}_{nt}_{nr}_labels.npz'.format(
    d=args.dataset, nt=args.noise_type, nr=args.noise_rate), train_labels=train_labels,
    val_labels=val_labels, train_and_val_labels=train_and_val_labels,
    train_and_val_labels_without_noise=train_and_val_labels_without_noise)

# np.save('{d}_data/{d}_{nt}_{nr}_train_labels.npy'.format(
#     d=args.dataset, nt=args.noise_type, nr=args.noise_rate), train_labels)
# np.save('{d}_data/{d}_{nt}_{nr}_val_labels.npy'.format(
#     d=args.dataset, nt=args.noise_type, nr=args.noise_rate), val_labels)
# np.save('{d}_data/{d}_{nt}_{nr}_train_and_val_labels.npy'.format(
#     d=args.dataset, nt=args.noise_type, nr=args.noise_rate), train_and_val_labels)
# np.save('{d}_data/{d}_{nt}_{nr}_train_and_val_labels_without_noise.npy'.format(
#     d=args.dataset, nt=args.noise_type, nr=args.noise_rate), train_and_val_labels_without_noise)

