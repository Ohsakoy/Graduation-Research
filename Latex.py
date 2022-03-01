import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Training')

parser.add_argument('--noise_rate', type=float,
                    help='overall corruption rate, should be less than 1', default=0.2)
parser.add_argument('--noise_type', type=str,
                    help='[instance, symmetric, asymmetric]', default='symmetric')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--dataset', type=str, default='mnist')
parser.add_argument('--method', type=str, default='CDR')
args = parser.parse_args()

methods = ['CE', 'CDR','Trimming','Trimming_change','Trimming_minibatch']

noise_rate =[0.2, 0.4] #
#noise_rate=[0.01,0.05, 0.1,0.2]

for _ , method in enumerate (methods):
    print(method, end = " ")
    for i, noise in enumerate(noise_rate):
        best_acc = np.load('best_acc_{d}/{nt}_{nr}_{m}_result.npy'.format(
            d=args.dataset,nt=args.noise_type, m=method, nr=noise))
        #print(best_acc)
        avg = np.mean(best_acc, axis=0)
        std = np.std(best_acc, axis=0)
        print('& $' , end=' ')
        print('%.4f \\pm %.4f' % (avg, std),  end=' ')
        print('$', end=' ')
    print('\\\\')

# print('Trimming_minibatch', end=" ")

# best_acc= np.zeros(9)
# for i in range(9):
#     results = np.load('{d}_result/{me}_{nt}_{nr}_result_{num}.npz'.format(d=args.dataset, me='Trimming_minibatch',
#         nt=args.noise_type, nr=args.noise_rate,num=i))
#     TrainLoss = results['train_loss_result']
#     OutlierDetectionAccuracy= results['outlier_detection_accuracy']#
#     best_acc[i] = OutlierDetectionAccuracy[np.argmin(TrainLoss)]

# avg = np.mean(best_acc, axis=0)
# std = np.std(best_acc, axis=0)
# print('& $' , end=' ')
# print('%.4f \\pm %.4f' % (avg, std),  end=' ')
# print('$', end=' ')


# best_acc = np.load('best_acc_{d}/{nt}_{nr}_{m}_result.npy'.format(
#             d=args.dataset,nt=args.noise_type, m='Trimming_minibatch', nr=noise))
# avg = np.mean(best_acc, axis=0)
# std = np.std(best_acc, axis=0)
# print('& $' , end=' ')
# print('%.4f \\pm %.4f' % (avg, std),  end=' ')
# print('$', end=' ')

# print('\\\\')