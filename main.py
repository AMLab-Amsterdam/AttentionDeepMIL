from __future__ import print_function

import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable

from dataloader import MnistBags
from model import Attention, GatedAttention

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--target_number', type=int, default=9, metavar='T',
                    help='bags have a positive labels if they contain at least one 9')
parser.add_argument('--mean_bag_length', type=int, default=10, metavar='ML',
                    help='average bag length')
parser.add_argument('--var_bag_length', type=int, default=2, metavar='VL',
                    help='variance of bag length')
parser.add_argument('--num_bags_train', type=int, default=200, metavar='NTrain',
                    help='number of bags in training set')
parser.add_argument('--num_bags_test', type=int, default=50, metavar='NTest',
                    help='number of bags in test set')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    print('\nGPU is ON!')

print('Load Train and Test Set')
loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                               mean_bag_length=args.mean_bag_length,
                                               var_bag_length=args.var_bag_length,
                                               num_bag=args.num_bags_train,
                                               seed=args.seed,
                                               train=True),
                                     batch_size=1,
                                     shuffle=True,
                                     **loader_kwargs)

test_loader = data_utils.DataLoader(MnistBags(target_number=args.target_number,
                                              mean_bag_length=args.mean_bag_length,
                                              var_bag_length=args.var_bag_length,
                                              num_bag=args.num_bags_test,
                                              seed=args.seed,
                                              train=False),
                                    batch_size=1,
                                    shuffle=False,
                                    **loader_kwargs)

print('Init Model')
if args.model=='attention':
    model = Attention()
elif args.model=='gated_attention':
    model = GatedAttention()
if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)


def train(epoch):
    model.train()
    train_loss = 0.
    train_error = 0.
    for batch_idx, (data, label) in enumerate(train_loader):
        bag_label = label[0]
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data, bag_label = Variable(data), Variable(bag_label)

        # reset gradients
        optimizer.zero_grad()
        # calculate loss and metrics
        loss, _ = model.calculate_objective(data, bag_label)
        train_loss += loss.data[0]
        error, _ = model.calculate_classification_error(data, bag_label)
        train_error += error
        # backward pass
        loss.backward()
        # step
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, Loss: {:.4f}, Train error: {:.4f}'.format(epoch, train_loss.cpu().numpy()[0], train_error))


def test():
    model.eval()
    test_loss = 0.
    test_error = 0.
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            bag_label = label[0]
            instance_labels = label[1]
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            data, bag_label = Variable(data), Variable(bag_label)
            loss, attention_weights = model.calculate_objective(data, bag_label)
            test_loss += loss.data[0]
            error, predicted_label = model.calculate_classification_error(data, bag_label)
            test_error += error

            if batch_idx < 5:  # plot bag labels and instance labels for first 5 bags
                bag_level = (bag_label.cpu().data.numpy()[0], int(predicted_label.cpu().data.numpy()[0][0]))
                instance_level = list(zip(instance_labels.numpy()[0].tolist(),
                                    np.round(attention_weights.cpu().data.numpy()[0], decimals=3).tolist()))

                print('\nTrue Bag Label, Predicted Bag Label: {}\n'
                    'True Instance Labels, Attention Weights: {}'.format(bag_level, instance_level))

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('\nTest Set, Loss: {:.4f}, Test error: {:.4f}'.format(test_loss.cpu().numpy()[0], test_error))


if __name__ == "__main__":
    print('Start Training')
    for epoch in range(1, args.epochs + 1):
        train(epoch)
    print('Start Testing')
    test()
