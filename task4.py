"""
@author: Gopal Krishna

Task 4 - Design your own experiment
"""

import sys
import warnings
from collections import defaultdict
from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from network import ExperimentNetwork
from torchsummary import summary

warnings.filterwarnings("ignore")

# method to get MNIST dataloaders
def get_dataloaders(batch_size):
    batch_size_train = batch_size_test = batch_size
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                           transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                           ])),
                           batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                            batch_size=batch_size_test, shuffle=True)

    return train_loader, test_loader

# method to build model according to different hyperparameters
def build_model(params, device):
    model = ExperimentNetwork(params['n_conv'], \
                            params['kernel_size'], \
                            params['n_dense'], \
                            params['drop'])
    model.to(device)

    print(summary(model, (1, 28, 28), device=device))
    print()

    return model
# method for train loop
def train(model, device, optimizer, train_loader):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

# method to compute accuracy for MNIST test set
def compute_accuracy(model, device, test_loader):
    model.eval()

    total = 0
    correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        value, pred = torch.max(output, 1)
        total += output.size(0)
        correct += torch.sum(pred==target)

    return correct*100./total

# method having training procedure
def train_model(model, device, n_epochs, train_loader, test_loader):
    batch_size = 1000
    learning_rate = 0.01
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    accuracies=[]

    for epoch in range(1, n_epochs + 1):
        train(model, device, optimizer, train_loader)
        accuracy  = compute_accuracy(model, device, test_loader)
        accuracies.append(accuracy.cpu())

    return accuracies

# method to genmerate comparitive plots for effects of different hyperparmaters
def plot_model_performance(acc_dict):
    cycol = cycle('bgrcmk')
    fig = plt.figure()
    for i, key in enumerate(acc_dict):
        plt.plot(acc_dict[key], color=next(cycol), label=key)

    plt.legend(loc=4)
    #fig.show()
    plt.savefig('accuracy.png', bbox_inches='tight')


def evaluate_convolution_filters(train_loader, test_loader, device):
    hyperparams = [[5,10], [10, 20], [20, 40], [40, 80], [80, 160]]
    params = {'n_conv':[10,20],
            'kernel_size':5,
            'n_dense':50,
            'drop':0.5}
    n_epochs = 5

    acc_dict = defaultdict(list)

    for i, hp in enumerate(hyperparams):
        print("Evaluating number of convolution filters {}/{}"
            .format(i+1, len(hyperparams)))
        params['n_conv'] = hp

        model = build_model(params, device)

        label = 'conv_filters='+str(hp)

        acc_list = train_model(model, device, n_epochs, train_loader, test_loader)
        acc_dict[label] = acc_list

    plot_model_performance(acc_dict)


def evaluate_convolution_kernels(train_loader, test_loader, device):
    hyperparams = [3, 5, 7]
    params = {'n_conv':[10,20],
            'kernel_size':5,
            'n_dense':50,
            'drop':0.5}
    n_epochs = 5

    acc_dict = defaultdict(list)

    for i, hp in enumerate(hyperparams):
        print("Evaluating size of convolution kernels {}/{}"
            .format(i+1, len(hyperparams)))
        params['kernel_size'] = hp

        model = build_model(params, device)

        label = 'conv_kernel='+str(hp)

        acc_list = train_model(model, device, n_epochs, train_loader, test_loader)
        acc_dict[label] = acc_list

    plot_model_performance(acc_dict)


def evaluate_dense_layer_size(train_loader, test_loader, device):
    hyperparams = [200, 100, 50]
    params = {'n_conv':[10,20],
            'kernel_size':5,
            'n_dense':50,
            'drop':0.5}
    n_epochs = 5

    acc_dict = defaultdict(list)

    for i, hp in enumerate(hyperparams):
        print("Evaluating number of hidden nodes {}/{}"
            .format(i+1, len(hyperparams)))
        params['n_dense'] = hp

        model = build_model(params, device)

        label = 'hidden_nodes='+str(hp)

        acc_list = train_model(model, device, n_epochs, train_loader, test_loader)
        acc_dict[label] = acc_list

    plot_model_performance(acc_dict)


def evaluate_dropout_rates(train_loader, test_loader, device):
    hyperparams = [0.25, 0.5, 0.75]
    params = {'n_conv':[10,20],
            'kernel_size':5,
            'n_dense':50,
            'drop':0.5}
    n_epochs = 5

    acc_dict = defaultdict(list)

    for i, hp in enumerate(hyperparams):
        print("Evaluating dropout rates {}/{}"
            .format(i+1, len(hyperparams)))
        params['drop'] = hp

        model = build_model(params, device)

        label = 'dropout='+str(hp)

        acc_list = train_model(model, device, n_epochs, train_loader, test_loader)
        acc_dict[label] = acc_list

    plot_model_performance(acc_dict)

def evaluate_train_epochs(train_loader, test_loader, device):
    hyperparams = [5, 10, 15]
    params = {'n_conv':[10,20],
            'kernel_size':5,
            'n_dense':50,
            'drop':0.5}
    n_epochs = 5

    acc_dict = defaultdict(list)

    for i, hp in enumerate(hyperparams):
        print("Evaluating no. of epochs {}/{}"
            .format(i+1, len(hyperparams)))
        n_epochs = hp

        model = build_model(params, device)

        label = 'epochs='+str(hp)

        acc_list = train_model(model, device, n_epochs, train_loader, test_loader)
        acc_dict[label] = acc_list

    plot_model_performance(acc_dict)

# main function
def main(argv):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random_seed = 42
    if device == 'cpu':
        torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    batch_size = 256
    train_loader, test_loader = get_dataloaders(batch_size)

    evaluate_convolution_filters(train_loader, test_loader, device)

    evaluate_convolution_kernels(train_loader, test_loader, device)

    evaluate_dense_layer_size(train_loader, test_loader, device)

    evaluate_dropout_rates(train_loader, test_loader, device)

    evaluate_train_epochs(train_loader, test_loader, device)

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)