"""
@author: Gopal Krishna

Task 1 - Build and train a network to recognize digits
Parts A, B, C, D, E
"""

import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from torchsummary import summary
from network import Network

warnings.filterwarnings("ignore")

# method to get train and test dataloaders
def get_dataloaders(batch_size_train, batch_size_test):
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

#method to plot first few digits in MNIST training set
def plot_example_digits(train_loader):
    example_data, example_targets = next(iter(train_loader))

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# method to build and load model in cpu or gpu
def get_model(device):
    model = Network()
    model.to(device)
    print(summary(model, (1, 28, 28), device=device))

    return model

# training loop for the network
def train(epoch, model, device, optimizer, batch_size_train, train_loader, train_losses, train_counter):
    log_interval = 1

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)

        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train - Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item()))

            train_losses.append(loss.item())
            train_counter.append((batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))

# testing loop for the network
def test(model, device, train_loader, test_loader, test_losses):
    model.eval()

    train_loss = 0
    test_loss = 0
    train_correct = 0
    test_correct = 0

    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            train_correct += pred.eq(target.data.view_as(pred)).sum()

        train_loss /= len(train_loader.dataset)
        print('\nTrain set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    train_loss,
                    train_correct,
                    len(train_loader.dataset),
                    100. * train_correct / len(train_loader.dataset)))

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            test_correct += pred.eq(target.data.view_as(pred)).sum()

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                    test_loss,
                    test_correct,
                    len(test_loader.dataset),
                    100. * test_correct / len(test_loader.dataset)))

# method to plot curves
def plot_curves(train_counter, train_losses, test_counter, test_losses):
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

# method for complete training procedure of the model alternating between training and testing loops
def train_model(model, device, learning_rate, n_epochs, batch_size_train, train_loader, test_loader):
    momentum = 0.5
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)
    log_interval = 1

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    test(model, device, train_loader, test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train(epoch, model, device, optimizer, batch_size_train, train_loader, train_losses, train_counter)
        test(model, device, train_loader, test_loader, test_losses)

    plot_curves(train_counter, train_losses, test_counter, test_losses)

    return model, optimizer

# method to save model and optimizer states
def save_model(model, optimizer):
    model_ckpt = {}
    model_ckpt['model'] = model.state_dict()
    model_ckpt['optimizer'] = optimizer.state_dict()

    torch.save(model_ckpt, './checkpoint.pth')

# main function
def main(argv):
    # handling command line arguments in argv
    batch_size_train = int(argv[1])
    batch_size_test = int(argv[2])
    device = argv[3]
    learning_rate = float(argv[4])
    n_epochs = int(argv[5])

    # Task 1A - get the MNIST digit data set
    train_loader, test_loader = get_dataloaders(batch_size_train, batch_size_test)

    plot_example_digits(train_loader)

    # Task 1B - make your network code repeatable
    random_seed = 42
    if device == 'cpu':
        torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Task 1C - build a network model
    model = get_model(device)

    # Task 1D - train the model
    model, optimizer = train_model(model, device, learning_rate, n_epochs, batch_size_train, train_loader, test_loader)

    # Task 1E - save the network to a file
    save_model(model, optimizer)

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)