"""
@author: Gopal Krishna

Extension - training a CNN on CIFAR10 dataset
"""

import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
from network import CifarNet
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

def get_dataloaders(batch_size):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes

def plot_example_images(train_loader, classes):
    example_data, example_targets = next(iter(train_loader))

    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i].numpy().transpose(1,2,0), interpolation='none')
        plt.title(classes[example_targets[i]])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def get_model(device, batch_size):
    model = CifarNet(batch_size)
    model.to(device)
    print(summary(model, (3, 32, 32), device=device))

    return model

def train_model(model, device, learning_rate, n_epochs, batch_size, train_loader, test_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        test_loss = 0.0

        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(train_loader.dataset)
        test_loss = test_loss/len(test_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, test_loss))

        if test_loss <= test_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            test_loss_min,
            test_loss))
            torch.save(model.state_dict(), './model_cifar.pth')
            test_loss_min = test_loss

    return model, optimizer

def visualize_test_samples(test_loader, model, device):
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    images.numpy()

    images = images.to(device)

    output = model(images)
    _, preds_tensor = torch.max(output, 1)
    preds = preds_tensor.cpu().numpy()

    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
        plt.imshow(images.detach().cpu().numpy().transpose(1,2,0)[idx])
        ax.set_title("{} ({})".format(classes[preds[idx]], classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))

# main function
def main(argv):
    # handling command line arguments in argv
    batch_size = int(argv[1])
    device = argv[2]
    learning_rate = float(argv[3])
    n_epochs = int(argv[4])

    random_seed = 42
    if device == 'cpu':
        torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    train_loader, test_loader, classes = get_dataloaders(batch_size)

    plot_example_images(train_loader, classes)

    model = get_model(device, batch_size)

    model, optimizer = train_model(model, device, learning_rate, n_epochs, batch_size, train_loader, test_loader)

    visualize_test_samples(test_loader, model, device)

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)