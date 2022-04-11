"""
@author: Gopal Krishna

Task 1 - Build and train a network to recognize digits
Parts F, G
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
from torchvision.datasets import ImageFolder
from torchvision import transforms

from network import Network

warnings.filterwarnings("ignore")

# method to initialize and load saved model into cpu or gpu
def load_model(save_path, device):
    test_model = Network()
    test_model.load_state_dict(torch.load(save_path)['model'])
    test_model.to(device)
    test_model.eval()

    return test_model

# method to print and display results on MNIST test set
def infer_on_test_set(batch_size, test_model, device):
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True, 
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                            batch_size=batch_size, shuffle=True)

    data, target = next(iter(test_loader))

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = test_model(data)
        pred = output.data.max(1, keepdim=True)[1]

    data, target = data.cpu(), target.cpu()

    print("Predicted logits:")
    print(np.round(output.cpu().numpy(), decimals=2))

    prob_pred = torch.exp(output).cpu().numpy()
    print("Predicted probablities:")
    print(np.round(prob_pred, decimals=2))

    print("Model output:\t{}".format(np.argmax(prob_pred, axis=1)))

    labels = target.cpu().numpy()
    print("Groundtruth labels:\t{}".format(labels))

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# method to infer on new custom inputs
def infer_on_new_inputs(batch_size, test_model, device):
    trans = transforms.Compose([
            # transforms.CenterCrop(size=(1500, 1500)),
            transforms.RandomInvert(p=1),
            transforms.Resize(28),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])

    digits_data = ImageFolder('./data/custom/digits/',
                        transform=trans)

    digits_loader = torch.utils.data.DataLoader(digits_data, batch_size, shuffle=True)

    data, target = next(iter(digits_loader))

    # data = data[0].numpy().transpose(1,2,0)
    # fig = plt.figure()
    # plt.imshow(data, cmap='gray')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    with torch.no_grad():
        data, target = data.to(device), target.to(device)
        output = test_model(data)
        pred = output.data.max(1, keepdim=True)[1]

    data, target = data.cpu(), target.cpu()

    print("Predicted logits:")
    print(np.round(output.cpu().numpy(), decimals=2))

    prob_pred = torch.exp(output).cpu().numpy()
    print("Predicted probablities:")
    print(np.round(prob_pred, decimals=2))

    print("Model output:\t{}".format(np.argmax(prob_pred, axis=1)))

    labels = target.cpu().numpy()
    print("Groundtruth labels:\t{}".format(labels))

    fig = plt.figure()
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# main function
def main(argv):
    # handling command line arguments in argv
    save_path = argv[1]
    batch_size = int(argv[2])
    device = argv[3]

    # Task 1F - read the network and run it on the test set
    test_model = load_model(save_path, device)

    infer_on_test_set(batch_size, test_model, device)

    print()

    # Task 1G - test the network on new inputs
    infer_on_new_inputs(batch_size, test_model, device)

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)

