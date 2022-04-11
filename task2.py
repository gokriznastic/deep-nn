"""
@author: Gopal Krishna

Task 2 - Examine your network
"""

import sys
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

from network import Network

warnings.filterwarnings("ignore")

# method to initialize and load saved model into cpu or gpu
def load_model(save_path):
    model = Network()
    model.load_state_dict(torch.load(save_path)['model'])

    return model

# method to visualize the filters of first layer of the CNN
def analyze_first_layer(model):
    conv1_wt = model.conv1.weight.detach().numpy()

    print("Filter weights:")
    print(conv1_wt)
    print("Filter weights shape: {}".format(conv1_wt.shape))

    fig = plt.figure()
    for i in range(10):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(conv1_wt[i, 0], interpolation='none')
        plt.title("Filter: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return conv1_wt

# method to show the visualization of the output of the convolution operation on an image
def show_effect_of_filters(model):
    conv1_wt = model.conv1.weight.detach().numpy()

    loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True, 
                            transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                            batch_size=1, shuffle=True)

    data, target = next(iter(loader))
    data_np = data.cpu().numpy()[0]
    data_np = data_np.transpose(1, 2, 0)

    effects = []
    with torch.no_grad():
        for wt in conv1_wt:
            effects.append(cv2.filter2D(data_np, -1, wt[0]))

    k = 1
    fig = plt.figure(figsize=(4,5))
    for i in range(10):
        plt.subplot(5,4,k)
        plt.imshow(conv1_wt[i, 0], cmap='gray', interpolation='none')
        k += 1
        plt.xticks([])
        plt.yticks([])

        plt.subplot(5,4,k)
        plt.imshow(effects[i], cmap='gray', interpolation='none')
        k += 1
        plt.xticks([])
        plt.yticks([])
    plt.show()

    return data

# method to build a truncated model from the base network architecture having only convolutional layers
def build_truncated_model(save_path, data, depth=2):
    class Submodel(Network):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        # override the forward method
        def forward( self, x ):
            x = F.relu(self.max_pool(self.conv1(x)))
            if depth == 2:
                x = F.relu(self.max_pool(self.dropout(self.conv2(x))))

            return x

    sub_model = Submodel()
    sub_model.load_state_dict(torch.load(save_path)['model'])
    sub_model.eval()

    output = sub_model(data)
    output = output.detach().numpy()

    for i in range(output.shape[1]):
        plt.subplot(4,5,i+1)
        plt.tight_layout()
        plt.imshow(output[0][i], cmap='gray', interpolation='none')
        plt.title("Channel: {}".format(i))
        plt.xticks([])
        plt.yticks([])
    plt.show()

# main function
def main(argv):
    # handling command line arguments in argv
    save_path = argv[1]

    # Task 2A - analyze the first layer
    model = load_model(save_path)

    analyze_first_layer(model)

    print()

    # Task 2B - show the effect of the filters
    data = show_effect_of_filters(model)

    # Task 2C - build a truncated model
    sub_model = build_truncated_model(save_path, data, 1)

    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)

