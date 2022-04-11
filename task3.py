"""
@author: Gopal Krishna

Task 3 - Create a digit embedding space
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

# method to fetch greek symbols dataloaders
def get_greek_dataloader(data_path):
    trans = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.RandomInvert(p=1),
                transforms.ToTensor(),
            ])

    greek_data = torchvision.datasets.ImageFolder(data_path, transform=trans)

    greek_dl = torch.utils.data.DataLoader(greek_data, batch_size=1, shuffle=True)

    return greek_dl

# method to build truncated model for generating embeddings
def build_truncated_model(save_path):
    class EmbedSubmodel(Network):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward( self, x ):
            x = F.relu(self.max_pool(self.conv1(x)))
            x = F.relu(self.max_pool(self.dropout(self.conv2(x))))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))

            return x

    sub_model = EmbedSubmodel()
    sub_model.load_state_dict(torch.load(save_path)['model'])
    sub_model.eval()

    return sub_model

# method to generate embeddings from given data
def get_embeddings(model, greek_dataloader):
    feature_set = []
    label_set = []
    for data, target in greek_dataloader:
        data = transforms.functional.invert(data)
        output = model(data)
        feature_set.append(output.cpu().detach().numpy())
        label_set.append(target.cpu().numpy())

    features = np.array(feature_set).reshape(len(feature_set), -1)
    labels = np.array(label_set)

    return features, labels

# method to calculate SSD
def ssd(A,B):
    diff = A.ravel() - B.ravel()
    return np.dot(diff, diff)

# method to compute distances
def compute_distances(features, labels):
    examples = {}

    for i, label in enumerate(labels):
        if (label[0] not in examples):
            examples[label[0]] = (i, features[i])

    ssdist = {}
    for key in examples.keys():
        distances = []
        for i, feature in enumerate(features):
            distances.append((i, ssd(examples[key][1], feature), labels[i][0]))
        distances.sort(key = lambda x: x[1])
        ssdist[key] = distances

    return ssdist

# method to print SSD distances
def display_ssd_values(ssdist):
    abg = {0:'alpha', 1:'beta', 2:'gamma'}

    for key in ssdist.keys():
        print("sorted distances from {} [{}]".format(abg[key], key))
        for tup in ssdist[key]:
            print(tup[1], tup[2])

# main function
def main(argv):
    # handling command line arguments in argv
    save_path = argv[1]
    data_path = argv[2]
    custom_data_path = argv[3]

    # Task 3A - create a greek symbol data set
    greek_dataloader = get_greek_dataloader(data_path)

    # Task 3B - create a truncated model
    model = build_truncated_model(save_path)

    # Task 3C - project the greek symbols into the embedding space
    features, labels = get_embeddings(model, greek_dataloader)

    # Task 3D - compute distances in the embedding space
    ssdist = compute_distances(features, labels)
    display_ssd_values(ssdist)

    print()
    # Task 3E - create your own greek symbol data
    custom_greek_dataloader = get_greek_dataloader(custom_data_path)
    data, target = next(iter(custom_greek_dataloader))

    print(data.size())
    data = data[0].numpy().transpose(1,2,0)
    fig = plt.figure()
    plt.imshow(data, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    feat, lab = get_embeddings(model, custom_greek_dataloader)
    ssdist = compute_distances(feat, lab)
    display_ssd_values(ssdist)


    # main function code
    return

if __name__ == "__main__":
    main(sys.argv)