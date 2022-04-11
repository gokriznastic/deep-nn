import torch.nn as nn
import torch.nn.functional as F

# the base network architecture as per required task
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.5)
        # fully connected layer
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.dropout(self.conv2(x))))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

# the generic network architecture for experimenting with different hyperparameters
class ExperimentNetwork(nn.Module):
    def __init__(self, num_conv_filters, conv_size, num_hidden_nodes, dropout_rate):
        super(ExperimentNetwork, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, num_conv_filters[0], kernel_size=conv_size)
        self.conv2 = nn.Conv2d(num_conv_filters[0], num_conv_filters[1], kernel_size=conv_size)
        # max pooling layer
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.intermediate = ((((28 - conv_size + 1)//2) - conv_size + 1)//2)**2 * num_conv_filters[1]
        # fully connected layer
        self.fc1 = nn.Linear(self.intermediate, num_hidden_nodes)
        self.fc2 = nn.Linear(num_hidden_nodes, 10)

    def forward(self, x):
        x = F.relu(self.max_pool(self.conv1(x)))
        x = F.relu(self.max_pool(self.dropout(self.conv2(x))))
        x = x.view(-1, self.intermediate)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x)

# the network architecture used for CIFAR10 dataset
class CifarNet(nn.Module):
    def __init__(self, batch_size):
        super(CifarNet, self).__init__()
        self.batch_size = batch_size
        # convolutional layer
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layer
        self.fc = nn.Linear(8*8*32, 256)

        self.output = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2048)
        x = F.relu(self.fc(x))
        x = self.output(x)

        return x
