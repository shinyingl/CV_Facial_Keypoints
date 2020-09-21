## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        # (224-5)/1 + 1 = 219 --> output (32, 219, 219)
        # after pool dimension (32, 109, 109)

        self.conv2 = nn.Conv2d(32, 64, 4)
        self.conv2_bn = nn.BatchNorm2d(64)
        # (109-4)/1+1 = 106 --> ouput (64, 106, 106)
        # self.pool = nn.MaxPool2d(2, 2)
        # after pool dimension (64, 53, 53)

        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv3_bn = nn.BatchNorm2d(128)
        # (53-3)/1+1 = 51 --> ouput (128, 51, 51)
        # self.pool = nn.MaxPool2d(2, 2)
        # after pool dimension (128, 25, 25)

        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv4_bn = nn.BatchNorm2d(256)
        # (25-2)/1+1 = 24 --> ouput (256, 24, 24)
        # self.pool = nn.MaxPool2d(2, 2)
        # after pool dimension (256, 12, 12)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256*12*12, 5000)
        self.fc1_bn = nn.BatchNorm1d(5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, 136)

        self.drop = nn.Dropout(p=0.5)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        # x = self.drop(x)

        x = x.view(x.size(0), -1)

        x = self.drop(F.relu(self.fc1_bn(self.fc1(x))))
        x = self.drop(F.relu(self.fc2_bn(self.fc2(x))))
        x = self.fc3(x)
        
    
        return x
