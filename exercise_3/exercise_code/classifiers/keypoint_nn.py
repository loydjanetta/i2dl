import torch
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        #######################################################################
        # TODO: Define all the layers of this CNN, the only requirements are: #
        # 1. This network takes in a square (same width and height),          #
        #    grayscale image as input.                                        #
        # 2. It ends with a linear layer that represents the keypoints.       #
        # It's suggested that you make this last layer output 30 values, 2    #
        # for each of the 15 keypoint (x, y) pairs                            #
        #                                                                     #
        # Note that among the layers to add, consider including:              #
        # maxpooling layers, multiple conv layers, fully-connected layers,    #
        # and other layers (such as dropout or  batch normalization) to avoid #
        # overfitting.                                                        #
        #######################################################################
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2))

        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU(True),
                                   nn.MaxPool2d(2))

        self.fc_layers = nn.Sequential(nn.Linear(256 * 6 * 6, 1000),
                                       nn.Linear(1000, 100),
                                       nn.Linear(100, 30))

        #######################################################################
        #                             END OF YOUR CODE                        #
        #######################################################################

    def forward(self, x):
        #######################################################################
        # TODO: Define the feedforward behavior of this model                 #
        # x is the input image and, as an example, here you may choose to     #
        # include a pool/conv step:                                           #
        # x = self.pool(F.relu(self.conv1(x)))                                #
        # a modified x, having gone through all the layers of your model,     #
        # should be returned                                                  #
        #######################################################################
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.fc_layers(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
