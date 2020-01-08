"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial


class ClassificationCNN(nn.Module):
    """
    A PyTorch implementation of a three-layer convolutional network
    with the following architecture:

    conv - relu - 2x2 max pool - fc - dropout - relu - fc

    The network operates on mini-batches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, kernel_size=7,
                 stride_conv=1, weight_scale=0.001, pool=2, stride_pool=2,
                 hidden_dim=100, num_classes=10, dropout=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_filters: Number of filters to use in the convolutional layer.
        - kernel_size: Size of filters to use in the convolutional layer.
        - hidden_dim: Number of units to use in the fully-connected hidden
                      layer
        - num_classes: Number of scores to produce from the final affine layer.
        - stride_conv: Stride for the convolution layer.
        - stride_pool: Stride for the max pooling layer.
        - weight_scale: Scale for the convolution weights initialization
        - pool: The size of the max pooling window.
        - dropout: Probability of an element to be zeroed.
        """
        super(ClassificationCNN, self).__init__()
        channels, height, width = input_dim

        #######################################################################
        # TODO: Initialize the necessary trainable layers to resemble the     #
        # ClassificationCNN architecture  from the class docstring.           #
        #                                                                     #
        # In- and output features should not be hard coded which demands some #
        # calculations especially for the input of the first fully            #
        # convolutional layer.                                                #
        #                                                                     #
        # The convolution should use "same" padding which can be derived from #
        # the kernel size and its weights should be scaled. Layers should     #
        # have a bias if possible.                                            #
        #                                                                     #
        # Note: Avoid using any of PyTorch's random functions or your output  #
        # will not coincide with the Jupyter notebook cell.                   #
        #######################################################################
        SameConv2d = get_same_padding_conv2d([height, width])
        conv = SameConv2d(channels, num_filters,
                          kernel_size=kernel_size,
                          stride=stride_conv,
                          )
        nn.init.xavier_normal_(conv.weight.data, gain=weight_scale)

        self.layer1 = nn.Sequential(conv,
                                    nn.ReLU(True),
                                    nn.MaxPool2d(pool, stride_pool))
        compute_spatial_size = lambda x: math.floor((x - pool) / stride_pool + 1)
        h, w = compute_spatial_size(height), compute_spatial_size(width)

        flat_size = num_filters * h * w
        self.layer2 = nn.Sequential(nn.Linear(flat_size, hidden_dim),
                                    nn.Dropout(p=dropout),
                                    nn.ReLU(True),
                                    nn.Linear(hidden_dim, num_classes))

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the neural network. Should not be called manually but
        by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """

        #######################################################################
        # TODO: Chain our previously initialized fully-connected neural       #
        # network layers to resemble the architecture drafted in the class    #
        # docstring. Have a look at the Variable.view function to make the    #
        # transition from the spatial input image to the flat fully connected #
        # layers.                                                             #
        #######################################################################
        x = self.layer1(x)
        x = x.view(x.size(0), -1).contiguous()
        x = self.layer2(x)
        #######################################################################
        #                          END OF YOUR CODE                           #
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


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x