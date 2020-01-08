"""SegmentationNN"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation.fcn import FCNHead


class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23):
        super(SegmentationNN, self).__init__()

        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
#         self.backbone = models.vgg16_bn(pretrained=True)
#         self.backbone.classifier = nn.Sequential(
#             nn.Conv2d(512, 4096, kernel_size=1),
#             nn.BatchNorm2d(4096),
#             nn.ReLU(True),
#             nn.Conv2d(4096, 4096, kernel_size=1),
#             nn.BatchNorm2d(4096),
#             nn.ReLU(True),
#             nn.Conv2d(4096, num_classes, kernel_size=1)
#         )
#         self.avgpool = nn.AvgPool2d(kernel_size=2)
#         self.upsample1 = nn.Sequential(
#             nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2,
#                                             stride=2, padding=0, output_padding=1),
#             nn.BatchNorm2d(num_classes),
#             nn.ReLU()
#             )
#         self.upsample2 = nn.Sequential(
#             nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2,
#                                stride=2, padding=0),
#             nn.BatchNorm2d(num_classes),
#             nn.ReLU()
#         )
#         self.upsample3 = nn.Sequential(
#             nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4,
#                                stride=8, padding=1, dilation=3),
# #             nn.BatchNorm2d(num_classes),
# #             nn.ReLU(True)
#         )

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(256, num_classes, 1),
#             nn.BatchNorm2d(num_classes),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(512, num_classes, 1),
#             nn.BatchNorm2d(num_classes),
#             nn.ReLU()
#         )
        self.fcn = models.segmentation.fcn_resnet101(pretrained=True)
        self.fcn.classifier = FCNHead(2048, num_classes)
        self.fcn.aux_classifier = None
        
        for name, param in self.fcn.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
#         low_res_feat_1 = None
#         low_res_feat_2 = None
#         for i, layer in enumerate(self.backbone.features):
#             x = layer(x)
#             if i == 23:
#                 # output of pool3, shape (batch_size, 28, 28, 256)
#                 low_res_feat_1 = x.clone()
#             if i == 33:
#                 # output of pool4, shape (batch_size, 14, 14, 512)
#                 low_res_feat_2 = x.clone()
#         x = self.backbone.avgpool(x)
#         x = self.backbone.classifier(x)

#         x = self.upsample1(x)
#         x += self.conv2(low_res_feat_2)

#         x = self.upsample2(x)
#         x += self.conv1(low_res_feat_1)

#         x = self.upsample3(x)
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
        x = self.fcn(x)['out']
        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)


# if __name__ == '__main__':
#     model = SegmentationNN()
#     x = torch.randn((5, 3, 240 ,240))
#     print(model(x).shape)