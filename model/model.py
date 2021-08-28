import os
import torch
from torch import nn
from torch.nn.modules import transformer
from torch.nn.modules.batchnorm import BatchNorm2d
from torch import Tensor
import torchvision.transforms as transforms
from .model_parts import *

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.firstDoubleConv = Convolution(1, 64)
        self.firstMaxPoolingWithDoubleConv = MaxPoolingWithDoubleConv(64, 128)
        self.secondMaxPoolingWithDoubleConv = MaxPoolingWithDoubleConv(128, 256)
        self.thirdMaxPoolingWithDoubleConv = MaxPoolingWithDoubleConv(256, 512)
        self.fourthMaxPoolingWithDoubleConv = MaxPoolingWithDoubleConv(512, 1024)
        
        self.firstUpConvWithCropAndDoubleConv = UpConvWithCropAndDoubleConv(1024, 512)
        self.secondUpConvWithCropAndDoubleConv = UpConvWithCropAndDoubleConv(512, 256)
        self.thirdUpConvWithCropAndDoubleConv = UpConvWithCropAndDoubleConv(256, 128)
        self.fourthUpConvWithCropAndDoubleConv = UpConvWithCropAndDoubleConv(128, 64)

        self.lastConvolution = Convolution(64, 1)
        

    def forward(self, x):
        x1 = self.firstDoubleConv(x)
        x2 = self.firstMaxPoolingWithDoubleConv(x1)
        x3 = self.secondMaxPoolingWithDoubleConv(x2)
        x4 = self.thirdMaxPoolingWithDoubleConv(x3)
        
        y1 = self.fourthMaxPoolingWithDoubleConv(x4)
        y2 = self.firstUpConvWithCropAndDoubleConv(x4, y1)
        y3 = self.secondUpConvWithCropAndDoubleConv(x3, y2)
        y4 = self.thirdUpConvWithCropAndDoubleConv(x2, y3)
        y5 = self.fourthUpConvWithCropAndDoubleConv(x1, y4)

        return self.lastConvolution(y5)