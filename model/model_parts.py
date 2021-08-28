import os
import torch
from torch import nn
from torch.nn.modules import transformer
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch import Tensor
import torchvision.transforms as transforms


class Convolution(nn.Module):
    # in_channel, out_channel代表有幾個filter。
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convolution = nn.Sequential(
            # kernel_size代表filter的size(根據論文取得)。
            # TODO: Check padding = 1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # TODO: Check if we should add BatchNorm2d.
            nn.BatchNorm2d(out_channels),
            # TODO: Check what inplace argument meaning.
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: Tensor):
        return self.convolution(x)

class DoubleConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.doubleConvolution = nn.Sequential(
            Convolution(in_channels, out_channels),
            Convolution(out_channels, out_channels),
        )

    def forward(self, x: Tensor):
        return self.doubleConvolution(x)

class MaxPoolingWithDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxPoolingWithDoubleConv = nn.Sequential(
            # kernel_size代表做幾乘幾的max pooling(e.g. 這邊做2x2 max pooling)，根據論文取得。
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolution(in_channels, out_channels)
        )

    def forward(self, x: Tensor):
        return self.maxPoolingWithDoubleConv(x)

class UpConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # TODO: Check if use ConvTranspose2d.
        # TODO: Check arguments mode
        # self.upConvolution = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upConvolution = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor):
        return self.upConvolution(x)

# TODO: Check if the method work(Tseng's method).
class CopyAndCrop(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x1: Tensor, x2: Tensor):
        _, _, height, width = x2.shape
        crop = transforms.CenterCrop((height, width))
        return torch.cat([x2, crop(x1)], dim = 1)


# up-convolution, copy-and-crop, and double convolution.
class UpConvWithCropAndDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upConvolution = UpConvolution(in_channels, out_channels)
        self.copyAndCrop = CopyAndCrop()
        self.doubleConvolution = DoubleConvolution(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor):
        x2 = self.upConvolution(x2)
        x3 = self.copyAndCrop(x1, x2)
        return self.doubleConvolution(x3)
