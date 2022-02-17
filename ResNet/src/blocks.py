# %%
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision

# %%
dataset = torchvision.datasets.CIFAR10(
    './data', 
    train=True, 
    transform=torchvision.transforms.ToTensor(), 
    download=True
    )

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=5, shuffle=False
    )

# %%
data, label = next(iter(dataloader))

if __name__ == '__main__':
    print(data.shape, label)

# %%
class ShortcutProjection(nn.Module):
    """Some Information about ShortcutProjection"""
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ShortcutProjection, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x: torch.Tensor):

        return self.bn(self.conv(x))

# %%
class ResidualBlock(nn.Module):
    """Some Information about ResidualBlock"""
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        return self.act2(x + shortcut)

# %%
class BottleneckResidualBlock(nn.Module):
    """Some Information about BottleneckResidualBlock"""
    def __init__(self, in_channels: int, bottleneck_channels: int, out_channels: int, stride: int):
        super(BottleneckResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ShortcutProjection(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()
        self.act3 = nn.ReLU()
        
    def forward(self, x: torch.Tensor):
        shortcut = self.shortcut(x)
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return self.act3(x + shortcut)

# %%



