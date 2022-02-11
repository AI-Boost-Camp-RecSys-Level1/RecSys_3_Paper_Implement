# %%
from typing import List, Optional
from blocks import BottleneckResidualBlock, ResidualBlock

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
data.shape, label

# %%
class ResNetBase(nn.Module):
    """Some Information about ResNetBase"""
    def __init__(self, n_blocks: List[int], n_channels: List[int],
                 bottlenecks: Optional[List[int]] = None,
                 img_channels: int = 3, first_kernel_size: int = 7):
        super(ResNetBase, self).__init__()
        assert len(n_blocks) == len(n_channels)
        assert bottlenecks is None or len(bottlenecks) == len(n_channels)
        self.conv = nn.Conv2d(img_channels, n_channels[0],
                              kernel_size=first_kernel_size, stride=2, padding=first_kernel_size // 2)
        self.bn = nn.BatchNorm2d(n_channels[0])
        blocks = []
        prev_channels = n_channels[0]
        for i, channels in enumerate(n_channels):
            stride = 2 if len(blocks) == 0 else 1
            if bottlenecks is None:
                blocks.append(ResidualBlock(prev_channels, channels, stride=stride))
            else:
                blocks.append(BottleneckResidualBlock(prev_channels, bottlenecks[i], channels,
                                                      stride=stride))
            prev_channels = channels
            for _ in range(n_blocks[i] - 1):
                if bottlenecks is None:
                    blocks.append(ResidualBlock(channels, channels, stride=1))
                else:
                    blocks.append(BottleneckResidualBlock(channels, bottlenecks[i], channels, stride=1))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor):
        x = self.bn(self.conv(x))
        x = self.blocks(x)
        x = x.view(x.shape[0], x.shape[1], -1)
        
        return x.mean(dim=-1)

# %%
n_blocks = [2,2,2,2]
n_channels = [64,128,256,512]
bottlenecks = None
img_channels = 3
first_kernel_size = 7

k = ResNetBase(n_blocks, n_channels, bottlenecks, img_channels, first_kernel_size)

# %%
print(k(data))


