"""
Channel and Spatial Attention mechanisms for GastroVisionNet
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


# Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logger.debug(f"ChannelAttention input shape: {x.shape}")
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_out)))
        max_out = self.fc2(self.relu(self.fc1(max_out)))
        out = self.sigmoid(avg_out + max_out)
        x = x * out
        logger.debug(f"ChannelAttention output shape: {x.shape}")
        return x


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logger.debug(f"SpatialAttention input shape: {x.shape}")
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        x = x * out
        logger.debug(f"SpatialAttention output shape: {x.shape}")
        return x
