"""
MobileSTRATIX-Net Model - Lightweight MobileNetV2 with MSCA attention
Uses pretrained MobileNetV2 from torchvision
"""

import torch.nn as nn
import logging
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MSCA(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(MSCA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = self.fc(self.global_avg_pool(x))
        max_out = self.fc(self.global_max_pool(x))
        attention = avg_out + max_out
        return x * attention, attention


class MobileSTRATIXNet(nn.Module):
    def __init__(self, num_classes=8):
        super(MobileSTRATIXNet, self).__init__()

        # Load pretrained MobileNetV2 and extract features
        self.mobilenet_v2 = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        self.mobilenet_v2.features = nn.Sequential(*self.mobilenet_v2.features)

        self.stage1_conv = ConvBlock(1280, 256, stride=1)
        self.stage1_attention = MSCA(256, reduction=4)

        self.stage2_conv = ConvBlock(256, 512, stride=2)
        self.stage2_attention = MSCA(512, reduction=16)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # Extract features from MobileNetV2
        x = self.mobilenet_v2.features(x)

        # Stage 1: Conv + Attention
        x = self.stage1_conv(x)
        x, attention1 = self.stage1_attention(x)

        # Stage 2: Conv + Attention
        x = self.stage2_conv(x)
        x, attention2 = self.stage2_attention(x)

        # Global pooling and classification
        x = self.global_pool(x)
        x = self.fc(x)

        return x
