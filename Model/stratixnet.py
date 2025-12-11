"""
STRATIX-Net Model - Custom Lightweight Architecture with MDA and ConvNeXt enhancements
Based on Inverted Residual Block (IRB) architecture with Multi-Dimensional Attention
"""

import torch
import torch.nn as nn
import logging
from .mda import MDA
from .convnext_block import EnhancedBlock
from .channel_spatial_attention import ChannelAttention, SpatialAttention

logger = logging.getLogger(__name__)


class InvertedResidualBlock(nn.Module):
    """
    Inverted Residual Block (IRB) - Core building block

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for depthwise convolution
        expand_ratio: Expansion factor for intermediate channels
    """

    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        self.use_residual = stride == 1 and in_channels == out_channels

        hidden_dim = int(in_channels * expand_ratio)

        layers = []

        # Expansion phase (pointwise)
        if expand_ratio != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                ]
            )

        # Depthwise convolution
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ]
        )

        # Projection phase (pointwise linear)
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LightweightBackbone(nn.Module):
    """
    Lightweight feature extraction backbone using Inverted Residual Blocks
    Architecture inspired by efficient mobile designs
    """

    def __init__(self):
        super().__init__()

        # Initial convolution layer
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        # IRB configuration: [expand_ratio, out_channels, num_blocks, stride]
        irb_config = [
            [1, 16, 1, 1],  # Stage 1
            [6, 24, 2, 2],  # Stage 2
            [6, 32, 3, 2],  # Stage 3
            [6, 64, 4, 2],  # Stage 4
            [6, 96, 3, 1],  # Stage 5
            [6, 160, 3, 2],  # Stage 6
            [6, 320, 1, 1],  # Stage 7
        ]

        # Build IRB stages
        self.stage1 = self._make_stage(32, *irb_config[0])
        self.stage2 = self._make_stage(16, *irb_config[1])
        self.stage3 = self._make_stage(24, *irb_config[2])
        self.stage4 = self._make_stage(32, *irb_config[3])
        self.stage5 = self._make_stage(64, *irb_config[4])
        self.stage6 = self._make_stage(96, *irb_config[5])
        self.stage7 = self._make_stage(160, *irb_config[6])

        # Final convolution layer (1x1 pointwise)
        self.final_conv = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        )

        self._initialize_weights()

    def _make_stage(self, in_channels, expand_ratio, out_channels, num_blocks, stride):
        """Create a stage with multiple IRB blocks"""
        layers = []

        # First block with specified stride
        layers.append(
            InvertedResidualBlock(in_channels, out_channels, stride, expand_ratio)
        )

        # Remaining blocks with stride 1
        for _ in range(1, num_blocks):
            layers.append(
                InvertedResidualBlock(out_channels, out_channels, 1, expand_ratio)
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights using best practices"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.final_conv(x)
        return x


class STRATIXNet(nn.Module):
    """
    STRATIX-Net: Strategic Tissue Recognition and Analysis using
    Inverted Residual with X-dimensional attention

    A lightweight deep learning architecture for gastrointestinal image classification
    combining Inverted Residual Blocks with Multi-Dimensional Attention (MDA)
    and ConvNeXt-inspired enhancements.

    Architecture:
        Stage 1: First 6 IRB blocks
        Stage 2: MDA + 4 IRB blocks
        Stage 3: MDA + 2 IRB blocks
        Stage 4: 1 IRB + MDA
        Stage 5: 4 IRB blocks
        Stage 6: Remaining IRB blocks + final conv + enhanced processing
        Classification Head: GAP → Linear → Dropout

    Args:
        num_classes: Number of output classes (default: 8 for Kvasir-v2)
    """

    def __init__(self, num_classes=8):
        super().__init__()

        # Custom lightweight backbone with IRB blocks
        backbone = LightweightBackbone()

        # Stage 1: First 6 IRB blocks (initial_conv + stage1 + stage2)
        # initial_conv(32) + stage1(16, 1 block) + stage2(24, 2 blocks) = 3 blocks total
        # Need to split to get 6 IRB blocks
        self.stage1 = nn.Sequential(
            backbone.initial_conv,  # Initial conv: 3→32
            backbone.stage1,  # 1 IRB: 32→16
            backbone.stage2,  # 2 IRB: 16→24
            backbone.stage3,  # 3 IRB: 24→32
        )

        # Stage 2: MDA + 4 IRB blocks
        self.stage2 = nn.Sequential(
            MDA(),  # Multi-dimensional attention
            backbone.stage4,  # 4 IRB: 32→64
        )

        # Stage 3: MDA + 2 IRB blocks (first 2 blocks of stage5)
        stage5_blocks = list(backbone.stage5.children())
        self.stage3 = nn.Sequential(
            MDA(),  # Multi-dimensional attention
            stage5_blocks[0],  # 1 IRB: 64→96
            stage5_blocks[1],  # 1 IRB: 96→96
        )

        # Stage 4: 1 IRB + MDA
        self.stage4 = nn.Sequential(
            stage5_blocks[2],  # 1 IRB: 96→96
            MDA(),  # Multi-dimensional attention
        )

        # Stage 5: 4 IRB blocks (first 4 blocks from stage6)
        stage6_blocks = list(backbone.stage6.children())
        self.stage5 = nn.Sequential(
            stage6_blocks[0],  # 1 IRB: 96→160
            stage6_blocks[1],  # 1 IRB: 160→160
            stage6_blocks[2],  # 1 IRB: 160→160
            backbone.stage7,  # 1 IRB: 160→320
        )

        # Stage 6: Rest of all in sequence (final conv + enhanced blocks)
        self.stage6 = nn.Sequential(
            backbone.final_conv,  # 1x1 conv: 320→1280
            EnhancedBlock(1280),
            ChannelAttention(1280),
            EnhancedBlock(1280),
            SpatialAttention(),
        )

        # Classification Head: GAP → Linear → Dropout → Linear
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, 512)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        logger.debug(f"Model input shape: {x.shape}")

        # Stage 1: First 6 IRB blocks
        x = self.stage1(x)
        logger.debug(f"After stage1 shape: {x.shape}")

        # Stage 2: MDA + 4 IRB blocks
        x = self.stage2(x)
        logger.debug(f"After stage2 shape: {x.shape}")

        # Stage 3: MDA + 2 IRB blocks
        x = self.stage3(x)
        logger.debug(f"After stage3 shape: {x.shape}")

        # Stage 4: 1 IRB + MDA
        x = self.stage4(x)
        logger.debug(f"After stage4 shape: {x.shape}")

        # Stage 5: 4 IRB blocks
        x = self.stage5(x)
        logger.debug(f"After stage5 shape: {x.shape}")

        # Stage 6: Rest of all in sequence
        x = self.stage6(x)
        logger.debug(f"After stage6 shape: {x.shape}")

        # Classification Head: GAP → Linear → Dropout
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        logger.debug(f"Model output shape: {x.shape}")
        return x

    def get_num_parameters(self):
        """Calculate total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_mb(self):
        """Calculate model size in megabytes"""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024**2)
