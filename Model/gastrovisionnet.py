"""
GastroVisionNet Model - Based on MobileNetV2 with TriadAttention and ConvNeXt enhancements
"""

import torch
import torch.nn as nn
import logging
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from .triad_attention import TriadAttention
from .convnext_block import EnhancedBlock
from .channel_spatial_attention import ChannelAttention, SpatialAttention

logger = logging.getLogger(__name__)


class GastroVisionNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        features_list = list(base_model.features.children())
        self.features = nn.Sequential(
            *features_list[0:7],
            TriadAttention(),
            *features_list[7:11],
            TriadAttention(),
            *features_list[11:14],
            TriadAttention(),
            *features_list[14:],
        )
        self.enhanced_blocks = nn.Sequential(
            EnhancedBlock(1280),
            ChannelAttention(1280),
            EnhancedBlock(1280),
            SpatialAttention(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, num_classes))

    def forward(self, x):
        logger.debug(f"Model input shape: {x.shape}")
        x = self.features(x)
        logger.debug(f"After features shape: {x.shape}")
        x = self.enhanced_blocks[0](x)
        logger.debug(f"After enhanced_blocks[0] shape: {x.shape}")
        x = self.enhanced_blocks[1](x)
        logger.debug(f"After enhanced_blocks[1] (ChannelAttention) shape: {x.shape}")
        x = self.enhanced_blocks[2](x)
        logger.debug(f"After enhanced_blocks[2] shape: {x.shape}")
        x = self.enhanced_blocks[3](x)
        logger.debug(f"After enhanced_blocks[3] (SpatialAttention) shape: {x.shape}")
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        logger.debug(f"Model output shape: {x.shape}")
        return x
