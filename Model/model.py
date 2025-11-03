"""
Model factory and parameter counting utilities
"""

import torch
import logging
from .gastrovisionnet import GastroVisionNet
from .mobilegastrovisionnet import MobileGastroVisionNet

logger = logging.getLogger(__name__)


def count_parameters(model):
    """Function to count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model(model_name="gastrovisionnet", num_classes=8, device=None):
    """
    Define model initialization

    Args:
        model_name: Name of the model ('gastrovisionnet' or 'mobilegastrovisionnet')
        num_classes: Number of classification classes
        device: Device to place model on (if None, uses CUDA if available)

    Returns:
        Initialized model on specified device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name.lower() == "gastrovisionnet":
        logger.info(f"Initializing GastroVisionNet with {num_classes} classes")
        model = GastroVisionNet(num_classes=num_classes)
    elif model_name.lower() == "mobilegastrovisionnet":
        logger.info(f"Initializing MobileGastroVisionNet with {num_classes} classes")
        model = MobileGastroVisionNet(num_classes=num_classes)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. Choose 'gastrovisionnet' or 'mobilegastrovisionnet'"
        )

    return model.to(device)
