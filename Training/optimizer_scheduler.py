"""
Optimizer and scheduler configurations for GastroVisionNet
"""

import torch.optim as optim
from torch.optim import lr_scheduler


def get_optimizer_scheduler(
    model, learning_rate=1e-4, weight_decay=1e-5, num_epochs=100, optimizer_type="AdamW"
):
    """
    Get optimizer and scheduler for training

    Args:
        model: The model to optimize
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        num_epochs: Number of epochs for scheduler
        optimizer_type: Type of optimizer ('AdamW', 'Adam', 'SGD')

    Returns:
        Tuple of (optimizer, scheduler)
    """
    # Get optimizer
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    # Get scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    return optimizer, scheduler
