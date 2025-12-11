"""Training utilities for STRATIX-Net"""

from .train_val import train_and_evaluate
from .optimizer_scheduler import get_optimizer_scheduler

__all__ = ["train_and_evaluate", "get_optimizer_scheduler"]
