"""
ConvNeXt Block components for GastroVisionNet
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LayerNorm2d(nn.Module):
    def __init__(self, normalized_shape):
        super(LayerNorm2d, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, height, width, channels = x.shape
        logger.debug(f"LayerNorm2d input shape: {x.shape}")
        x = x.contiguous()
        x = x.reshape(-1, channels)
        x = self.ln(x)
        x = x.reshape(batch, height, width, channels)
        logger.debug(f"LayerNorm2d output shape: {x.shape}")
        return x


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims).contiguous()


class StochasticDepth(nn.Module):
    def __init__(self, p: float, mode: str) -> None:
        super().__init__()
        self.p = p
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p <= 0.0 or not self.training:
            return x
        if self.mode == "row":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
            noise = x.new_tensor(data=torch.ones(shape)).uniform_(0.0, 1.0) < self.p
            return x.div_(1.0 - self.p).mul_(noise)
        elif self.mode == "batch":
            shape = [x.shape[0]] + [1] * (x.ndim - 1)
            noise = x.new_tensor(data=torch.ones(shape)).uniform_(0.0, 1.0) < self.p
            return x.div_(1.0 - self.p).mul_(noise)
        else:
            raise NotImplementedError


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            LayerNorm2d(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        logger.debug(f"ConvNeXtBlock input shape: {input.shape}")
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        logger.debug(f"ConvNeXtBlock output shape: {result.shape}")
        return result


class EnhancedBlock(nn.Module):
    def __init__(self, dim, layer_scale=1e-6, stochastic_depth_prob=0.0):
        super().__init__()
        self.convnext = ConvNeXtBlock(dim, layer_scale, stochastic_depth_prob)

    def forward(self, x):
        logger.debug(f"EnhancedBlock input shape: {x.shape}")
        x = self.convnext(x)
        logger.debug(f"EnhancedBlock output shape: {x.shape}")
        return x
