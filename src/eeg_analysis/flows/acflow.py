#!/usr/bin/env python3
"""
Arbitrary-conditional normalizing flow for EEG MIB modeling.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn

from .config import ACFlowConfig


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    return nn.SiLU()


class Conditioner(nn.Module):
    """Mask-aware MLP used inside affine coupling blocks."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, depth: int, dropout: float, activation: str):
        super().__init__()
        layers: List[nn.Module] = []
        current_dim = in_dim
        act = _get_activation(activation)

        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    act,
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedAffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling that respects observed/conditioning masks.
    """

    def __init__(self, dim: int, base_mask: torch.Tensor, hidden_dim: int, config: ACFlowConfig):
        super().__init__()
        self.dim = dim
        self.register_buffer("base_mask", base_mask.float())
        conditioner_in = dim * 3  # pass-through, observed values, mask indicator
        self.conditioner = Conditioner(
            in_dim=conditioner_in,
            out_dim=dim * 2,
            hidden_dim=hidden_dim,
            depth=config.conditioner_depth,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.scale_clip = config.scale_clip

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_mask = self._effective_mask(observed_mask)
        x_pass = effective_mask * x
        x_transform = (1.0 - effective_mask) * x

        conditioner_input = torch.cat([x_pass, observed_mask * x, observed_mask], dim=-1)
        shift, log_scale = self._condition(conditioner_input, effective_mask)

        y_transform = x_transform * torch.exp(log_scale) + shift
        y = x_pass + y_transform
        log_det = (log_scale).sum(dim=-1)
        return y, log_det

    def inverse(self, y: torch.Tensor, observed_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_mask = self._effective_mask(observed_mask)
        y_pass = effective_mask * y
        y_transform = (1.0 - effective_mask) * y

        conditioner_input = torch.cat([y_pass, observed_mask * y, observed_mask], dim=-1)
        shift, log_scale = self._condition(conditioner_input, effective_mask)

        x_transform = (y_transform - shift) * torch.exp(-log_scale)
        x = y_pass + x_transform
        log_det = -(log_scale).sum(dim=-1)
        return x, log_det

    def _condition(self, conditioner_input: torch.Tensor, effective_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift, log_scale = torch.chunk(self.conditioner(conditioner_input), 2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.scale_clip

        transform_mask = (1.0 - effective_mask)
        log_scale = log_scale * transform_mask
        shift = shift * transform_mask
        return shift, log_scale

    def _effective_mask(self, observed_mask: torch.Tensor) -> torch.Tensor:
        active_mask = 1.0 - observed_mask
        base_mask = self.base_mask.unsqueeze(0)
        return torch.where(active_mask > 0, base_mask, torch.ones_like(base_mask))


class RandomPermutation(nn.Module):
    """Fixed random permutation used between coupling blocks."""

    def __init__(self, dim: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(dim, generator=generator)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(dim)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        if reverse:
            return x.index_select(1, self.inv_perm), observed_mask.index_select(1, self.inv_perm)
        return x.index_select(1, self.perm), observed_mask.index_select(1, self.perm)


class ACFlow(nn.Module):
    """
    Arbitrary-conditional normalizing flow built from mask-aware affine couplings.
    """

    def __init__(self, config: ACFlowConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.input_dim = config.input_dim
        self.layers = nn.ModuleList(self._build_layers(config))

    def forward(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        """Alias for log_prob to integrate with nn.Module conventions."""
        return self.log_prob(x, observed_mask)

    def log_prob(self, x: torch.Tensor, observed_mask: torch.Tensor) -> torch.Tensor:
        if observed_mask.dtype != x.dtype:
            observed_mask = observed_mask.to(dtype=x.dtype)
        z, log_det, _ = self._transform(x, observed_mask, reverse=False)
        active_mask = 1.0 - observed_mask

        log_base = -0.5 * ((z * z + math.log(2.0 * math.pi)) * active_mask).sum(dim=-1)
        return log_base + log_det

    def sample(
        self,
        observed_mask: torch.Tensor,
        observed_values: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Draw conditional samples given observed dimensions.

        Parameters
        ----------
        observed_mask : tensor [B, D]
            1 where the value is observed/conditioned upon.
        observed_values : tensor [B, D]
            Values for the observed subset (others are ignored).
        noise : tensor, optional
            Optional pre-sampled noise for the unobserved subset.
        """

        if observed_mask.dtype != observed_values.dtype:
            observed_mask = observed_mask.to(observed_values.dtype)

        device = observed_values.device
        if noise is None:
            noise = torch.randn_like(observed_values)

        active_mask = 1.0 - observed_mask
        base = observed_values * observed_mask + noise * active_mask
        x, _, _ = self._transform(base, observed_mask, reverse=True)
        return x

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _transform(
        self,
        x: torch.Tensor,
        observed_mask: torch.Tensor,
        reverse: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        modules = reversed(self.layers) if reverse else self.layers

        for layer in modules:
            if isinstance(layer, RandomPermutation):
                x, observed_mask = layer(x, observed_mask, reverse=reverse)
            elif reverse:
                x, delta = layer.inverse(x, observed_mask)
                log_det = log_det + delta
            else:
                x, delta = layer(x, observed_mask)
                log_det = log_det + delta

        return x, log_det, observed_mask

    def _build_layers(self, config: ACFlowConfig) -> Iterable[nn.Module]:
        layers: List[nn.Module] = []
        base_mask = self._alternating_mask(config.input_dim)

        for block_idx in range(config.num_blocks):
            mask = base_mask if block_idx % 2 == 0 else 1.0 - base_mask
            layers.append(
                MaskedAffineCoupling(
                    dim=config.input_dim,
                    base_mask=mask,
                    hidden_dim=config.hidden_dim,
                    config=config,
                )
            )
            if config.use_permutations and block_idx != config.num_blocks - 1:
                layers.append(RandomPermutation(config.input_dim, seed=block_idx))

        return layers

    @staticmethod
    def _alternating_mask(dim: int) -> torch.Tensor:
        mask = torch.zeros(dim)
        mask[::2] = 1.0
        return mask
