#!/usr/bin/env python3
"""
Random mask sampler for arbitrary-conditional training.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np
import torch

from .config import MaskSamplerConfig


class MaskSampler:
    """
    Samples binary masks indicating observed (conditioned) dimensions.

    The sampler draws from a 3-component mixture:
      1. Unconditional (no observed dims, trains marginals).
      2. Bipartition-like (balanced observed size).
      3. Random subsets covering the remaining sizes.
    """

    def __init__(self, config: MaskSamplerConfig):
        self.config = config
        self.config.validate()
        self._rng = np.random.default_rng(self.config.seed)

    @property
    def dim(self) -> int:
        return self.config.dim

    def spawn(self, seed: int) -> "MaskSampler":
        """Return a cloned sampler with a different seed."""
        return MaskSampler(replace(self.config, seed=seed))

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Draw a batch of masks with shape [batch_size, dim].

        Returns
        -------
        torch.Tensor
            Binary tensor with 1 where the dimension is observed/conditioned.
        """

        masks = self._sample_with_rng(batch_size, self._rng)
        if device is not None:
            masks = masks.to(device)
        return masks

    def sample_fixed(self, batch_size: int, seed: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Deterministically draw masks using a provided seed (useful for validation).
        """

        rng = np.random.default_rng(seed)
        masks = self._sample_with_rng(batch_size, rng)
        if device is not None:
            masks = masks.to(device)
        return masks

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _sample_with_rng(self, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
        probs = self.config.normalized_probs()
        modes = rng.choice(3, size=batch_size, p=probs)
        masks = np.zeros((batch_size, self.dim), dtype=np.float32)

        min_obs = 0 if self.config.allow_empty else 1
        min_obs = max(min_obs, self.config.min_condition)
        max_obs = max(min(self.dim - min_obs, self.dim - 1), 1)

        for i, mode in enumerate(modes):
            if mode == 0:  # unconditional
                continue
            elif mode == 1:  # bipartition-like
                observed = self._triangular_size(rng, min_obs, max_obs)
            else:  # random subset
                observed = rng.integers(min_obs, max_obs + 1)

            observed = int(np.clip(observed, min_obs, max_obs))
            if observed <= 0:
                continue

            idx = rng.choice(self.dim, size=observed, replace=False)
            masks[i, idx] = 1.0

        return torch.from_numpy(masks)

    @staticmethod
    def _triangular_size(rng: np.random.Generator, low: int, high: int) -> int:
        """
        Sample an integer biased toward the center of [low, high].
        """

        if high <= low:
            return low
        # Triangular via summing uniforms
        span = high - low
        u = (rng.random() + rng.random()) * 0.5
        return low + int(round(u * span))
