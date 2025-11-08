#!/usr/bin/env python3
"""
Configuration helpers for the ACFlow training stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ACFlowConfig:
    """Hyperparameters controlling the flow architecture."""

    input_dim: int = 64
    hidden_dim: int = 512
    num_blocks: int = 8
    conditioner_depth: int = 3
    dropout: float = 0.05
    scale_clip: float = 5.0
    use_permutations: bool = True
    activation: str = "silu"

    def validate(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if self.conditioner_depth < 1:
            raise ValueError("conditioner_depth must be at least 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.scale_clip <= 0:
            raise ValueError("scale_clip must be positive")


@dataclass
class TrainerConfig:
    """Optimization hyperparameters and runtime settings."""

    batch_size: int = 1024
    max_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    use_amp: bool = True
    val_interval: int = 1
    early_stopping_patience: int = 15
    log_interval: int = 50
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    run_id: Optional[str] = None
    device: Optional[str] = None

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if self.val_interval <= 0:
            raise ValueError("val_interval must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


@dataclass
class MaskSamplerConfig:
    """Parameters governing the random mask sampler."""

    dim: int = 64
    uncond_prob: float = 0.2
    bipartition_prob: float = 0.5
    random_prob: float = 0.3
    min_condition: int = 8
    seed: int = 0
    allow_empty: bool = False

    def validate(self) -> None:
        if self.dim <= 1:
            raise ValueError("dim must be greater than 1")
        probs = [self.uncond_prob, self.bipartition_prob, self.random_prob]
        if any(p < 0 for p in probs):
            raise ValueError("Mask sampler probabilities must be non-negative")
        if sum(probs) == 0:
            raise ValueError("At least one mask sampler probability must be positive")
        if not 0 <= self.min_condition < self.dim:
            raise ValueError("min_condition must be between 0 and dim-1")

    def normalized_probs(self) -> tuple[float, float, float]:
        total = self.uncond_prob + self.bipartition_prob + self.random_prob
        return (
            self.uncond_prob / total,
            self.bipartition_prob / total,
            self.random_prob / total,
        )

