#!/usr/bin/env python3
"""
Lightweight tests for the ACFlow components.
"""

import numpy as np
import torch

from eeg_analysis.flows import (
    ACFlow,
    ACFlowConfig,
    ACFlowTrainer,
    ChannelwiseStandardizer,
    EEGWindowDataset,
    MaskSampler,
    MaskSamplerConfig,
    TrainerConfig,
    create_dataloader,
)


def test_acflow_log_prob_shapes():
    cfg = ACFlowConfig(input_dim=6, hidden_dim=32, num_blocks=2, conditioner_depth=2, dropout=0.0)
    model = ACFlow(cfg)
    x = torch.randn(5, cfg.input_dim)
    masks = torch.zeros_like(x)
    masks[:, :3] = 1.0
    log_prob = model.log_prob(x, masks)
    assert log_prob.shape == (5,)
    loss = -log_prob.mean()
    loss.backward()


def test_mask_sampler_respects_dimensions():
    sampler = MaskSampler(MaskSamplerConfig(dim=10, min_condition=2, seed=123))
    masks = sampler.sample(batch_size=16)
    assert masks.shape == (16, 10)
    # Every mask must leave at least one unobserved dimension
    assert torch.all(masks.sum(dim=1) < 10)


def test_trainer_single_epoch_runs():
    rng = np.random.default_rng(0)
    data = rng.standard_normal((64, 6)).astype(np.float32)
    scaler = ChannelwiseStandardizer().fit(data)
    dataset = EEGWindowDataset(data, normalizer=scaler)
    dataloader = create_dataloader(dataset, batch_size=16, shuffle=True, pin_memory=False)

    model = ACFlow(ACFlowConfig(input_dim=6, hidden_dim=64, num_blocks=3, conditioner_depth=2, dropout=0.0))
    trainer_cfg = TrainerConfig(batch_size=16, max_epochs=1, learning_rate=1e-3, log_interval=10, use_amp=False)
    sampler_cfg = MaskSamplerConfig(dim=6, min_condition=1, seed=42)
    trainer = ACFlowTrainer(model, trainer_cfg, mask_sampler=MaskSampler(sampler_cfg))
    trainer.fit(dataloader, verbose=False)
    assert trainer.global_step > 0

