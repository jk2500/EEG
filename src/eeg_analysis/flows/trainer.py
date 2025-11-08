#!/usr/bin/env python3
"""
Training utilities for the ACFlow model.
"""

from __future__ import annotations

import copy
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from .acflow import ACFlow
from .config import MaskSamplerConfig, TrainerConfig
from .mask_sampler import MaskSampler


class ACFlowTrainer:
    """
    Thin training harness around ACFlow with early stopping and AMP support.
    """

    def __init__(
        self,
        model: ACFlow,
        config: TrainerConfig,
        *,
        optimizer: Optional[torch.optim.Optimizer] = None,
        mask_sampler: Optional[MaskSampler] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(
            config.device if config.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.autocast = torch.cuda.amp.autocast if self.use_amp else nullcontext
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.mask_sampler = mask_sampler or MaskSampler(MaskSamplerConfig(dim=self.model.input_dim))
        self.history = []
        self.global_step = 0
        self.best_state_dict: Optional[dict] = None
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        *,
        checkpoint_name: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train the flow, optionally validating and checkpointing the best epoch.
        """

        best_val = float("inf")
        epochs_without_improvement = 0
        ckpt_path = None
        if checkpoint_name:
            ckpt_path = Path(self.config.checkpoint_dir) / checkpoint_name

        last_epoch = 0
        for epoch in range(1, self.config.max_epochs + 1):
            last_epoch = epoch
            start_time = time.time()
            train_metrics = self._run_epoch(train_loader, train=True, verbose=verbose)
            metrics = {"epoch": epoch, **train_metrics}

            if val_loader and epoch % self.config.val_interval == 0:
                val_metrics = self._run_epoch(val_loader, train=False, verbose=verbose)
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                current_val = val_metrics["loss"]

                if current_val + 1e-6 < best_val:
                    best_val = current_val
                    epochs_without_improvement = 0
                    self.best_state_dict = copy.deepcopy(self.model.state_dict())
                    if ckpt_path:
                        self.save_checkpoint(ckpt_path, extra={"epoch": epoch, "val_loss": current_val})
                else:
                    epochs_without_improvement += 1
                    if 0 < self.config.early_stopping_patience <= epochs_without_improvement:
                        if verbose:
                            print(f"[ACFlowTrainer] Early stopping at epoch {epoch}.")
                        break

            metrics["epoch_time_sec"] = time.time() - start_time
            self.history.append(metrics)
            if verbose:
                msg = f"[ACFlowTrainer] Epoch {epoch:03d} | loss={train_metrics['loss']:.3f}"
                if "val_loss" in metrics:
                    msg += f" | val_loss={metrics['val_loss']:.3f}"
                msg += f" | time={metrics['epoch_time_sec']:.1f}s"
                print(msg)

        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)
        elif ckpt_path and not val_loader:
            # Save the last state if we never entered the validation checkpoint logic.
            self.save_checkpoint(ckpt_path, extra={"epoch": last_epoch, "loss": train_metrics["loss"]})

        return self.history[-1] if self.history else {}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        *,
        masks: Optional[torch.Tensor] = None,
        batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Evaluate negative log-likelihood on a dataloader.
        """

        self.model.eval()
        total_loss = 0.0
        total_count = 0

        for batch_idx, batch in enumerate(dataloader):
            if batches is not None and batch_idx >= batches:
                break
            x = self._prepare_batch(batch)
            batch_masks = masks
            if batch_masks is None:
                batch_masks = self.mask_sampler.sample(x.size(0), device=self.device)
            log_prob = self.model.log_prob(x, batch_masks)
            total_loss += (-log_prob).sum().item()
            total_count += x.size(0)

        return {"loss": total_loss / max(total_count, 1)}

    def save_checkpoint(self, path: Path, extra: Optional[Dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "config": self.config.__dict__,
            "mask_sampler_config": self.mask_sampler.config.__dict__,
            "global_step": self.global_step,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_epoch(self, dataloader: DataLoader, *, train: bool, verbose: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_count = 0

        for batch_idx, batch in enumerate(dataloader):
            x = self._prepare_batch(batch)
            masks = self.mask_sampler.sample(x.size(0), device=self.device)

            with self.autocast():
                log_prob = self.model.log_prob(x, masks)
                loss = -log_prob.mean()

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    if self.config.gradient_clip:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.config.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()

                self.global_step += 1

            total_loss += loss.detach().item() * x.size(0)
            total_count += x.size(0)

            if train and verbose and self.global_step % self.config.log_interval == 0:
                print(f"[ACFlowTrainer] step={self.global_step} | loss={loss.detach().item():.4f}")

        mean_loss = total_loss / max(total_count, 1)
        return {"loss": mean_loss}

    def _prepare_batch(self, batch) -> torch.Tensor:
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch)
        return batch.to(self.device).float()
