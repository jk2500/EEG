#!/usr/bin/env python3
"""
Datasets and preprocessing utilities for ACFlow training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], str, Path]


@dataclass
class ChannelwiseStandardizer:
    """Simple per-channel z-score normalizer."""

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    eps: float = 1e-6

    def fit(self, data: np.ndarray) -> "ChannelwiseStandardizer":
        arr = np.asarray(data)
        if arr.ndim != 2:
            raise ValueError("ChannelwiseStandardizer expects [N, D] data.")

        self.mean = arr.mean(axis=0, dtype=np.float64)
        std = arr.std(axis=0, dtype=np.float64)
        self.std = np.maximum(std, self.eps)
        return self

    def transform(self, sample: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("ChannelwiseStandardizer must be fitted before use.")
        return (sample - self.mean) / self.std

    def __call__(self, sample: np.ndarray) -> np.ndarray:
        return self.transform(sample)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state: dict) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = state.get("eps", self.eps)


class EEGWindowDataset(Dataset):
    """
    Dataset wrapper for per-window EEG features stored as [N, D] arrays.

    Parameters
    ----------
    data : np.ndarray or path to .npy file
        Windows shaped [num_windows, num_channels].
    memmap : bool
        Use numpy memmap when loading from disk (useful for large arrays).
    normalizer : callable, optional
        Applied to each sample before conversion to tensor.
    fit_normalizer : bool
        If True and normalizer has a `fit` method, fit it on the full dataset.
    dtype : np.dtype
        Numpy dtype used before conversion to tensor.
    """

    def __init__(
        self,
        data: ArrayLike,
        *,
        memmap: bool = False,
        normalizer: Optional[ChannelwiseStandardizer] = None,
        fit_normalizer: bool = False,
        dtype: np.dtype = np.float32,
    ) -> None:
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(path)
            mmap_mode = "r" if memmap else None
            self._array = np.load(path, mmap_mode=mmap_mode)
        else:
            self._array = np.asarray(data, dtype=dtype)

        if self._array.ndim != 2:
            raise ValueError("EEGWindowDataset expects data shaped [N, D].")

        self.dtype = dtype
        self.normalizer = normalizer
        if fit_normalizer and self.normalizer is not None and hasattr(self.normalizer, "fit"):
            self.normalizer.fit(self._array)

    def __len__(self) -> int:
        return int(self._array.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self._array.shape[1])

    @property
    def array(self) -> np.ndarray:
        return self._array

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = np.asarray(self._array[idx], dtype=self.dtype)
        if self.normalizer is not None:
            sample = self.normalizer(sample)
        return torch.from_numpy(sample).float()


def create_dataloader(
    data: Union[Dataset, ArrayLike],
    *,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Convenience function to wrap numpy arrays or datasets into a DataLoader.
    """

    dataset = data if isinstance(data, Dataset) else EEGWindowDataset(data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
    )
