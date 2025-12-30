"""
Data Caching Utilities
======================

Provides efficient caching systems for EEG data to avoid redundant file loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mne

if TYPE_CHECKING:
    from mne.io import Raw


class EEGDataCache:
    """
    Efficient LRU caching system for raw EEG data.

    This cache stores loaded MNE Raw objects in memory to avoid redundant
    file I/O operations when processing the same files multiple times.

    Parameters
    ----------
    max_cache_size : int
        Maximum number of Raw objects to keep in cache (default: 3).

    Example
    -------
    >>> cache = EEGDataCache(max_cache_size=5)
    >>> raw = cache.get_raw_data('/path/to/file.vhdr')
    >>> # File is now cached for future access
    >>> raw2 = cache.get_raw_data('/path/to/file.vhdr')  # Returns cached version
    """

    def __init__(self, max_cache_size: int = 3) -> None:
        self.cache: dict[str, Any] = {}
        self.access_order: list[str] = []
        self.max_cache_size = max_cache_size

    def get_raw_data(self, file_path: str, verbose: bool = True) -> Raw:
        """
        Get raw EEG data from cache or load from file.

        Parameters
        ----------
        file_path : str
            Path to the BrainVision header file (.vhdr).
        verbose : bool
            Whether to print loading messages.

        Returns
        -------
        mne.io.Raw
            Raw EEG data object.
        """
        if file_path in self.cache:
            self.access_order.remove(file_path)
            self.access_order.append(file_path)
            return self.cache[file_path]

        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)

        if len(self.cache) >= self.max_cache_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[file_path] = raw
        self.access_order.append(file_path)
        return raw

    def clear(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        self.access_order.clear()

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self.cache)

    def __contains__(self, file_path: str) -> bool:
        """Check if file is in cache."""
        return file_path in self.cache
