"""
Bipartition Generation
======================

Provides functions for generating channel bipartitions for MIB analysis.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..utils.helpers import log_print


def generate_bipartitions(
    n_channels: int,
    max_partitions: int | None = None,
    verbose: bool = True,
    random_state: int | None = None,
) -> list[tuple[int, ...]]:
    """
    Generate all non-trivial bipartitions of channels for MIB analysis.

    For n channels, there are 2^(n-1) - 1 unique bipartitions.
    Example: 4 channels -> 7 bipartitions, 8 channels -> 127 bipartitions.
    """
    all_partitions: list[tuple[int, ...]] = []

    # Generate subsets of size 1 to n//2 (larger subsets are complements)
    for subset_size in range(1, n_channels // 2 + 1):
        for partition in combinations(range(n_channels), subset_size):
            # For even n and half-size subsets, only keep those starting with 0
            # to avoid counting {0,1} and {2,3} as different partitions
            if n_channels % 2 == 0 and subset_size == n_channels // 2:
                if partition[0] == 0:
                    all_partitions.append(partition)
            else:
                all_partitions.append(partition)

    total_partitions = len(all_partitions)
    if max_partitions and max_partitions < total_partitions:
        log_print(
            f"Limiting to {max_partitions} partitions sampled from {total_partitions}.",
            verbose,
        )
        rng = np.random.default_rng(random_state)
        indices = rng.choice(total_partitions, max_partitions, replace=False)
        return [all_partitions[i] for i in indices]
    return all_partitions
