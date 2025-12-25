"""
Bipartition Generation
======================

Provides functions for generating channel bipartitions for MIB analysis.
"""

import numpy as np
from itertools import combinations

from ..utils.helpers import log_print


def generate_bipartitions(n_channels, max_partitions=None, verbose=True, random_state=None):
    """
    Generate all possible non-trivial bipartitions of channels deterministically.

    A bipartition divides n_channels into two non-empty, complementary subsets.
    For MIB analysis, we only need to store one subset of each partition (the
    complement is implicit).

    Parameters
    ----------
    n_channels : int
        Number of channels to partition.
    max_partitions : int, optional
        Maximum number of partitions to return. If None, returns all partitions.
    verbose : bool
        Whether to print progress messages.
    random_state : int, optional
        Random seed for reproducible partition sampling when max_partitions is set.

    Returns
    -------
    List[Tuple[int, ...]]
        List of tuples, each representing channel indices in one subset of a bipartition.
    """
    all_partitions = []
    for subset_size in range(1, n_channels // 2 + 1):
        for partition in combinations(range(n_channels), subset_size):
            if n_channels % 2 == 0 and subset_size == n_channels // 2:
                if partition[0] == 0:
                    all_partitions.append(partition)
            else:
                all_partitions.append(partition)

    total_partitions = len(all_partitions)
    if max_partitions and max_partitions < total_partitions:
        log_print(f"Limiting to {max_partitions} partitions sampled from {total_partitions}.", verbose)
        rng = np.random.default_rng(random_state)
        indices = rng.choice(total_partitions, max_partitions, replace=False)
        return [all_partitions[i] for i in indices]
    return all_partitions
