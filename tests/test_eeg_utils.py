import numpy as np
import pytest

from eeg_analysis.eeg_utils import generate_bipartitions, EEGDataCache


def test_generate_bipartitions_full_count():
    partitions = generate_bipartitions(4, verbose=False)
    assert len(partitions) == 7  # Deterministic count for 4 channels


def test_generate_bipartitions_deterministic_seed():
    first = generate_bipartitions(6, max_partitions=5, random_state=7, verbose=False)
    second = generate_bipartitions(6, max_partitions=5, random_state=7, verbose=False)
    assert first == second
    assert len(first) == 5


def test_eeg_data_cache_reuses_and_evicts(monkeypatch):
    calls = []

    class DummyRaw:
        pass

    def fake_reader(path, preload=True, verbose=False):
        calls.append(path)
        return DummyRaw()

    monkeypatch.setattr(
        "eeg_analysis.eeg_utils.mne.io.read_raw_brainvision",
        fake_reader,
    )

    cache = EEGDataCache(max_cache_size=1)
    raw_a_first = cache.get_raw_data("file_a.vhdr", verbose=False)
    raw_a_second = cache.get_raw_data("file_a.vhdr", verbose=False)
    assert raw_a_first is raw_a_second
    assert calls == ["file_a.vhdr"]

    raw_b = cache.get_raw_data("file_b.vhdr", verbose=False)
    assert isinstance(raw_b, DummyRaw)
    assert calls == ["file_a.vhdr", "file_b.vhdr"]

    # file_a should have been evicted because max_cache_size == 1
    cache.get_raw_data("file_a.vhdr", verbose=False)
    assert calls == ["file_a.vhdr", "file_b.vhdr", "file_a.vhdr"]
