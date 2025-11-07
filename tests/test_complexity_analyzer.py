import numpy as np
import pytest

from eeg_analysis.analyzers.complexity_analyzer import (
    ComplexityAnalyzer,
    BroadbandResult
)


class ConstantEstimator:
    name = "Constant"
    params = {}

    def calculate_integration(self, data, partition_indices):
        return float(np.mean(data))


def test_analyze_eeg_file_returns_expected_metrics(monkeypatch):
    estimator = ConstantEstimator()
    analyzer = ComplexityAnalyzer(
        estimator,
        verbose=False,
        n_channels=2,
        n_jobs=1,
        max_partitions=1,
        partition_seed=0
    )

    epochs = np.ones((3, 2, 10))
    channel_names = ['C1', 'C2']

    monkeypatch.setattr(analyzer, "_load_epochs", lambda _: (epochs, channel_names))
    result = analyzer.analyze_eeg_file("dummy_file.vhdr")

    assert isinstance(result, BroadbandResult)
    assert result.n_epochs == 3
    assert result.mean_metric == pytest.approx(1.0)
    assert result.std_metric == pytest.approx(0.0)


def test_run_analysis_spectral_path(monkeypatch, tmp_path):
    estimator = ConstantEstimator()
    analyzer = ComplexityAnalyzer(
        estimator,
        verbose=False,
        n_channels=2,
        n_jobs=1,
        max_partitions=1,
        partition_seed=0
    )

    fake_band_data = {
        'alpha': np.ones((2, 2, 10)),
        'beta': np.ones((1, 2, 10)) * 2
    }

    def fake_preprocess_by_bands(raw, **kwargs):
        return fake_band_data, ['C1', 'C2']

    monkeypatch.setattr(
        "eeg_analysis.analyzers.complexity_analyzer.preprocess_eeg_by_bands",
        fake_preprocess_by_bands
    )
    monkeypatch.setattr(analyzer, "_get_raw", lambda path: object())
    monkeypatch.setattr("eeg_analysis.analyzers.complexity_analyzer.plot_spectral_complexity_results", lambda *a, **k: None)
    monkeypatch.setattr("eeg_analysis.analyzers.complexity_analyzer.plot_results", lambda *a, **k: None)
    monkeypatch.setattr("eeg_analysis.analyzers.complexity_analyzer.print_spectral_summary", lambda *a, **k: None)

    plots_dir = tmp_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results = analyzer.run_analysis(
        analysis_type='spectral',
        file_paths={'condA': 'fileA.vhdr'},
        output_dir=str(tmp_path)
    )

    assert 'spectral' in results
    assert 'condA' in results['spectral']
    assert 'alpha' in results['spectral']['condA']
