#!/usr/bin/env python3
"""Raw EEG exploratory analysis for a single subject.

Usage:
    python analysis/scripts/eda_sub1010.py                           # Default: sub-1010
    python analysis/scripts/eda_sub1010.py --subject sub-1019
    python analysis/scripts/eda_sub1010.py --subject sub-1010 --dataset /path/to/ds005620
"""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
os.environ.setdefault("MNE_USE_NUMBA", "false")

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import kurtosis

import matplotlib.pyplot as plt


def make_run_pattern(subject: str) -> re.Pattern:
    """Create regex pattern for the subject's BrainVision files."""
    return re.compile(
        rf"{subject}_task-(?P<task>[^_]+)_acq-(?P<acq>[^_]+)(?:_run-(?P<run>\d+))?_eeg\.vhdr"
    )

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

WINDOW_SEC = 10.0
TARGET_SFREQ = 250.0
SAMPLES_PER_WINDOW = 2000
SAMPLE_CAP = 200_000
RNG = np.random.default_rng(7)


def parse_run_info(path: Path, pattern: re.Pattern) -> dict:
    match = pattern.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized filename: {path.name}")
    task = match.group("task")
    acq = match.group("acq")
    run = match.group("run")
    condition = f"{task}_{acq}"
    run_id = condition if run is None else f"{condition}_run-{run}"
    return {"task": task, "acq": acq, "run": run, "condition": condition, "run_id": run_id}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_channels(channels_path: Path) -> dict:
    df = pd.read_csv(channels_path, sep="\t")
    type_counts = df["type"].value_counts().to_dict()
    status_counts = df["status"].value_counts().to_dict()
    return {
        "channel_types": type_counts,
        "channel_status": status_counts,
        "n_channels": len(df),
    }


def summarize_events(events_path: Path) -> dict:
    df = pd.read_csv(events_path, sep="\t")
    if "trial_type" in df.columns:
        type_counts = df["trial_type"].value_counts().to_dict()
    else:
        type_counts = {}
    return {"n_events": len(df), "event_types": type_counts}


def compute_run_stats(vhdr_path: Path, pattern: re.Pattern) -> tuple[dict, pd.DataFrame, dict, tuple[np.ndarray, np.ndarray]]:
    run_info = parse_run_info(vhdr_path, pattern)
    eeg_json = vhdr_path.with_suffix(".json")
    channels_path = vhdr_path.with_name(vhdr_path.name.replace("_eeg.vhdr", "_channels.tsv"))
    events_path = vhdr_path.with_name(vhdr_path.name.replace("_eeg.vhdr", "_events.tsv"))

    eeg_meta = load_json(eeg_json)
    channel_summary = summarize_channels(channels_path)
    event_summary = summarize_events(events_path)

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose="ERROR")
    picks = mne.pick_types(raw.info, eeg=True, exclude=[])

    sfreq = float(raw.info["sfreq"])
    n_times = int(raw.n_times)
    duration = n_times / sfreq
    window_size = int(WINDOW_SEC * sfreq)

    n_channels = len(picks)
    sum_uV = np.zeros(n_channels, dtype=np.float64)
    sumsq_uV = np.zeros(n_channels, dtype=np.float64)
    min_uV = np.full(n_channels, np.inf, dtype=np.float64)
    max_uV = np.full(n_channels, -np.inf, dtype=np.float64)
    sample_count = 0

    sample_cache = []
    psd_sum = None
    psd_windows = 0
    decim = max(1, int(round(sfreq / TARGET_SFREQ)))
    psd_sfreq = sfreq / decim

    for start in range(0, n_times, window_size):
        stop = min(start + window_size, n_times)
        data = raw.get_data(start=start, stop=stop, picks=picks) * 1e6  # uV
        sample_count += data.shape[1]

        sum_uV += data.sum(axis=1)
        sumsq_uV += np.square(data).sum(axis=1)
        min_uV = np.minimum(min_uV, data.min(axis=1))
        max_uV = np.maximum(max_uV, data.max(axis=1))

        flat = data.reshape(-1)
        if flat.size:
            take = min(SAMPLES_PER_WINDOW, flat.size)
            idx = RNG.choice(flat.size, size=take, replace=False)
            sample_cache.append(flat[idx])

        data_decim = data[:, ::decim]
        if data_decim.shape[1] < 4:
            continue
        nperseg = min(1024, data_decim.shape[1])
        freqs, psd = welch(
            data_decim,
            fs=psd_sfreq,
            nperseg=nperseg,
            axis=-1,
        )
        psd_mean = psd.mean(axis=0)
        if psd_sum is None:
            psd_sum = np.zeros_like(psd_mean)
        psd_sum += psd_mean
        psd_windows += 1

    mean_uV = sum_uV / sample_count
    var_uV = sumsq_uV / sample_count - np.square(mean_uV)
    std_uV = np.sqrt(np.maximum(var_uV, 0.0))

    samples = np.concatenate(sample_cache) if sample_cache else np.array([], dtype=np.float32)
    if samples.size > SAMPLE_CAP:
        samples = RNG.choice(samples, size=SAMPLE_CAP, replace=False)

    global_stats = {
        "mean_uV": float(np.mean(mean_uV)),
        "std_uV": float(np.mean(std_uV)),
        "median_std_uV": float(np.median(std_uV)),
        "max_abs_uV": float(np.max(np.maximum(np.abs(min_uV), np.abs(max_uV)))),
    }
    if samples.size:
        global_stats.update(
            {
                "median_uV": float(np.median(samples)),
                "p05_uV": float(np.percentile(samples, 5)),
                "p95_uV": float(np.percentile(samples, 95)),
                "kurtosis": float(kurtosis(samples, fisher=False)),
            }
        )
    else:
        global_stats.update({"median_uV": np.nan, "p05_uV": np.nan, "p95_uV": np.nan, "kurtosis": np.nan})

    channel_stats = pd.DataFrame(
        {
            "run_id": run_info["run_id"],
            "channel": [raw.ch_names[p] for p in picks],
            "mean_uV": mean_uV,
            "std_uV": std_uV,
            "min_uV": min_uV,
            "max_uV": max_uV,
        }
    )

    median_std = np.median(std_uV)
    mad_std = np.median(np.abs(std_uV - median_std)) or 1.0
    noisy_mask = std_uV > (median_std + 3.0 * mad_std)
    flat_mask = std_uV < (median_std - 3.0 * mad_std)
    noisy_channels = [raw.ch_names[picks[i]] for i in np.where(noisy_mask)[0]]
    flat_channels = [raw.ch_names[picks[i]] for i in np.where(flat_mask)[0]]

    if psd_sum is None or psd_windows == 0:
        freqs = np.array([])
        psd_avg = np.array([])
        bandpower = {band: np.nan for band in BANDS}
        rel_bandpower = {f"rel_{band}": np.nan for band in BANDS}
        alpha_peak = np.nan
        total_power = np.nan
    else:
        psd_avg = psd_sum / psd_windows
        freq_mask = (freqs >= 1.0) & (freqs <= 45.0)
        total_power = float(np.trapezoid(psd_avg[freq_mask], freqs[freq_mask]))
        bandpower = {}
        rel_bandpower = {}
        for band, (low, high) in BANDS.items():
            band_mask = (freqs >= low) & (freqs <= high)
            power = float(np.trapezoid(psd_avg[band_mask], freqs[band_mask]))
            bandpower[band] = power
            rel_bandpower[f"rel_{band}"] = power / total_power if total_power else np.nan
        alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
        alpha_peak = float(freqs[alpha_mask][np.argmax(psd_avg[alpha_mask])]) if alpha_mask.any() else np.nan

    summary = {
        **run_info,
        "sfreq": sfreq,
        "n_times": n_times,
        "duration_s": duration,
        "json_recording_duration_s": eeg_meta.get("RecordingDuration"),
        "events_count": event_summary["n_events"],
        "event_types": json.dumps(event_summary["event_types"]),
        **global_stats,
        "alpha_peak_hz": alpha_peak,
        "total_power_uV2": total_power,
        "noisy_channels": ",".join(noisy_channels),
        "flat_channels": ",".join(flat_channels),
    }
    summary.update({f"band_{k}": v for k, v in bandpower.items()})
    summary.update(rel_bandpower)

    meta_info = {
        "channel_types": json.dumps(channel_summary["channel_types"]),
        "channel_status": json.dumps(channel_summary["channel_status"]),
    }
    summary.update(meta_info)

    return summary, channel_stats, bandpower | rel_bandpower, (freqs, psd_avg)


def run_eda(subject: str, dataset_dir: Path) -> None:
    """Run EDA for the specified subject."""
    eeg_dir = dataset_dir / subject / "eeg"
    out_dir = Path(f"analysis/outputs/ds005620/{subject}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pattern = make_run_pattern(subject)
    vhdr_paths = sorted(eeg_dir.glob(f"{subject}*_eeg.vhdr"))
    if not vhdr_paths:
        raise SystemExit(f"No BrainVision files found for {subject} in {eeg_dir}")

    summaries = []
    channel_stats_list = []
    bandpower_rows = []
    psd_by_condition = {}
    condition_counts = {}
    psd_freqs = None

    for vhdr_path in vhdr_paths:
        summary, channel_stats, bandpower, psd = compute_run_stats(vhdr_path, pattern)
        summaries.append(summary)
        channel_stats_list.append(channel_stats)

        bandpower_rows.append({"run_id": summary["run_id"], "condition": summary["condition"], **bandpower})

        freqs, psd_avg = psd
        if freqs.size:
            if psd_freqs is None:
                psd_freqs = freqs
            elif psd_freqs.shape != freqs.shape or not np.allclose(psd_freqs, freqs):
                raise RuntimeError("PSD frequency grid mismatch across runs.")
            condition = summary["condition"]
            if condition not in psd_by_condition:
                psd_by_condition[condition] = np.zeros_like(psd_avg)
                condition_counts[condition] = 0
            psd_by_condition[condition] += psd_avg
            condition_counts[condition] += 1

    run_summary_df = pd.DataFrame(summaries).sort_values("run_id")
    run_summary_df.to_csv(out_dir / "run_summary.csv", index=False)

    channel_stats_df = pd.concat(channel_stats_list, ignore_index=True)
    channel_stats_df.to_csv(out_dir / "channel_stats.csv", index=False)

    bandpower_df = pd.DataFrame(bandpower_rows)
    bandpower_df.to_csv(out_dir / "bandpower.csv", index=False)

    condition_summary = bandpower_df.groupby("condition").mean(numeric_only=True).reset_index()
    condition_summary.to_csv(out_dir / "condition_bandpower.csv", index=False)

    channel_std = (
        channel_stats_df.groupby("channel")["std_uV"].mean().sort_values(ascending=False).head(12)
    )
    plt.figure(figsize=(10, 5))
    channel_std.plot(kind="bar")
    plt.title(f"Top Channels by Mean Std (uV) - {subject}")
    plt.ylabel("Std (uV)")
    plt.tight_layout()
    plt.savefig(out_dir / "top_channel_std.png", dpi=150)
    plt.close()

    if psd_by_condition and psd_freqs is not None:
        plt.figure(figsize=(10, 6))
        for condition, psd_sum in psd_by_condition.items():
            psd_avg = psd_sum / condition_counts[condition]
            freq_mask = (psd_freqs >= 1.0) & (psd_freqs <= 45.0)
            plt.plot(
                psd_freqs[freq_mask],
                10 * np.log10(psd_avg[freq_mask]),
                label=condition,
            )
        plt.title(f"{subject}: Average PSD by Condition (1-45 Hz)")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (dB uV^2/Hz)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "psd_by_condition.png", dpi=150)
        plt.close()

    rel_cols = [col for col in condition_summary.columns if col.startswith("rel_")]
    if rel_cols:
        plt.figure(figsize=(10, 6))
        bar_df = condition_summary.set_index("condition")[rel_cols]
        bar_df.plot(kind="bar")
        plt.title(f"{subject}: Relative Bandpower by Condition")
        plt.ylabel("Relative Power")
        plt.tight_layout()
        plt.savefig(out_dir / "relative_bandpower.png", dpi=150)
        plt.close()

    print(f"EDA complete for {subject}. Results saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run raw EEG exploratory analysis for a single subject."
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="sub-1010",
        help="Subject ID (default: sub-1010)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ds005620",
        help="Path to the dataset directory (default: datasets/ds005620)",
    )
    args = parser.parse_args()
    run_eda(args.subject, Path(args.dataset))


if __name__ == "__main__":
    main()
