#!/usr/bin/env python3
"""Export analysis outputs into the standalone paper/ folder.

This script:
  - copies required figures into paper/figures/
  - generates LaTeX tables into paper/tables/ from analysis/outputs CSVs
  - copies source CSVs into paper/source_data/ for zipping

It assumes analysis outputs follow the conventions used by:
  - analysis/scripts/eda_results_all_subjects.py
  - analysis/scripts/eda_sub1010_results.py
  - analysis/scripts/analyze_bin_effect.py
  - analysis/scripts/plot_sedation_fourway.py
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing required file: {src}")
    _ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def _latex_escape(text: str) -> str:
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(" ", r"\ ")


def _num(value: float, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return r"\text{NA}"
    return rf"\num{{{value:.{places}f}}}"


def _p_sci(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return r"\text{NA}"
    return rf"\num{{{value:.2e}}}"


def _p_gauss(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return r"\text{NA}"
    if value == 0.0 or value < 1e-300:
        return r"$<10^{-300}$"
    return rf"\num{{{value:.2e}}}"


def _write_text(path: Path, content: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def _write_table_condition_means(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = [
        "band",
        "awake_eo_mean",
        "awake_eo_std",
        "awake_ec_mean",
        "awake_ec_std",
        "sedation_1_mean",
        "sedation_1_std",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{@{}l r r r@{}}")
    lines.append(r"\toprule")
    lines.append(r"Band & {awake EO} & {awake EC} & {sedation\_1}\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        band = _latex_escape(str(row["band"]))
        ao = rf"{_num(row['awake_eo_mean'])} $\pm$ {_num(row['awake_eo_std'])}"
        ac = rf"{_num(row['awake_ec_mean'])} $\pm$ {_num(row['awake_ec_std'])}"
        sed = rf"{_num(row['sedation_1_mean'])} $\pm$ {_num(row['sedation_1_std'])}"
        lines.append(rf"{band} & {ao} & {ac} & {sed} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_longtable_awake_vs_sedation(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = ["metric", "awake_label", "mean_diff", "ci_low", "ci_high", "pos_frac", "t_p", "wilcoxon_p", "cohen_d"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begingroup")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{1pt}")
    lines.append(r"\begin{longtable}{@{}l l r l r l l r@{}}")
    lines.append(rf"\caption{{{caption}}}\\")
    lines.append(rf"\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(r"Metric & Awake label & Mean diff & 95\% CI & Pos. frac & $p_t$ & $p_W$ & $d$\\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Metric & Awake label & Mean diff & 95\% CI & Pos. frac & $p_t$ & $p_W$ & $d$\\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for _, row in df.iterrows():
        metric = _latex_escape(str(row["metric"]))
        awake = _latex_escape(str(row["awake_label"]))
        mean = _num(float(row["mean_diff"]))
        ci = rf"[{_num(float(row['ci_low']))}, {_num(float(row['ci_high']))}]"
        pos = _num(float(row["pos_frac"]), places=2)
        pt = _p_sci(float(row["t_p"]))
        pw = _p_sci(float(row["wilcoxon_p"]))
        d = _num(float(row["cohen_d"]))
        lines.append(rf"{metric} & {awake} & {mean} & {ci} & {pos} & {pt} & {pw} & {d} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_table_sub1010_means(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = ["condition", "band", "overall_mean_of_repeat_means"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    band_order = ["delta", "theta", "alpha", "beta", "gamma", "broadband"]
    cond_map = {
        "awake_eyes_open": "awake EO",
        "awake_eyes_closed": "awake EC",
        "sedation_1": "sedation\\_1",
    }
    pivot = (
        df[df["condition"].isin(cond_map.keys())]
        .pivot_table(index="band", columns="condition", values="overall_mean_of_repeat_means", aggfunc="first")
        .reindex(band_order)
    )
    if pivot.empty:
        raise SystemExit(f"No expected conditions found in {csv_path}")

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{@{}l r r r@{}}")
    lines.append(r"\toprule")
    lines.append(r"Band & {awake EO} & {awake EC} & {sedation\_1}\\")
    lines.append(r"\midrule")
    for band in band_order:
        if band not in pivot.index:
            continue
        row = pivot.loc[band]
        lines.append(
            rf"{_latex_escape(band)} & {_num(row['awake_eyes_open'])} & {_num(row['awake_eyes_closed'])} & {_num(row['sedation_1'])} \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_longtable_sub1010_comparisons(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = ["band", "cond_a", "cond_b", "mean_diff", "cohen_d", "t_pvalue_fdr", "mw_pvalue_fdr"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begingroup")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\begin{longtable}{@{}l l r r l l@{}}")
    lines.append(rf"\caption{{{caption}}}\\")
    lines.append(rf"\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(r"Band & Comparison & {Mean diff} & {$d$} & {$p_{t,\,\mathrm{FDR}}$} & {$p_{\mathrm{MW},\,\mathrm{FDR}}$}\\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Band & Comparison & {Mean diff} & {$d$} & {$p_{t,\,\mathrm{FDR}}$} & {$p_{\mathrm{MW},\,\mathrm{FDR}}$}\\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    df = df.sort_values(["band", "cond_a", "cond_b"])
    for _, row in df.iterrows():
        band = _latex_escape(str(row["band"]))
        comp = _latex_escape(f"{row['cond_a']} - {row['cond_b']}")
        lines.append(
            rf"{band} & {comp} & {_num(float(row['mean_diff']))} & {_num(float(row['cohen_d']))} & {_p_sci(float(row['t_pvalue_fdr']))} & {_p_sci(float(row['mw_pvalue_fdr']))} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_table_gaussianity(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = [
        "condition",
        "n_samples",
        "skewness",
        "kurtosis_excess",
        "shapiro_p",
        "dagostino_p",
        "jarque_bera_p",
        "pct_channels_normal_shapiro",
        "pct_channels_normal_jb",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    df = df.copy()
    df["condition"] = df["condition"].astype(str)
    df = df.set_index("condition").reindex(["Awake EO", "Awake EC", "Sedation"]).reset_index()

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begingroup")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\begin{tabular}{@{}l r r r l l l r r@{}}")
    lines.append(r"\toprule")
    lines.append(r"Condition & $N$ samples & Skew & Kurt. (excess) & Shapiro $p$ & D'Agostino $p$ & JB $p$ & \% pass Shapiro & \% pass JB\\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        cond = _latex_escape(str(row["condition"]))
        n = rf"\num{{{int(row['n_samples'])}}}"
        skew = _num(float(row["skewness"]))
        kurt = _num(float(row["kurtosis_excess"]))
        sh = _p_sci(float(row["shapiro_p"]))
        dp = _p_gauss(float(row["dagostino_p"]))
        jb = _p_gauss(float(row["jarque_bera_p"]))
        pct_sw = rf"{_num(float(row['pct_channels_normal_shapiro']), places=1)}\%"
        pct_jb = rf"{_num(float(row['pct_channels_normal_jb']), places=1)}\%"
        lines.append(rf"{cond} & {n} & {skew} & {kurt} & {sh} & {dp} & {jb} & {pct_sw} & {pct_jb} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\endgroup")
    lines.append(r"\end{table}")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_table_bin_count_keypoints(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
    main_bins: int,
    high_bins: int,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = [
        "condition",
        "n_bins",
        "mean_mib",
        "std_mib",
        "gaussian_mean_mib",
        "gaussian_std_mib",
        "pct_of_gaussian_mean",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    df = df.copy()
    df["n_bins"] = df["n_bins"].astype(int)

    cond_order = ["Awake EO", "Awake EC", "Sedation"]
    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{@{}l l l l@{}}")
    lines.append(r"\toprule")
    lines.append(
        rf"Condition & Gaussian baseline & Binning ($B={main_bins}$) & Binning ($B={high_bins}$)\\"
    )
    lines.append(r"\midrule")

    for cond in cond_order:
        sub = df[df["condition"] == cond]
        if sub.empty:
            raise SystemExit(f"Missing condition {cond} in {csv_path}")
        # Gaussian baseline is constant across bins; read from any row.
        gauss_mean = float(sub.iloc[0]["gaussian_mean_mib"])
        gauss_std = float(sub.iloc[0]["gaussian_std_mib"])
        gauss = rf"{_num(gauss_mean)} $\pm$ {_num(gauss_std)}"

        def _cell(n_bins: int) -> str:
            row = sub[sub["n_bins"] == n_bins]
            if row.empty:
                raise SystemExit(f"Missing n_bins={n_bins} for condition {cond} in {csv_path}")
            r0 = row.iloc[0]
            mean = float(r0["mean_mib"])
            std = float(r0["std_mib"])
            pct = float(r0["pct_of_gaussian_mean"])
            return rf"{_num(mean)} $\pm$ {_num(std)} ({_num(pct, places=1)}\%)"

        lines.append(
            rf"{_latex_escape(cond)} & {gauss} & {_cell(main_bins)} & {_cell(high_bins)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_longtable_sedation_summary(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = ["band", "condition", "n_subjects", "mean", "std", "ci_low", "ci_high"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    df = df.copy()
    df = df.sort_values(["band", "condition"])

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begin{longtable}{@{}l l r r r l@{}}")
    lines.append(rf"\caption{{{caption}}}\\")
    lines.append(rf"\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(r"Band & Condition & $N$ & Mean & SD & 95\% CI\\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Band & Condition & $N$ & Mean & SD & 95\% CI\\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")
    for _, row in df.iterrows():
        band = _latex_escape(str(row["band"]))
        cond = _latex_escape(str(row["condition"]))
        n = rf"\num{{{int(row['n_subjects'])}}}"
        mean = _num(float(row["mean"]))
        sd = _num(float(row["std"]))
        ci = rf"[{_num(float(row['ci_low']))}, {_num(float(row['ci_high']))}]"
        lines.append(rf"{band} & {cond} & {n} & {mean} & {sd} & {ci} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    _write_text(out_path, "\n".join(lines) + "\n")


def _write_longtable_sedation_pairwise(
    *,
    csv_path: Path,
    out_path: Path,
    caption: str,
    label: str,
    header_note: str,
) -> None:
    df = pd.read_csv(csv_path)
    required_cols = ["band", "pair", "n_subjects", "mean_diff", "ci_low", "ci_high", "t_p"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing}")

    df = df.copy().sort_values(["band", "pair"])

    lines = []
    lines.append(f"% Auto-generated from {header_note}")
    lines.append(r"\begingroup")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\begin{longtable}{@{}l l r r l l@{}}")
    lines.append(rf"\caption{{{caption}}}\\")
    lines.append(rf"\label{{{label}}}\\")
    lines.append(r"\toprule")
    lines.append(r"Band & Pair & $N$ & Mean diff & 95\% CI & $p$\\")
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")
    lines.append(r"\toprule")
    lines.append(r"Band & Pair & $N$ & Mean diff & 95\% CI & $p$\\")
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    for _, row in df.iterrows():
        band = _latex_escape(str(row["band"]))
        pair = _latex_escape(str(row["pair"]))
        n = rf"\num{{{int(row['n_subjects'])}}}"
        mean = _num(float(row["mean_diff"]))
        ci = rf"[{_num(float(row['ci_low']))}, {_num(float(row['ci_high']))}]"
        p = _p_sci(float(row["t_p"]))
        lines.append(rf"{band} & {pair} & {n} & {mean} & {ci} & {p} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    lines.append(r"\endgroup")
    _write_text(out_path, "\n".join(lines) + "\n")


def export_all(*, analysis_outputs: Path, paper_dir: Path, main_bins: int, high_bins: int) -> None:
    # ---------------------------------------------------------------------
    # Copy figures
    # ---------------------------------------------------------------------
    figures_map = [
        (analysis_outputs / "robustness" / "gaussianity_comparison.png", paper_dir / "figures" / "gaussianity_comparison.png"),
        (analysis_outputs / "robustness" / "bin_count_effect_multi.png", paper_dir / "figures" / "bin_count_effect_multi.png"),
        (analysis_outputs / "robustness" / "epoch_length_effect.png", paper_dir / "figures" / "epoch_length_effect.png"),
        (analysis_outputs / "robustness" / "channel_count_effect.png", paper_dir / "figures" / "channel_count_effect_extended.png"),
        (analysis_outputs / "ds005620" / "all_subjects_results" / "awake_vs_sedation_by_band.png",
         paper_dir / "figures" / "ds005620" / "all_subjects_results" / "awake_vs_sedation_by_band.png"),
        (analysis_outputs / "ds005620" / "all_subjects_results" / "awake_avg_vs_sedation_composites.png",
         paper_dir / "figures" / "ds005620" / "all_subjects_results" / "awake_avg_vs_sedation_composites.png"),
        (analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_awake_avg_vs_sedation_by_band.png",
         paper_dir / "figures" / "ds005620" / "all_subjects_results_epoch10" / "epoch10_awake_avg_vs_sedation_by_band.png"),
        (analysis_outputs / "ds005620" / "sub1010_results" / "sub1010_band_means.png",
         paper_dir / "figures" / "ds005620" / "sub1010_results" / "sub1010_band_means.png"),
        (analysis_outputs / "sedation_resting_state" / "fourway_all_bands.png",
         paper_dir / "figures" / "sedation_resting_state" / "fourway_all_bands.png"),
        (analysis_outputs / "sedation_resting_state" / "fourway_band_heatmap.png",
         paper_dir / "figures" / "sedation_resting_state" / "fourway_band_heatmap.png"),
        (analysis_outputs / "sedation_resting_state" / "alpha_threeway_comparison.png",
         paper_dir / "figures" / "sedation_resting_state" / "alpha_threeway_comparison.png"),
    ]
    for src, dst in figures_map:
        _copy_file(src, dst)

    # ---------------------------------------------------------------------
    # Copy source CSVs
    # ---------------------------------------------------------------------
    source_root = paper_dir / "source_data"
    _ensure_dir(source_root)

    csv_inputs = [
        analysis_outputs / "robustness" / "ds005620_sub1067_gaussianity_summary.csv",
        analysis_outputs / "robustness" / "ds005620_sub1067_bin_count_sweep.csv",
        analysis_outputs / "ds005620" / "all_subjects_results" / "all_subjects_band_means.csv",
        analysis_outputs / "ds005620" / "all_subjects_results" / "all_subjects_band_means_long.csv",
        analysis_outputs / "ds005620" / "all_subjects_results" / "all_subjects_awake_vs_sedation.csv",
        analysis_outputs / "ds005620" / "all_subjects_results" / "all_subjects_composites.csv",
        analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_band_means.csv",
        analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_band_means_long.csv",
        analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_awake_vs_sedation.csv",
        analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_composites.csv",
        analysis_outputs / "ds005620" / "sub1010_results" / "sub1010_results_summary.csv",
        analysis_outputs / "ds005620" / "sub1010_results" / "sub1010_condition_comparisons.csv",
        analysis_outputs / "sedation_resting_state" / "fourway_band_means_raw.csv",
        analysis_outputs / "sedation_resting_state" / "fourway_band_summary.csv",
        analysis_outputs / "sedation_resting_state" / "fourway_band_pairwise.csv",
    ]
    for src in csv_inputs:
        rel = src.relative_to(analysis_outputs)
        _copy_file(src, source_root / rel)

    # ---------------------------------------------------------------------
    # Generate LaTeX tables
    # ---------------------------------------------------------------------
    tables_dir = paper_dir / "tables"

    _write_table_gaussianity(
        csv_path=analysis_outputs / "robustness" / "ds005620_sub1067_gaussianity_summary.csv",
        out_path=tables_dir / "robustness_gaussianity.tex",
        caption="Gaussianity diagnostics on preprocessed broadband epochs (ds005620 sub-1067; $k=8$, \\SI{5}{s} epochs). All $p$-values test the Gaussian null and are reported for overall pooled samples.",
        label="tab:gaussianity",
        header_note="analysis/outputs/robustness/ds005620_sub1067_gaussianity_summary.csv",
    )

    _write_table_bin_count_keypoints(
        csv_path=analysis_outputs / "robustness" / "ds005620_sub1067_bin_count_sweep.csv",
        out_path=tables_dir / "robustness_bin_count_keypoints.tex",
        caption=(
            "Bin-count sensitivity summary (ds005620 sub-1067; $k=8$, \\SI{5}{s} epochs). "
            f"For each condition we show the Gaussian baseline MIB and the binned MIB at $B={main_bins}$ "
            f"(used in the main pipeline) and $B={high_bins}$, with the percentage of the Gaussian baseline recovered."
        ),
        label="tab:bin_count_keypoints",
        header_note="analysis/outputs/robustness/ds005620_sub1067_bin_count_sweep.csv",
        main_bins=main_bins,
        high_bins=high_bins,
    )

    _write_table_condition_means(
        csv_path=analysis_outputs / "ds005620" / "all_subjects_results" / "all_subjects_band_means.csv",
        out_path=tables_dir / "ds005620_condition_means_epoch5.tex",
        caption="ds005620 (BrainVision EEG-BIDS) condition-level means by band (epoch length \\SI{5}{s}; $N=20$). Each entry is the mean $\\pm$ SD across subjects of the subject-level mean-of-per-repeat means.",
        label="tab:ds005620_condition_means_epoch5",
        header_note="analysis/outputs/ds005620/all_subjects_results/all_subjects_band_means.csv",
    )

    _write_table_condition_means(
        csv_path=analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_band_means.csv",
        out_path=tables_dir / "ds005620_condition_means_epoch10.tex",
        caption="ds005620 (BrainVision EEG-BIDS) condition-level means by band (epoch length \\SI{10}{s}; $N=20$). Each entry is the mean $\\pm$ SD across subjects of the subject-level mean-of-per-repeat means.",
        label="tab:ds005620_condition_means_epoch10",
        header_note="analysis/outputs/ds005620/all_subjects_results_epoch10/epoch10_band_means.csv",
    )

    _write_longtable_awake_vs_sedation(
        csv_path=analysis_outputs / "ds005620" / "all_subjects_results" / "all_subjects_awake_vs_sedation.csv",
        out_path=tables_dir / "ds005620_all_subjects_awake_vs_sedation_epoch5.tex",
        caption="ds005620 (BrainVision EEG-BIDS) across-subject awake minus sedation differences (epoch length \\SI{5}{s}; $N=20$ subjects). Pos. frac is the fraction of subjects with a positive difference. The $t$-test is a one-sample test on within-subject differences; $p_W$ is Wilcoxon signed-rank.",
        label="tab:ds005620_all_subjects_epoch5",
        header_note="analysis/outputs/ds005620/all_subjects_results/all_subjects_awake_vs_sedation.csv",
    )

    _write_longtable_awake_vs_sedation(
        csv_path=analysis_outputs / "ds005620" / "all_subjects_results_epoch10" / "epoch10_awake_vs_sedation.csv",
        out_path=tables_dir / "ds005620_all_subjects_awake_vs_sedation_epoch10.tex",
        caption="ds005620 (BrainVision EEG-BIDS) across-subject awake minus sedation differences (epoch length \\SI{10}{s}; $N=20$ subjects). Statistics as in Table~\\ref{tab:ds005620_all_subjects_epoch5}.",
        label="tab:ds005620_all_subjects_epoch10",
        header_note="analysis/outputs/ds005620/all_subjects_results_epoch10/epoch10_awake_vs_sedation.csv",
    )

    _write_table_sub1010_means(
        csv_path=analysis_outputs / "ds005620" / "sub1010_results" / "sub1010_results_summary.csv",
        out_path=tables_dir / "ds005620_sub1010_condition_means.tex",
        caption="Sub-1010 (ds005620) spectral MIB summary (mean of per-repeat means; 50 repeats, $k=16$, \\SI{5}{s} epochs).",
        label="tab:sub1010_means",
        header_note="analysis/outputs/ds005620/sub1010_results/sub1010_results_summary.csv",
    )

    _write_longtable_sub1010_comparisons(
        csv_path=analysis_outputs / "ds005620" / "sub1010_results" / "sub1010_condition_comparisons.csv",
        out_path=tables_dir / "ds005620_sub1010_condition_comparisons.tex",
        caption="Sub-1010 repeat-level condition comparisons by band (50 repeats per condition; Welch $t$-test and Mann--Whitney U; Benjamini--Hochberg FDR across all bands/comparisons).",
        label="tab:sub1010_comparisons",
        header_note="analysis/outputs/ds005620/sub1010_results/sub1010_condition_comparisons.csv",
    )

    _write_longtable_sedation_summary(
        csv_path=analysis_outputs / "sedation_resting_state" / "fourway_band_summary.csv",
        out_path=tables_dir / "sedation_fourway_band_summary.tex",
        caption="Sedation-RestingState band-level summaries ($N=20$). Means/SD/CIs are computed across subjects on the per-subject mean-of-per-repeat means.",
        label="tab:sedation_fourway_summary",
        header_note="analysis/outputs/sedation_resting_state/fourway_band_summary.csv",
    )

    _write_longtable_sedation_pairwise(
        csv_path=analysis_outputs / "sedation_resting_state" / "fourway_band_pairwise.csv",
        out_path=tables_dir / "sedation_fourway_pairwise.tex",
        caption="Sedation-RestingState paired differences by band and condition pair ($N=20$; paired $t$-tests across subjects).",
        label="tab:sedation_fourway_pairwise",
        header_note="analysis/outputs/sedation_resting_state/fourway_band_pairwise.csv",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export analysis outputs into paper/ assets.")
    parser.add_argument(
        "--analysis-outputs",
        type=str,
        default="analysis/outputs",
        help="Root directory containing analysis outputs (default: analysis/outputs).",
    )
    parser.add_argument(
        "--paper-dir",
        type=str,
        default="paper",
        help="Paper directory to populate (default: paper).",
    )
    parser.add_argument(
        "--main-bins",
        type=int,
        default=100,
        help="Main bin count to highlight in the bin-sweep keypoints table (default: 100).",
    )
    parser.add_argument(
        "--high-bins",
        type=int,
        default=200,
        help="High bin count to highlight in the bin-sweep keypoints table (default: 200).",
    )
    args = parser.parse_args()
    export_all(
        analysis_outputs=Path(args.analysis_outputs),
        paper_dir=Path(args.paper_dir),
        main_bins=args.main_bins,
        high_bins=args.high_bins,
    )


if __name__ == "__main__":
    main()

