# Sub-1010 Results EDA (mib_analysis_optimal spectral binning)

## Scope and inputs
- Source: results/ds005620/mib_analysis_optimal/ds005620/spectral/binning/sub-1010
- Conditions: awake_eyes_closed, awake_eyes_open, sedation_1
- Bands: delta, theta, alpha, beta, gamma, broadband
- Each band: 50 repeats, 16 channels selected from 62 (random), 5 s epochs

## Summary of MIB levels (per-repeat means)
Condition means by band (higher means = higher MIB):
- awake_eyes_closed: delta 6.056, theta 6.599, alpha 6.671, beta 6.773, gamma 5.792, broadband 6.575
- awake_eyes_open:  delta 4.725, theta 4.395, alpha 5.563, beta 6.706, gamma 6.142, broadband 4.790
- sedation_1:       delta 6.020, theta 6.246, alpha 6.287, beta 6.390, gamma 5.961, broadband 6.559

Key patterns:
- awake_eyes_open is lowest for delta/theta/broadband and highest for gamma.
- awake_eyes_closed and sedation_1 are similar in delta and broadband, but sedation_1 is lower in theta/alpha/beta.
- beta changes are small for EC vs EO, but sedation_1 drops vs both awake conditions.

## Statistical comparisons (per-repeat means; N=50 per condition)
Welch t-test and Mann-Whitney were run per band; FDR correction applied.

Awake EC vs Awake EO (mean_diff, Cohen's d, t p-value):
- delta: +1.3306, d=4.85, p=5.6e-43 (EC higher)
- theta: +2.2039, d=7.53, p=4.7e-60 (EC higher)
- alpha: +1.1081, d=3.26, p=2.2e-29 (EC higher)
- beta:  +0.0677, d=0.22,  p=2.8e-01 (no clear difference)
- gamma: -0.3505, d=-1.08, p=5.1e-07 (EO higher)
- broadband: +1.7852, d=4.97, p=7.5e-43 (EC higher)

Sedation 1 vs Awake EO:
- delta/theta/alpha/broadband are all much higher in sedation (d=2.1 to 6.6, p<<1e-16)
- beta/gamma are lower in sedation (beta d=-0.89, p=2.2e-05; gamma d=-0.49, p=1.6e-02)

Sedation 1 vs Awake EC:
- delta and broadband are similar (d=-0.15 and -0.04; p>0.45)
- theta/alpha/beta are lower in sedation (d ~ -1.1 to -1.3; p<=1e-8)
- gamma slightly higher in sedation (d=0.42; p=3.9e-02)

## Stability and variability
- CV of per-repeat means (avg across bands):
  - awake_eyes_closed: 0.0506
  - awake_eyes_open:   0.0586 (least stable across repeats)
  - sedation_1:        0.0560
- Epoch stability CV (fixed channel set, avg across bands):
  - awake_eyes_closed: 0.066
  - awake_eyes_open:   0.144
  - sedation_1:        0.123
  This suggests greater epoch-to-epoch variability in awake_eyes_open and sedation_1.

## Channel selection behavior (sanity check)
- Expected selection count per channel: 50 * 16 / 62 = 12.9
- Observed counts range from 7 to 22 across conditions/bands.
- Mean Jaccard overlap between repeat channel sets: 0.153 (consistent with random selection).

## Outputs
- Tables:
  - analysis/outputs/ds005620/sub1010_results/sub1010_results_summary.csv
  - analysis/outputs/ds005620/sub1010_results/sub1010_repeat_means.csv
  - analysis/outputs/ds005620/sub1010_results/sub1010_epoch_distribution.csv
  - analysis/outputs/ds005620/sub1010_results/sub1010_epoch_stability.csv
  - analysis/outputs/ds005620/sub1010_results/sub1010_condition_comparisons.csv
  - analysis/outputs/ds005620/sub1010_results/sub1010_channel_selection_counts.csv
  - analysis/outputs/ds005620/sub1010_results/sub1010_selection_jaccard.csv
- Plots:
  - analysis/outputs/ds005620/sub1010_results/sub1010_repeat_means_boxplot.png
  - analysis/outputs/ds005620/sub1010_results/sub1010_band_means.png
  - analysis/outputs/ds005620/sub1010_results/sub1010_repeat_mean_cv.png
  - analysis/outputs/ds005620/sub1010_results/sub1010_epoch_stability_cv.png

## Notes
- These statistics reflect within-subject variability across random channel selections, not group-level inference.
- If you want, I can add a comparison against sub-1016 or aggregate across subjects in the results folder.
