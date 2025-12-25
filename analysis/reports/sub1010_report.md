# Sub-1010 EDA Summary (ds005620)

## Dataset slice
- Subject: sub-1010 (participants.tsv: sub-SD_1010, 25, Female, 3 awakenings; not excluded)
- Recordings: 8 BrainVision runs (65 channels, 5000 Hz)
  - awake_EO: 1 x 300 s
  - awake_EC: 1 x 300 s
  - sed_rest: 3 x 300 s (total 900 s)
  - sed2_rest: 3 x 60 s (total 180 s)
- Total duration: ~28.0 minutes
- Events: only “New Segment/” markers across runs (no task-structured events)
- Metadata check: recording durations in JSON match data duration within 0.0002 s

## Channel inventory and metadata consistency
- Channels.tsv marks all 65 channels as EEG and “good”.
- Raw data channel names include VEOG, HEOG, EMG (present in the last 3 channels).
  - This is a metadata mismatch: consider labeling these as EOG/EMG in downstream analysis.

## Signal amplitude overview (uV)
- Mean per-channel std (proxy for overall amplitude):
  - awake_EC: 201.4
  - awake_EO: 183.2
  - sed_rest: 226.1
  - sed2_rest: 48.2 (markedly lower variance than other conditions)
- Max absolute amplitudes reach ~7–10 mV in some runs (likely transient artifacts).
- Runs with flatlined channels (MAD-based):
  - sed2_rest_run-1: CP1, F5
  - sed2_rest_run-3: F8, Fp1

## Spectral summaries (1–45 Hz, relative bandpower)
Condition means:
- awake_EC: delta 0.704, theta 0.098, alpha 0.068, beta 0.057, gamma 0.053
- awake_EO: delta 0.733, theta 0.189, alpha 0.031, beta 0.008, gamma 0.004
- sed_rest: delta 0.533, theta 0.085, alpha 0.110, beta 0.190, gamma 0.058
- sed2_rest: delta 0.377, theta 0.062, alpha 0.270, beta 0.252, gamma 0.009

Notable patterns:
- Eyes-closed shows higher relative alpha than eyes-open (EC–EO ≈ +0.037), consistent with classic alpha enhancement.
- sed2_rest shifts toward higher alpha/beta and lower delta compared with sed_rest.
- Alpha peak frequency is faster in sedation runs (~11.5–12.7 Hz) than awake (~8–9 Hz).

## Channel variance hot spots (mean std, top 10)
TP9, CP3, C2, PO7, Fp1, C1, CP2, F6, CP1, HEOG
(HEOG/VEOG/EMG appear in high-variance lists, reinforcing the need for correct channel typing.)

## Outputs
- Tables:
  - analysis/outputs/ds005620/sub1010/sub1010_run_summary.csv
  - analysis/outputs/ds005620/sub1010/sub1010_bandpower.csv
  - analysis/outputs/ds005620/sub1010/sub1010_condition_bandpower.csv
  - analysis/outputs/ds005620/sub1010/sub1010_channel_stats.csv
- Plots:
  - analysis/outputs/ds005620/sub1010/sub1010_psd_by_condition.png
  - analysis/outputs/ds005620/sub1010/sub1010_relative_bandpower.png
  - analysis/outputs/ds005620/sub1010/sub1010_top_channel_std.png

## Suggested next analyses
- Re-label VEOG/HEOG/EMG channels and re-run PSD/bandpower excluding non-EEG.
- Apply re-referencing (e.g., average reference) and repeat spectral summaries.
- Artifact mitigation (blink/EMG rejection) before comparing conditions.
- Topographic maps of bandpower changes (EC vs EO, sed vs sed2).
