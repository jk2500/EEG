# All-Subjects Results EDA (mib_analysis_optimal spectral binning)

## Scope
- Source: results/ds005620/mib_analysis_optimal/ds005620/spectral/binning/sub-*/
- Conditions: awake_eyes_closed, awake_eyes_open, sedation_1
- Subjects: 21 total; 20 with sedation_1 + both awake conditions
- Bands: delta, theta, alpha, beta, gamma, broadband
- Each subject/condition: 50 repeats, 16 random channels, 5 s epochs

## Do the sub-1010 findings hold across subjects?
Short answer: **partially**.

- **Gamma is not a reliable “awake > sedation” separator.**
  - awake_avg − sed: mean diff +0.280, p=0.185 (not significant)
  - awake_EO − sed: +0.224, p=0.309 (ns)
  - awake_EC − sed: +0.336, p=0.115 (ns)

- **Alpha shows a modest, consistent awake advantage**, mainly from eyes-closed:
  - awake_EC − sed: +0.269, p=0.009
  - awake_EO − sed: +0.141, p=0.137 (ns)
  - awake_avg − sed: +0.205, p=0.027

- **Eyes-closed vs sedation is clearly higher** for delta/theta/alpha:
  - delta: +0.309, p=5e-06
  - theta: +0.249, p=0.003
  - alpha: +0.269, p=0.009

- **Eyes-open vs sedation is mixed**:
  - delta/broadband are **lower** in awake_EO (delta −0.422, p=2.4e-4; broadband −0.389, p=0.0027)
  - gamma/alpha/beta are slightly higher but not significant

## Composite metrics (awake_avg − sedation)
If you want a metric that is higher for both awake conditions, composites are more stable than any single band:
- high_abc = mean(alpha, beta, gamma): +0.203, p=0.037
- high_over_low = mean(alpha,beta,gamma) − mean(delta,theta): +0.190, p=0.031
- mid_alpha_beta = mean(alpha, beta): +0.164, p=0.047

## Takeaway
- A **single-band “awake > sedation”** discriminator does **not** hold across subjects.
- **Alpha** is the strongest single-band candidate, but mostly driven by eyes-closed.
- A **high-frequency composite** (alpha+beta+gamma) is the most robust way to keep both awake conditions higher than sedation.

## Outputs
- Tables:
  - analysis/outputs/ds005620/all_subjects_results/all_subjects_band_means.csv
  - analysis/outputs/ds005620/all_subjects_results/all_subjects_composites.csv
  - analysis/outputs/ds005620/all_subjects_results/all_subjects_awake_vs_sedation.csv
- Plots:
  - analysis/outputs/ds005620/all_subjects_results/awake_avg_vs_sedation_by_band.png
  - analysis/outputs/ds005620/all_subjects_results/awake_vs_sedation_by_band.png
  - analysis/outputs/ds005620/all_subjects_results/awake_avg_vs_sedation_composites.png
