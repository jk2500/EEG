# EEG Analysis Statistical Report
Generated on: 2025-07-08 07:57:34

## Data Summary
- Total subjects: 2
- Total conditions: 3
- Total metrics: 1
- Total data points: 1495

## Descriptive Statistics
                                    count      mean       std       min       25%       50%       75%       max
condition         metric_estimator                                                                             
awake_eyes_closed MIB-KSG           685.0  0.583260  0.301731 -0.181274  0.409865  0.590090  0.769103  1.350553
awake_eyes_open   MIB-KSG           677.0  0.408834  0.175914 -0.050111  0.303223  0.421096  0.517191  0.856363
sedation          MIB-KSG           133.0  0.701449  0.397790  0.011896  0.445575  0.602613  0.917962  1.805632

## Statistical Tests
### MIB-KSG
**awake_eyes_closed vs awake_eyes_open:**
- Mann-Whitney U: 327527.000, p = 0.000000
- Effect size (Cohen's d): 0.706
- Interpretation: Large

**awake_eyes_closed vs sedation:**
- Mann-Whitney U: 40526.000, p = 0.043844
- Effect size (Cohen's d): -0.335
- Interpretation: Medium

**awake_eyes_open vs sedation:**
- Mann-Whitney U: 23474.000, p = 0.000000
- Effect size (Cohen's d): -0.951
- Interpretation: Very Large

