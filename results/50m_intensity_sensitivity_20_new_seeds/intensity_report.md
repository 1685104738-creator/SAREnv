# Frozen 50 m planner: independent validation and intensity sensitivity

## Fixed design

Formal parameters were frozen at Y=0.05, sigma=40 m and influence radius=120 m. The 20 seeds were imported from `09_final_experiment.py::TRIAL_SEEDS` and have no overlap with the parameter-selection seeds:

`113, 227, 331, 439, 547, 653, 761, 877, 983, 1091, 1193, 1297, 1423, 1523, 1627, 1733, 1831, 1931, 2027, 2131`

All **100/100 seed × intensity tests succeeded** (0 failed), producing 300 paired scenario rows. Each seed used one Original mission shared by all five intensity levels. Only source strength/activity was multiplied; background was unchanged.

## PART A — Independent Final Validation at 1x

These 20 seeds were unseen during Y selection. Deltas are Radiation-aware relative to the same seed's Original planner.

| Intensity | Scenario | SAR mean Δ% | Time mean Δ% | Found mean Δ | Total mean Δ% | P95 mean Δ% | Max mean Δ% | UAV mean Δ% | Total better /20 | P95 better /20 | Max better /20 | UAV better /20 | SAR ≥95% /20 | Mean episodes | Mean radiation steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1x | Single | -2.58 | -16.60 | -2.15 | -25.85 | -38.85 | -49.18 | 18.96 | 18 | 19 | 16 | 11 | 14 | 1.00 | 3600.00 |
| 1x | Multi | -5.60 | -25.52 | -4.90 | 1.31 | -19.44 | 55.63 | 14.08 | 13 | 17 | 12 | 6 | 13 | 1.00 | 3600.00 |
| 1x | Surface | -1.73 | -6.99 | -2.45 | -57.47 | -55.18 | -54.15 | 20.16 | 18 | 19 | 17 | 10 | 16 | 21.05 | 558.95 |

## PART B — Radiation Intensity Sensitivity

| Intensity | Scenario | SAR mean Δ% | Time mean Δ% | Found mean Δ | Total mean Δ% | P95 mean Δ% | Max mean Δ% | UAV mean Δ% | Total better /20 | P95 better /20 | Max better /20 | UAV better /20 | SAR ≥95% /20 | Mean episodes | Mean radiation steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25x | Single | -2.29 | -16.58 | -1.60 | -0.58 | -20.41 | -35.98 | 16.41 | 12 | 19 | 15 | 9 | 14 | 1.00 | 3600.00 |
| 0.25x | Multi | -6.43 | -25.85 | -5.30 | 12.57 | -16.07 | 59.73 | 14.58 | 8 | 17 | 11 | 7 | 11 | 1.00 | 3600.00 |
| 0.25x | Surface | -3.99 | -6.95 | -5.00 | -51.03 | -38.97 | -42.87 | 39.62 | 16 | 16 | 15 | 11 | 11 | 17.50 | 410.15 |
| 0.5x | Single | -1.24 | -16.13 | -0.55 | -17.06 | -28.72 | -42.75 | 8.87 | 15 | 19 | 14 | 10 | 15 | 1.00 | 3600.00 |
| 0.5x | Multi | -4.55 | -24.98 | -4.50 | 4.11 | -21.05 | 43.81 | 15.36 | 11 | 18 | 13 | 5 | 14 | 1.00 | 3600.00 |
| 0.5x | Surface | -1.46 | -6.94 | -1.10 | -61.84 | -49.12 | -58.90 | 44.74 | 17 | 18 | 17 | 10 | 17 | 18.80 | 551.60 |
| 1x | Single | -2.58 | -16.60 | -2.15 | -25.85 | -38.85 | -49.18 | 18.96 | 18 | 19 | 16 | 11 | 14 | 1.00 | 3600.00 |
| 1x | Multi | -5.60 | -25.52 | -4.90 | 1.31 | -19.44 | 55.63 | 14.08 | 13 | 17 | 12 | 6 | 13 | 1.00 | 3600.00 |
| 1x | Surface | -1.73 | -6.99 | -2.45 | -57.47 | -55.18 | -54.15 | 20.16 | 18 | 19 | 17 | 10 | 16 | 21.05 | 558.95 |
| 2x | Single | -0.85 | -16.89 | -0.50 | -37.74 | -51.49 | -46.59 | 6.18 | 20 | 20 | 15 | 10 | 15 | 1.00 | 3600.00 |
| 2x | Multi | -4.99 | -25.28 | -4.80 | 1.47 | -19.59 | 64.22 | 9.23 | 12 | 18 | 12 | 9 | 14 | 1.00 | 3600.00 |
| 2x | Surface | -2.63 | -7.38 | -4.00 | -55.58 | -54.31 | -53.05 | 46.29 | 18 | 18 | 17 | 14 | 16 | 20.75 | 628.05 |
| 4x | Single | -1.11 | -16.84 | -0.60 | -42.68 | -58.85 | -51.40 | 8.98 | 20 | 20 | 16 | 10 | 15 | 1.00 | 3600.00 |
| 4x | Multi | -5.51 | -25.53 | -4.95 | -0.26 | -20.83 | 59.54 | 12.32 | 14 | 18 | 13 | 7 | 14 | 1.00 | 3600.00 |
| 4x | Surface | -2.67 | -7.44 | -3.20 | -64.51 | -54.21 | -62.27 | 25.24 | 19 | 18 | 19 | 11 | 14 | 28.65 | 773.25 |

### Cross-morphology means

| Intensity | SAR Δ% | Time Δ% | Found Δ | Total Δ% | P95 Δ% | Max Δ% | UAV Δ% | Total better /60 | SAR ≥95% /60 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25x | -4.24 | -16.46 | -3.97 | -13.01 | -25.15 | -6.37 | 23.54 | 36 | 36 |
| 0.5x | -2.42 | -16.02 | -2.05 | -24.93 | -32.96 | -19.28 | 22.99 | 43 | 46 |
| 1x | -3.31 | -16.37 | -3.17 | -27.34 | -37.82 | -15.90 | 17.73 | 49 | 43 |
| 2x | -2.82 | -16.52 | -3.10 | -30.62 | -41.80 | -11.81 | 20.56 | 50 | 45 |
| 4x | -3.09 | -16.60 | -2.92 | -35.82 | -44.63 | -18.04 | 15.51 | 53 | 43 |

### Full 20-seed distributions

Each cell is `mean / median / sample SD [min, max]`. Absolute dose is not compared across intensity levels; all dose columns are paired RA-vs-Original percentage deltas within the same intensity.

| Intensity | Scenario | SAR Δ% | Time Δ% | Found Δ | Total Δ% | P95 Δ% | Max Δ% | UAV Δ% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.25x | Single | -2.29 / -1.21 / 6.29 [-16.99, 4.43] | -16.58 / -14.52 / 6.54 [-26.49, -7.89] | -1.60 / -2.00 / 5.07 [-12.00, 7.00] | -0.58 / -2.51 / 26.58 [-47.06, 59.60] | -20.41 / -15.77 / 20.78 [-62.18, 4.48] | -35.98 / -42.12 / 49.87 [-93.97, 68.66] | 16.41 / 4.23 / 43.53 [-56.92, 133.40] |
| 0.25x | Multi | -6.43 / -4.25 / 8.49 [-23.33, 4.32] | -25.85 / -25.86 / 7.49 [-37.70, -14.39] | -5.30 / -3.50 / 7.58 [-23.00, 4.00] | 12.57 / 12.01 / 42.52 [-65.02, 120.73] | -16.07 / -23.30 / 31.38 [-47.55, 90.86] | 59.73 / -6.08 / 175.95 [-97.15, 508.19] | 14.58 / 13.27 / 30.26 [-24.75, 100.03] |
| 0.25x | Surface | -3.99 / -3.24 / 4.47 [-16.40, 1.22] | -6.95 / -0.96 / 8.86 [-30.19, 0.24] | -5.00 / -4.50 / 3.87 [-13.00, 1.00] | -51.03 / -80.88 / 65.42 [-97.49, 181.21] | -38.97 / -47.12 / 51.19 [-94.34, 94.43] | -42.87 / -86.36 / 88.81 [-98.95, 217.64] | 39.62 / -4.14 / 124.38 [-73.47, 420.01] |
| 0.5x | Single | -1.24 / 1.10 / 5.36 [-15.13, 4.43] | -16.13 / -14.31 / 6.27 [-26.49, -8.27] | -0.55 / 0.00 / 4.90 [-12.00, 7.00] | -17.06 / -20.73 / 19.44 [-60.34, 17.10] | -28.72 / -27.15 / 17.25 [-62.73, 0.59] | -42.75 / -59.65 / 52.13 [-96.41, 71.18] | 8.87 / 1.43 / 32.00 [-56.53, 82.31] |
| 0.5x | Multi | -4.55 / -1.83 / 7.74 [-23.33, 4.32] | -24.98 / -25.18 / 7.94 [-37.70, -9.36] | -4.50 / -3.50 / 7.29 [-23.00, 4.00] | 4.11 / -7.40 / 47.06 [-71.19, 132.88] | -21.05 / -32.50 / 48.22 [-62.95, 167.10] | 43.81 / -25.38 / 175.99 [-96.59, 508.19] | 15.36 / 14.70 / 30.90 [-25.58, 100.01] |
| 0.5x | Surface | -1.46 / -0.70 / 5.00 [-16.75, 3.20] | -6.94 / -0.97 / 8.81 [-30.28, 0.03] | -1.10 / -1.00 / 5.50 [-14.00, 7.00] | -61.84 / -81.21 / 47.61 [-97.11, 82.40] | -49.12 / -57.21 / 37.58 [-94.34, 55.06] | -58.90 / -86.72 / 73.96 [-98.95, 218.06] | 44.74 / -0.78 / 158.81 [-62.72, 652.27] |
| 1x | Single | -2.58 / 1.63 / 8.56 [-30.78, 4.43] | -16.60 / -14.52 / 6.50 [-26.49, -8.27] | -2.15 / 0.00 / 8.95 [-34.00, 7.00] | -25.85 / -27.11 / 24.80 [-70.95, 22.68] | -38.85 / -39.59 / 16.18 [-67.68, 0.90] | -49.18 / -65.60 / 51.90 [-97.67, 72.82] | 18.96 / -0.19 / 64.37 [-56.92, 257.51] |
| 1x | Multi | -5.60 / -3.91 / 7.25 [-23.33, 4.08] | -25.52 / -25.19 / 7.70 [-37.70, -14.39] | -4.90 / -3.50 / 7.30 [-23.00, 4.00] | 1.31 / -9.33 / 52.17 [-75.65, 140.65] | -19.44 / -26.34 / 62.27 [-72.30, 225.86] | 55.63 / -27.18 / 182.93 [-96.76, 508.19] | 14.08 / 14.24 / 31.27 [-25.58, 100.01] |
| 1x | Surface | -1.73 / -0.51 / 4.80 [-17.83, 3.08] | -6.99 / -1.11 / 8.81 [-30.30, 0.02] | -2.45 / -2.00 / 5.61 [-18.00, 8.00] | -57.47 / -79.65 / 70.27 [-96.64, 214.87] | -55.18 / -60.36 / 30.47 [-94.34, 8.42] | -54.15 / -88.00 / 87.01 [-98.69, 251.67] | 20.16 / 0.76 / 83.10 [-81.01, 231.90] |
| 2x | Single | -0.85 / 1.82 / 5.46 [-15.13, 4.43] | -16.89 / -14.52 / 6.22 [-26.49, -8.27] | -0.50 / 0.50 / 4.92 [-12.00, 7.00] | -37.74 / -41.04 / 22.93 [-78.03, -1.87] | -51.49 / -53.94 / 17.05 [-75.01, -5.50] | -46.59 / -67.09 / 54.77 [-98.32, 73.66] | 6.18 / 1.41 / 35.79 [-56.92, 82.31] |
| 2x | Multi | -4.99 / -2.61 / 7.63 [-23.33, 4.08] | -25.28 / -25.18 / 8.12 [-37.70, -11.75] | -4.80 / -4.00 / 7.42 [-23.00, 4.00] | 1.47 / -10.88 / 59.39 [-78.02, 176.40] | -19.59 / -28.43 / 64.45 [-72.54, 233.91] | 64.22 / -18.87 / 195.38 [-96.85, 599.35] | 9.23 / 9.58 / 33.92 [-41.20, 100.01] |
| 2x | Surface | -2.63 / -2.16 / 5.90 [-23.08, 3.11] | -7.38 / -1.09 / 9.01 [-30.30, -0.04] | -4.00 / -3.50 / 7.30 [-26.00, 4.00] | -55.58 / -78.01 / 70.02 [-97.46, 217.40] | -54.31 / -69.72 / 32.49 [-94.37, 10.09] | -53.05 / -86.27 / 86.64 [-98.95, 253.82] | 46.29 / -23.66 / 243.73 [-72.60, 1039.73] |
| 4x | Single | -1.11 / 1.60 / 5.41 [-15.13, 4.43] | -16.84 / -14.52 / 6.28 [-26.49, -8.27] | -0.60 / 0.00 / 4.92 [-12.00, 7.00] | -42.68 / -48.00 / 24.35 [-82.68, -6.29] | -58.85 / -62.71 / 17.54 [-79.77, -13.51] | -51.40 / -68.18 / 53.56 [-98.65, 74.08] | 8.98 / 1.43 / 32.69 [-56.92, 82.31] |
| 4x | Multi | -5.51 / -3.79 / 7.27 [-23.33, 4.08] | -25.53 / -25.19 / 7.70 [-37.70, -14.39] | -4.95 / -3.50 / 7.29 [-23.00, 4.00] | -0.26 / -14.24 / 61.32 [-79.23, 179.89] | -20.83 / -33.85 / 67.32 [-72.53, 245.89] | 59.54 / -28.59 / 196.38 [-96.89, 599.35] | 12.32 / 13.09 / 31.59 [-25.58, 100.01] |
| 4x | Surface | -2.67 / -0.92 / 5.13 [-14.26, 3.37] | -7.44 / -1.09 / 9.14 [-30.24, 0.03] | -3.20 / -2.00 / 6.54 [-18.00, 6.00] | -64.51 / -77.69 / 45.73 [-97.48, 110.81] | -54.21 / -71.51 / 40.27 [-94.41, 62.48] | -62.27 / -86.27 / 69.81 [-98.95, 217.46] | 25.24 / -6.34 / 105.78 [-59.48, 404.14] |

## Interpretation

### Independent 1x validation

The unseen-seed validation supports a **morphology-qualified**, not universal, conclusion. At 1x, Single reduced survivor total/P95/maximum dose by 25.85%/38.85%/49.18% on average, with improvement in 18/19/16 seeds; Surface reductions were 57.47%/55.18%/54.15%, with improvement in 18/19/17 seeds. Their SAR likelihood costs were modest on average (-2.58% Single, -1.73% Surface), although time-discounted likelihood fell 16.60% and 6.99%.

Multi is the weak morphology. Its median total and maximum dose changes were beneficial (-9.33% and -27.18%), and P95 improved in 17/20 seeds, but large adverse outliers moved the mean total to +1.31% and mean maximum to +55.63%. Multi also had the largest rescue cost: SAR -5.60%, time-discounted likelihood -25.52%, and 4.90 fewer survivors on average. The frozen planner therefore validates clearly for Single and Surface radiation-risk reduction, but only partially for Multi.

UAV cumulative exposure is not consistently reduced: at 1x its mean change was +18.96%/+14.08%/+20.16% for Single/Multi/Surface, with reductions in only 11/6/10 seeds. The planner's demonstrated benefit is survivor pre-discovery dose, not lower UAV dose.

### Intensity trend

At 0.25x the planner has limited and morphology-dependent value. Single total dose was nearly neutral (-0.58%, 12/20 improved), Multi was worse on mean total (+12.57%, only 8/20 improved), while Surface still showed a large mean reduction (-51.03%, 16/20). The weak-hazard routes do **not** converge to Original: Point and Multi still enter one radiation episode lasting all 3600 steps, so SAR/time penalties remain.

At 0.5x a stable benefit begins for Single and Surface: mean total dose changes reached -17.06% and -61.84%, while their mean SAR costs were only -1.24% and -1.46%. Multi remained mixed (+4.11% mean total, although median -7.40% and P95 -21.05%).

At 1x the independent validation is strongest for Single and Surface as described above. At 2x and 4x, Single risk reduction strengthens monotonically: total dose changes progress to -37.74% and -42.68%, and all 20 seeds improve total and P95 at both levels. Surface remains strongly beneficial, reaching -64.51% total and -62.27% maximum at 4x. Multi improves only weakly with intensity: total is +1.47% at 2x and -0.26% at 4x, P95 remains about -20%, and maximum-dose means stay adverse because of outliers.

Increasing intensity does not cause a monotonic rescue-performance collapse. Cross-morphology SAR change ranges from -2.42% to -4.24%, survivors found from -2.05 to -3.97, and neither worsens steadily from 0.25x to 4x. The time-discounted penalty is essentially intensity-invariant at about 16.0-16.6% overall; it is morphology-driven (about 16.5% Single, 25.5% Multi and 7% Surface), not severity-driven.

The clearest hazard-benefit region is **2x-4x for Single and Surface**. Across all morphologies, total-dose improvement counts rise from 36/60 at 0.25x to 50/60 at 2x and 53/60 at 4x, while mean total reduction strengthens from -13.01% to -30.62% and -35.82%. This is evidence of useful intensity robustness, but not morphology-independent robustness: Multi remains the limiting case at every strength.

Surface radiation-mode activity grows at the strongest level (mean 28.65 episodes and 773.25 steps at 4x versus 17.50 and 410.15 at 0.25x). Single and Multi remain in one 3600-step episode at every tested intensity. Thus stronger hazards increase Surface radiation engagement, while point morphologies were already saturated even at 0.25x.

## Failures and output

Failed tests: 0. Raw paired data remain in `intensity_summary.csv`; runtime and errors are in `run_status.csv`. No per-run figures, arrays, routes or duplicated Original outputs were retained.
