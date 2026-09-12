# 50 m UAV sensing: Y sensitivity over 20 paired seeds

## Fixed design and completion

The 20 seeds were imported directly from `examples/radiation/09_final_experiment.py::TRIAL_SEEDS`; none were generated for this sweep:

`101, 203, 307, 401, 509, 601, 709, 809, 907, 1009, 1103, 1201, 1307, 1409, 1511, 1601, 1709, 1801, 1907, 2003`

All **100/100 seed × Y tests succeeded** (0 failed), producing 300 paired Single/Multi/Surface rows. Each seed used one shared radiation-blind Original route and one shared truth definition per morphology. The only varied parameter was Y in `0.004, 0.01, 0.02, 0.05, 0.2`; sigma remained 40 m and influence radius 120 m.

Before the formal matrix, one disposable full sanity trial used seed 101 and Y=0.02. It completed all three morphologies and verified parameter propagation and summary extraction; its metrics were discarded and it was not counted among the 100 formal tests.

Statistics below use sample standard deviation (`n-1`). Dose improvement means a strictly lower value than that seed's Original baseline. “SAR near/or above” means at least 95% of Original.

## Paired results by morphology

| Y | Scenario | SAR mean Δ% | Found mean Δ | Total dose mean Δ% | P95 mean Δ% | Max mean Δ% | Total better /20 | P95 better /20 | Max better /20 | SAR ≥95% /20 | Mean episodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.004 | Single | -2.99 | -3.70 | -15.37 | -39.19 | -45.36 | 13 | 19 | 19 | 15 | 1.00 |
| 0.004 | Multi | -2.07 | -4.20 | -6.25 | -23.24 | -11.73 | 13 | 17 | 15 | 15 | 1.00 |
| 0.004 | Surface | -2.32 | -4.15 | -14.31 | -44.58 | 49.07 | 17 | 17 | 16 | 17 | 33.40 |
| 0.01 | Single | -4.10 | -4.45 | -13.86 | -39.91 | -14.06 | 14 | 20 | 18 | 14 | 1.00 |
| 0.01 | Multi | -1.90 | -4.30 | -7.71 | -22.37 | -11.49 | 13 | 17 | 15 | 16 | 1.00 |
| 0.01 | Surface | -1.18 | -2.65 | -13.49 | -43.86 | 48.45 | 16 | 17 | 15 | 18 | 28.35 |
| 0.02 | Single | -4.10 | -4.45 | -13.86 | -39.91 | -14.06 | 14 | 20 | 18 | 14 | 1.00 |
| 0.02 | Multi | -1.90 | -4.30 | -7.71 | -22.37 | -11.49 | 13 | 17 | 15 | 16 | 1.00 |
| 0.02 | Surface | -1.76 | -3.45 | -18.67 | -45.61 | 39.87 | 16 | 19 | 15 | 17 | 27.70 |
| 0.05 | Single | -4.25 | -4.70 | -16.94 | -39.09 | -49.61 | 14 | 19 | 19 | 14 | 1.00 |
| 0.05 | Multi | -2.24 | -4.40 | -6.41 | -22.79 | -11.73 | 13 | 17 | 15 | 15 | 1.00 |
| 0.05 | Surface | -2.24 | -4.05 | -18.13 | -43.77 | 42.96 | 17 | 18 | 15 | 16 | 24.05 |
| 0.2 | Single | -4.25 | -4.70 | -16.94 | -39.09 | -49.61 | 14 | 19 | 19 | 14 | 1.00 |
| 0.2 | Multi | -2.24 | -4.40 | -6.41 | -22.79 | -11.73 | 13 | 17 | 15 | 15 | 1.00 |
| 0.2 | Surface | -2.78 | -4.90 | -27.50 | -25.94 | -9.85 | 15 | 14 | 15 | 15 | 15.75 |

### Full distribution check

Every cell below is `mean / median / sample SD [min, max]` over the same 20 paired seeds. Percentage deltas are relative to that seed's Original result; Found is an absolute survivor-count delta. This exposes worst seeds rather than hiding them behind means.

| Y | Scenario | SAR Δ% | Time-discounted Δ% | Found Δ | Total dose Δ% | P95 Δ% | Max Δ% | UAV dose Δ% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.004 | Single | -2.99 / 1.86 / 8.85 [-24.04, 4.83] | -21.02 / -17.94 / 8.69 [-39.49, -12.00] | -3.70 / -1.00 / 9.12 [-26.00, 7.00] | -15.37 / -18.26 / 36.09 [-89.60, 52.69] | -39.19 / -44.82 / 20.00 [-63.95, 13.10] | -45.36 / -61.79 / 41.16 [-98.83, 40.90] | 16.68 / -12.76 / 75.42 [-57.03, 234.48] |
| 0.004 | Multi | -2.07 / 1.98 / 7.46 [-21.90, 4.54] | -24.48 / -23.48 / 7.46 [-37.43, -12.83] | -4.20 / -3.00 / 8.03 [-27.00, 6.00] | -6.25 / -9.57 / 39.27 [-62.62, 95.17] | -23.24 / -37.34 / 56.37 [-67.65, 192.37] | -11.73 / -45.29 / 109.17 [-89.01, 371.86] | 12.45 / 18.26 / 34.19 [-36.46, 84.20] |
| 0.004 | Surface | -2.32 / -0.26 / 5.77 [-16.39, 3.07] | -12.24 / -9.61 / 13.07 [-36.26, -0.04] | -4.15 / -2.00 / 7.08 [-18.00, 7.00] | -14.31 / -51.67 / 145.05 [-96.41, 535.60] | -44.58 / -56.39 / 47.94 [-93.18, 107.84] | 49.07 / -68.19 / 395.61 [-97.82, 1684.81] | 18.04 / 9.87 / 59.67 [-71.21, 131.03] |
| 0.01 | Single | -4.10 / 0.48 / 9.49 [-24.04, 4.58] | -21.31 / -17.94 / 8.86 [-39.49, -12.04] | -4.45 / -1.50 / 9.44 [-26.00, 7.00] | -13.86 / -18.26 / 40.48 [-89.60, 77.97] | -39.91 / -44.82 / 18.11 [-63.95, -3.26] | -14.06 / -70.09 / 172.03 [-98.83, 696.86] | 14.35 / -12.76 / 72.32 [-57.03, 234.48] |
| 0.01 | Multi | -1.90 / 1.00 / 7.16 [-21.90, 4.54] | -24.46 / -23.48 / 7.51 [-37.43, -12.24] | -4.30 / -2.50 / 7.76 [-27.00, 6.00] | -7.71 / -12.30 / 39.54 [-62.62, 95.17] | -22.37 / -36.23 / 56.37 [-67.65, 192.37] | -11.49 / -45.28 / 108.60 [-89.01, 371.86] | 9.90 / 18.26 / 28.68 [-36.46, 68.15] |
| 0.01 | Surface | -1.18 / -0.84 / 3.21 [-9.13, 3.03] | -12.23 / -9.79 / 13.09 [-36.26, 0.19] | -2.65 / -3.00 / 4.59 [-11.00, 7.00] | -13.49 / -50.81 / 139.50 [-96.41, 534.85] | -43.86 / -50.45 / 47.96 [-93.18, 107.84] | 48.45 / -56.49 / 390.96 [-97.82, 1682.62] | 5.26 / -12.55 / 59.00 [-67.64, 160.01] |
| 0.02 | Single | -4.10 / 0.48 / 9.49 [-24.04, 4.58] | -21.31 / -17.94 / 8.86 [-39.49, -12.04] | -4.45 / -1.50 / 9.44 [-26.00, 7.00] | -13.86 / -18.26 / 40.48 [-89.60, 77.97] | -39.91 / -44.82 / 18.11 [-63.95, -3.26] | -14.06 / -70.09 / 172.03 [-98.83, 696.86] | 14.35 / -12.76 / 72.32 [-57.03, 234.48] |
| 0.02 | Multi | -1.90 / 1.00 / 7.16 [-21.90, 4.54] | -24.46 / -23.48 / 7.51 [-37.43, -12.24] | -4.30 / -2.50 / 7.76 [-27.00, 6.00] | -7.71 / -12.30 / 39.54 [-62.62, 95.17] | -22.37 / -36.23 / 56.37 [-67.65, 192.37] | -11.49 / -45.28 / 108.60 [-89.01, 371.86] | 9.90 / 18.26 / 28.68 [-36.46, 68.15] |
| 0.02 | Surface | -1.76 / 0.11 / 5.88 [-23.54, 3.04] | -12.20 / -9.75 / 13.09 [-35.97, 0.19] | -3.45 / -3.00 / 6.23 [-20.00, 7.00] | -18.67 / -55.34 / 137.31 [-96.32, 532.54] | -45.61 / -46.61 / 44.37 [-93.18, 107.84] | 39.87 / -67.57 / 389.45 [-97.82, 1675.87] | 24.07 / -12.36 / 114.25 [-66.57, 431.13] |
| 0.05 | Single | -4.25 / 0.25 / 9.46 [-24.04, 4.58] | -21.20 / -17.94 / 8.96 [-39.49, -12.00] | -4.70 / -1.50 / 9.44 [-26.00, 7.00] | -16.94 / -18.26 / 35.11 [-89.60, 52.69] | -39.09 / -44.82 / 20.11 [-63.95, 13.10] | -49.61 / -70.09 / 40.78 [-98.83, 40.90] | 13.55 / -12.76 / 71.81 [-57.03, 234.48] |
| 0.05 | Multi | -2.24 / 1.55 / 7.38 [-21.90, 4.54] | -24.49 / -23.48 / 7.45 [-37.43, -12.83] | -4.40 / -3.00 / 7.92 [-27.00, 6.00] | -6.41 / -10.95 / 39.28 [-62.62, 95.17] | -22.79 / -37.23 / 56.22 [-67.65, 192.37] | -11.73 / -45.28 / 109.17 [-89.01, 371.86] | 13.31 / 18.26 / 33.27 [-36.46, 84.20] |
| 0.05 | Surface | -2.24 / -1.23 / 3.74 [-11.83, 2.50] | -12.22 / -9.87 / 13.12 [-36.26, 0.23] | -4.05 / -3.00 / 4.72 [-17.00, 4.00] | -18.13 / -56.81 / 138.71 [-96.35, 533.36] | -43.77 / -60.25 / 50.32 [-93.18, 128.20] | 42.96 / -68.08 / 391.25 [-97.82, 1677.39] | 22.12 / 6.82 / 66.62 [-74.36, 189.37] |
| 0.2 | Single | -4.25 / 0.25 / 9.46 [-24.04, 4.58] | -21.20 / -17.94 / 8.96 [-39.49, -12.00] | -4.70 / -1.50 / 9.44 [-26.00, 7.00] | -16.94 / -18.26 / 35.11 [-89.60, 52.69] | -39.09 / -44.82 / 20.11 [-63.95, 13.10] | -49.61 / -70.09 / 40.78 [-98.83, 40.90] | 13.55 / -12.76 / 71.81 [-57.03, 234.48] |
| 0.2 | Multi | -2.24 / 1.55 / 7.38 [-21.90, 4.54] | -24.49 / -23.48 / 7.45 [-37.43, -12.83] | -4.40 / -3.00 / 7.92 [-27.00, 6.00] | -6.41 / -10.95 / 39.28 [-62.62, 95.17] | -22.79 / -37.23 / 56.22 [-67.65, 192.37] | -11.73 / -45.28 / 109.17 [-89.01, 371.86] | 13.31 / 18.26 / 33.27 [-36.46, 84.20] |
| 0.2 | Surface | -2.78 / -1.31 / 6.44 [-24.20, 3.49] | -11.08 / -2.33 / 13.51 [-36.26, 2.23] | -4.90 / -3.50 / 7.35 [-23.00, 4.00] | -27.50 / -50.04 / 67.27 [-96.32, 169.01] | -25.94 / -33.44 / 60.58 [-92.94, 167.51] | -9.85 / -45.06 / 120.35 [-97.82, 370.27] | 41.81 / -17.24 / 166.38 [-60.34, 568.46] |

## Surface mode chatter

| Y | Episodes mean | median | SD | max | Radiation steps mean | median | SD | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.004 | 33.40 | 34.00 | 13.52 | 58.00 | 972.05 | 924.50 | 343.51 | 1767.00 |
| 0.01 | 28.35 | 27.00 | 14.20 | 73.00 | 852.05 | 822.50 | 315.51 | 1449.00 |
| 0.02 | 27.70 | 24.50 | 11.29 | 52.00 | 734.25 | 630.50 | 272.84 | 1260.00 |
| 0.05 | 24.05 | 21.00 | 10.03 | 39.00 | 623.15 | 605.00 | 223.19 | 1044.00 |
| 0.2 | 15.75 | 11.00 | 13.42 | 64.00 | 425.55 | 409.00 | 154.41 | 867.00 |

The most severe Surface chatter was **Y=0.004**, judged by the highest 20-seed mean episode count. Intermediate Y values should be read against both their dose columns and this episode/step table; fewer transitions alone is not evidence of a better rescue outcome.

## Cross-morphology robustness

Each row below pools 60 paired observations (20 seeds × 3 morphologies). A “catastrophic reverse” is a row with SAR below Original by more than 10%, or total/P95/maximum dose above Original by more than 50%.

| Y | SAR mean Δ% | Time-discounted mean Δ% | Found mean Δ | Total dose mean Δ% | Total better /60 | SAR within 5% /60 | Found ≥ Original /60 | Catastrophic reverses | Surface episode mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.004 | -2.46 | -19.25 | -4.02 | -11.97 | 43/60 | 47/60 | 23/60 | 19 | 33.40 |
| 0.01 | -2.39 | -19.33 | -3.80 | -11.69 | 43/60 | 48/60 | 19/60 | 19 | 28.35 |
| 0.02 | -2.58 | -19.32 | -4.07 | -13.41 | 43/60 | 47/60 | 20/60 | 18 | 27.70 |
| 0.05 | -2.91 | -19.31 | -4.38 | -13.83 | 44/60 | 45/60 | 19/60 | 16 | 24.05 |
| 0.2 | -3.09 | -18.92 | -4.67 | -16.95 | 42/60 | 44/60 | 20/60 | 18 | 15.75 |

Using a conservative ordering—fewest catastrophic reverses, then most total-dose improvements, SAR stability, survivor-count stability, and finally lower Surface switching—the 20-seed robust candidate is **Y=0.05**.

For **Single**, Y=0.004 is the most stable choice: it has the best mean SAR and survivor-count deltas while retaining strong P95/maximum-dose reductions. For **Multi**, Y=0.01 and Y=0.02 are an exact route/outcome tie across these 20 seeds and are jointly the most stable; Y=0.02 is preferable only for consistency with the cross-morphology direction, not because Multi distinguishes them. For **Surface**, Y=0.05 is the best trade-off: total/P95/maximum dose improved in 17/18/15 seeds, SAR remained at least 95% of Original in 16/20, and its episode distribution (mean 24.05, SD 10.03, max 39) is materially tighter than Y=0.004 (33.40, 13.52, 58), Y=0.01 (28.35, 14.20, 73), or Y=0.02 (27.70, 11.29, 52). Y=0.2 has the lowest mean episode count, but weaker Surface SAR, survivors-found and P95 reliability make it a less robust outcome choice.

The requested intermediate-Y question is therefore answered **yes**: Y=0.05 preserves or slightly improves the 20-seed radiation-risk pattern of Y=0.004 while reducing mean Surface episodes by 9.35 (28.0%) and mean Surface radiation-mode steps from 972.05 to 623.15 (35.9%). It produced fewer Surface episodes in 12/20 paired seeds. Single and Multi were unchanged in 19/20 seeds, so this trade-off is driven by Surface rather than by sacrificing the point-source cases.

No Y eliminates adverse tails. At Y=0.05, seed 709 still has Surface total dose +533.36% and maximum dose +1677.39% versus its low Original values, and seed 1801 has P95 +128.20%. Y=0.2 reduces that maximum-dose outlier but still has worst-case Surface total/P95 increases of +169.01%/+167.51%, alongside worse rescue outcomes. These worst seeds are why Y=0.05 is a robust compromise rather than a claim of universal improvement.

Time-discounted likelihood is a common trade-off for every Y: its cross-morphology mean loss remains about 19%, mainly because Single/Multi stay in radiation mode for the full 3600 steps. The Y selection does not remove that limitation.

This recommendation is based on all 20 common seeds, not seed 42. It is a parameter-selection result only: sigma, influence radius, sources, planner, trigger structure, estimator, sensing/evaluation heights, mission settings, and all other formal parameters were unchanged. No additional Y, seed, or follow-on experiment was run.

## Output and failures

Failed tests: 0. The master CSV retains every raw paired seed × Y × scenario result so alternative summaries can be recomputed without rerunning missions. Runtime and any errors are recorded in `run_status.csv`; concise progress is in `sweep.log`.
