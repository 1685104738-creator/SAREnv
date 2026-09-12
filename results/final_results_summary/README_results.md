# Final Results Summary

## Data basis and checks

The statistical figures and Table 1 use the completed `intensity_summary.csv`: 20 independent unseen seeds at 1× and paired sensitivity comparisons at 0.25×, 0.5×, 2× and 4×. Each seed retains the same radiation geometry across intensity levels. This is 20 independent worlds, not 100 independent worlds. All 100 seed × intensity runs succeeded and all 300 morphology rows were present, with no duplicate keys or missing required metrics.

The analysis independently confirmed the frozen validation parameters: Y = 0.05, sigma = 40 m and influence radius = 120 m. The rounded means and improvement counts reconcile with `intensity_report.md`. That report's abbreviated “UAV mean Δ%” values correspond to the CSV's cumulative UAV **excess-dose** percentage change; Table 1 uses the explicit label “UAV excess dose Δ”.

## Figure 1 — Representative route comparison

`figure_01_route_comparison` shows Original and Radiation-Aware routes for Single-point, Multi-point and Surface radiation in the completed representative experiment. The map extent, survivors, start and radiation truth are held constant within each row. Radiation information visibly changes the executed route in all three morphologies.

This figure supports a qualitative behavioural comparison only. It is not a significance test and the representative run predates the 20-seed 50 m validation configuration.

## Figure 2 — Independent 1× validation distributions

`figure_02_validation_distribution` shows every unseen seed, boxplots and means for survivor Total, P95 and Maximum exposure changes. Negative values indicate improvement; no outlier is removed.

- Single Total: mean -25.85%, median -27.11%, SD 24.80%, range [-70.95%, +22.68%], 18/20 improved.
- Surface Total: mean -57.47%, median -79.65%, SD 70.27%, range [-96.64%, +214.87%], 18/20 improved.
- P95 improved in 19/20 Single, 17/20 Multi and 19/20 Surface seeds. Single: mean -38.85%, median -39.59%, SD 16.18%, range [-67.68%, +0.90%], 19/20 improved. Multi: mean -19.44%, median -26.34%, SD 62.27%, range [-72.30%, +225.86%], 17/20 improved. Surface: mean -55.18%, median -60.36%, SD 30.47%, range [-94.34%, +8.42%], 19/20 improved.
- Multi Maximum has severe tail instability: mean +55.63%, median -27.18%, SD 182.93%, range [-96.76%, +508.19%], 12/20 improved. Its mean is adverse despite a beneficial median, with a largest observed change of +508.19%.

The figure establishes distributional stability and tail risk. It does not identify the causal path mechanism behind individual outliers.

## Figure 3 — Intensity sensitivity

`figure_03_intensity_sensitivity` plots mean survivor Total and P95 changes with 95% t confidence intervals across the same 20 paired seeds. Single Total and P95 reductions strengthen with intensity. Surface shows a large benefit from 0.25× onward, although its mean is not monotonic at every level. Multi P95 remains beneficial on average at every intensity, while Multi Total remains close to neutral or adverse until 4×.

The paired curves support intensity robustness with strong morphology dependence. They do not represent 100 independent spatial worlds.

## Figure 4 — Improvement consistency

`figure_04_improvement_consistency` reports improved seeds out of 20 for Total, P95 and Maximum exposure. The outlined 1× column is the independent validation. Single reaches 20/20 for Total and P95 at 2× and 4×. Multi has consistently lower Total and Maximum improvement counts, while its P95 counts remain 17–18/20. Surface is usually beneficial but retains a small number of adverse seeds.

Counts describe direction, not the size or practical importance of a change.

## Figure 5 — SAR–radiation trade-off

`figure_05_sar_radiation_tradeoff` places each morphology × intensity mean according to time-discounted SAR change and survivor P95 exposure change. Surface combines the largest radiation benefit with the smallest average time-discounted SAR cost (about 7%). Single provides clear radiation benefit with an approximately 16–17% time-discounted SAR cost. Multi has the largest SAR cost (about 25%) and a smaller mean P95 benefit (about 16–21%).

This is an empirical trade-off summary. It does not assign a preferred operating point or causal weight to either objective.

## Figure 6 — Radiation-mode behaviour

`figure_06_radiation_mode_behaviour` shows mean radiation-mode steps and episode counts. Single and Multi enter one episode lasting all 3,600 recorded radiation-mode steps at every tested intensity, including 0.25×. Surface instead shows repeated local entry and exit: mean steps rise from 410.15 at 0.25× to 773.25 at 4×, while mean episodes rise from 17.50 to 28.65.

The data directly support Point/Multi mode saturation and local Surface engagement. A plausible limitation is that low-intensity Point/Multi cases do not recover Original-like search behaviour. The retained sensitivity output does not include route-level event logs, so it cannot prove that any specific movement mechanism caused the Multi outliers.

## Figure 7 — Representative online radiation estimate

`figure_07_representative_estimated_radiation_map` shows the saved estimated field, executed route and a subset of measurement locations for the representative Single-point experiment. The plot reads the NPZ's frozen legacy metadata: a 1 m-resolution, ground-equivalent excess gamma dose-rate estimate referenced to 1 m and expressed in uSv/h. It is not reinterpreted under the newer 50 m contract.

The figure demonstrates that route measurements populated an online estimate used by the system. It is not a quantitative map-reconstruction validation: no ground-truth comparison, RMSE, MAE, correlation or accuracy score was calculated.

## Table 1 — Final quantitative summary

`table_01_final_results` gives one row for each of the 15 morphology × intensity combinations. The bold 1× rows are the independent final validation. Values are seed-level paired means; distributional detail remains in Figures 2–4.

## Key findings

1. Single and Surface survivor-exposure reductions were stable in the independent 1× validation, with Total/P95 improvements in 18/20 and 19/20 seeds for both morphologies.
2. P95 was the most directionally consistent radiation-benefit metric across morphologies and intensities.
3. Multi retained a typical P95 benefit but showed severe worst-case tail instability. At 1× its Maximum mean was +55.63% while its median was -27.18%.
4. Survivor-risk reduction generally strengthened with intensity, especially for Single, while SAR penalties did not worsen monotonically with intensity. Morphology was the stronger separator of SAR cost.
5. Radiation-Aware search produced a clear SAR–radiation trade-off: lower survivor pre-discovery exposure commonly coincided with lower time-discounted SAR performance and, in some conditions, fewer survivors found.
6. UAV excess dose did not decrease consistently. The supported claim is survivor-risk prioritisation, not universal radiation avoidance or guaranteed worst-case reduction.
