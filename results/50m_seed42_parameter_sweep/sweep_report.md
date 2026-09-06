# Physical 50 m radiation sensing: seed-42 parameter sweep

## Outcome

The fixed 3 × 3 sweep completed with the statuses below. The candidate recommendation and interpretation are finalised after inspecting all nine fixed configurations; no tenth configuration, extra seed, source-strength sweep, or planner redesign was run.

| Test | Y | sigma (m) | radius (m) | status | runtime (s) |
| --- | --- | --- | --- | --- | --- |
| T01 | 10.0 | 15.0 | 45.0 | SUCCESS | 50.798755 |
| T02 | 10.0 | 40.0 | 120.0 | SUCCESS | 63.073191 |
| T03 | 10.0 | 50.0 | 150.0 | SUCCESS | 66.441134 |
| T04 | 0.2 | 15.0 | 45.0 | SUCCESS | 52.707865 |
| T05 | 0.2 | 40.0 | 120.0 | SUCCESS | 63.107269 |
| T06 | 0.2 | 50.0 | 150.0 | SUCCESS | 72.832300 |
| T07 | 0.004 | 15.0 | 45.0 | SUCCESS | 70.697967 |
| T08 | 0.004 | 40.0 | 120.0 | SUCCESS | 76.525655 |
| T09 | 0.004 | 50.0 | 150.0 | SUCCESS | 91.742244 |

## Code and data-flow audit

The formal workflow is `examples/radiation/09_final_experiment.py`. Before this change its UAV flew at 50 m but `NoiseFreeRadiationSensor` was configured with `value_reference_height_m=1.0`, so the sensor queried perfect 1 m truth, subtracted the unchanged background, and passed ground-equivalent excess to the estimator. The planner itself did not accept truth: it consumed only the incremental estimator. The post-run evaluator was already split correctly, with UAV exposure queried at 50 m and survivor exposure at 1 m.

The new formal flow is:

`same hidden source → independent truth @ 1 m → survivor evaluation`

`same hidden source → independent truth @ 50 m → UAV sensor → airborne estimator → trigger/mode → planner`

`truth @ 50 m → UAV exposure evaluation`

The sensor's correction option was removed, and the measurement contract now rejects a value-reference height that differs from platform altitude. Point and Multiple Point use their unchanged analytic 3-D softened inverse-square model at explicit z=1 and z=50; source positions, per-source strength, source count, core radius, background and relative offsets are unchanged. Surface uses one unchanged activity-density/cell-activity raster, constructs separate physical kernels with `R=sqrt(dx²+dy²+h²)` at h=1 and h=50, and performs separate linear `fftconvolve(..., mode="same")` forward simulations. Cell area, FFT shape/padding, attenuation, gamma yield, kerma conversion, units and lack of normalisation remain unchanged.

No planner-facing truth parameter exists. The disabled legacy `sarenv.core.radiation` master-heatmap API remains non-operational and is not used. The sweep creates fresh truth/estimator objects in temporary directories and does not load radiation caches or old estimator arrays.

Changed implementation files are `sarenv/radiation/online/sensor.py`, `sarenv/radiation/online/measurement.py`, `sarenv/evaluation/evaluator.py`, `examples/radiation/09_final_experiment.py`, the older active example `06_full_radiation_aware_simulation.py`, its evaluator entry point, related tests, and this minimal sweep wrapper.

## Parameter semantics

Y is the code's `base_hazard_reference`. Measurements are source excess after subtracting background, so Y is **not** compared with background-inclusive total rate. The trigger starts at `0.01 × Y`, increases by the same amount after each completed radiation episode, caps at Y, and uses `exit = 0.9 × current_enter`. Y is also the denominator of the planner's radiation-equivalent score; because it is common to all candidates in one decision, it does not change their radiation ordering directly, while its trigger effect changes when radiation mode is active.

`sigma` is the Gaussian weighting scale used when assimilating each scalar measurement. `influence_radius` is the circular spatial update radius around that measurement; it is not a distance-based sampling frequency. Sampling remains exactly one noise-free reading per executed route node.

## 1 m versus 50 m truth

Peaks below are sampled on the single saved comparison figure's 5 m display grid. They are diagnostic total-rate peaks, not extra experiment outputs.

| Scenario | 1 m peak | 50 m peak | 50 m / 1 m |
| --- | --- | --- | --- |
| Single | 1818.38 | 10.1493 | 0.5581% |
| Multi | 1818.61 | 10.3792 | 0.5707% |
| Surface | 1.81898 | 0.176715 | 9.715% |

The 50 m point response has a lower peak and broader lateral footprint; the three fixed point responses overlap more strongly. The 50 m Surface result is smoother because the same activity raster is convolved with the broader 50 m kernel, not because a 1 m map was rescaled.

## Validation and smoke test

Directly related radiation-model, sensor, estimator, trigger/planner, evaluator and formal-workflow tests: **PASSED** — `106 passed, 2 warnings in 6.56s`. Failed: 0; skipped counts are reported by pytest in that same summary when present.

The required Y=0.2, sigma=40 m, radius=120 m, seed=42 integration smoke test was **PASSED**. It completed all Single/Multi/Surface missions, queried both height planes, verified every UAV measurement row was 50 m, verified UAV evaluation at 50 m and survivor evaluation at 1 m, and extracted the final summaries. It was discarded and is not a tenth sweep result.

## Compact cross-configuration view

| Test | Y | sigma/radius | episodes S/M/Surf | mean SAR Δ vs Original | mean survivor-total Δ vs Original |
| --- | --- | --- | --- | --- | --- |
| T01 | 10.0 | 15.0/45.0 | 18/1/1 | -6.176% | -31.339% |
| T02 | 10.0 | 40.0/120.0 | 15/1/1 | -0.299% | -25.053% |
| T03 | 10.0 | 50.0/150.0 | 7/1/1 | 0.331% | -26.454% |
| T04 | 0.2 | 15.0/45.0 | 1/1/18 | -6.291% | 42.142% |
| T05 | 0.2 | 40.0/120.0 | 1/1/15 | 1.350% | 0.828% |
| T06 | 0.2 | 50.0/150.0 | 1/1/11 | -0.885% | -13.108% |
| T07 | 0.004 | 15.0/45.0 | 1/1/58 | -9.250% | 41.044% |
| T08 | 0.004 | 40.0/120.0 | 1/1/58 | 0.503% | -31.734% |
| T09 | 0.004 | 50.0/150.0 | 1/1/59 | -0.222% | -12.543% |

Full old/new values, absolute deltas, relative deltas and the shared Original values for every core metric and scenario are in `sweep_summary.csv`.

## Y effect

Y=10 is not uniformly “too hard” for the point scenarios: depending on the spatial level, Single entered 7–18 episodes and Multi entered one mission-long episode. It is too hard for the 50 m Surface morphology after the first short episode: the 50 m sampled peak is about 0.177 uGy/h, while completing the first Y=10 episode raises the next entry threshold from 0.1 to 0.2. Consequently T01–T03 gave Surface only 5–6 radiation steps and never re-entered.

Y=0.2 is the operational middle setting. Single and Multi each entered one mission-long radiation episode, while Surface produced 11–18 finite episodes, so the trigger remained responsive without the strongest mode chatter.

Y=0.004 was over-sensitive for Surface: T07–T09 produced 58–59 episodes and 1,023–1,659 radiation-mode steps for S2/S3. For a fixed spatial level, Y=0.2 and Y=0.004 produced identical Single and Multi routes because both entered once and stayed active for all 3,600 steps; lowering Y further therefore added no point-scenario benefit in this seed. It did materially change Surface routing.

## Sigma / influence-radius effect

The legacy 15/45 m support is too local for the broadened 50 m measurements. S1 produced the weakest Multi SAR outcome at every Y (about -15% likelihood versus Original) and was also poor on Surface at low Y. The 40/120 m theoretical baseline was the most stable spatial level: T08 kept SAR likelihood within -1.32% to +3.79% of Original across all morphologies and reduced both total and P95 survivor dose in all three. The 50/150 m setting helped some point outcomes but was less stable on Surface; T06 and T09 lost 5.57% and 3.57% SAR likelihood respectively, and their Surface survivor maximum dose rose 31.97% and 35.45% versus Original. This is consistent with the broad setting diffusing measurements too far for this morphology.

## Cross-scenario stability and recommendation

**Candidate for the later repeated experiment: T08 — Y=0.004, sigma=40 m, influence_radius=120 m.** It is the strongest seed-42 cross-morphology result: survivor counts were 96/98/97 versus Original's 93/93/93; SAR likelihood changes were -0.96%/+3.79%/-1.32%; survivor total-dose changes were -24.65%/-42.01%/-28.54%; and survivor P95 changes were -34.79%/-41.39%/-81.47% for Single/Multi/Surface. No other configuration combined near-neutral SAR with total- and tail-dose reductions in all three morphologies.

This is a candidate, not a claim that Y=0.004 is already validated. Its 58 Surface episodes are clear evidence of over-sensitivity and possible mode chatter. T05 (Y=0.2 with the same S2 spatial level) is the calmer engineering alternative at 15 Surface episodes, but on this seed its Surface survivor total dose rose 69.14% and maximum dose rose 106.60% versus Original. Those cross-scenario failures make T05 less stable than T08 despite its more moderate trigger behaviour. Per the fixed scope, no extra configuration or seed was run to resolve that trade-off.

## Old seed-42 comparison

The unique prior formal result is `results/final_experiment`, created 2026-08-25 with seed=42, complete Original/Single/Multi/Surface outputs, Y=10, sigma=15 m, radius=45 m, and explicit `perfect_ground_equivalent_scalar` sensing metadata. It was not overwritten.

The old UAV **sensing** value was ideal 1 m ground-equivalent excess. Its post-run UAV cumulative dose was, however, already evaluated against 50 m truth. Therefore old/new UAV exposure values use the same evaluation altitude and are numerically comparable as route outcomes; their routes differ because the sensing/planning contract changed. Survivor dose remains 1 m in both old and new results.

The table below gives the complete old/new/absolute/relative comparison for the recommended T08 on the principal metrics; `sweep_summary.csv` contains the same four-way comparison for every metric and all T01–T09 rows.

| Scenario | Metric | Unit | Old | T08 new | Absolute delta | Relative delta |
| --- | --- | --- | --- | --- | --- | --- |
| Single | SAR likelihood | probability | 0.90127183 | 0.92675907 | +0.025487239 | +2.828% |
| Single | Time-discounted likelihood | score | 0.1172884 | 0.11469606 | -0.0025923403 | -2.210% |
| Single | Survivors found | count | 88 | 96 | +8 | +9.091% |
| Single | Route distance | m | 87441.882 | 86530.612 | -911.26984 | -1.042% |
| Single | Survivor total pre-discovery dose | uSv | 28.96377 | 21.178609 | -7.785161 | -26.879% |
| Single | Survivor P95 pre-discovery dose | uSv | 0.73659688 | 0.579925 | -0.15667188 | -21.270% |
| Single | UAV cumulative total dose | uSv | 1.2488497 | 1.1012424 | -0.14760735 | -11.819% |
| Multi | SAR likelihood | probability | 0.95262684 | 0.97120574 | +0.018578902 | +1.950% |
| Multi | Time-discounted likelihood | score | 0.11708958 | 0.1121727 | -0.0049168767 | -4.199% |
| Multi | Survivors found | count | 100 | 98 | -2 | -2.000% |
| Multi | Route distance | m | 87176.785 | 86630.023 | -546.7619 | -0.627% |
| Multi | Survivor total pre-discovery dose | uSv | 50.465234 | 59.590198 | +9.1249643 | +18.082% |
| Multi | Survivor P95 pre-discovery dose | uSv | 0.9612068 | 1.6041779 | +0.64297109 | +66.892% |
| Multi | UAV cumulative total dose | uSv | 2.7376234 | 3.2559337 | +0.51831032 | +18.933% |
| Surface | SAR likelihood | probability | 0.95161524 | 0.92340576 | -0.028209483 | -2.964% |
| Surface | Time-discounted likelihood | score | 0.12623603 | 0.08239084 | -0.043845193 | -34.733% |
| Surface | Survivors found | count | 94 | 97 | +3 | +3.191% |
| Surface | Route distance | m | 87748.4 | 87284.48 | -463.91919 | -0.529% |
| Surface | Survivor total pre-discovery dose | uGy | 0.87833916 | 1.0654524 | +0.1871132 | +21.303% |
| Surface | Survivor P95 pre-discovery dose | uGy | 0.011545486 | 0.00089937481 | -0.010646112 | -92.210% |
| Surface | UAV cumulative total dose | uGy | 0.024527526 | 0.013983367 | -0.010544158 | -42.989% |

Relative to Original, T08 preserves the main radiation-aware conclusion in a qualified form: all three morphologies found more survivors and reduced total, P95 and maximum survivor pre-discovery dose, while SAR likelihood stayed close to Original. It does **not** preserve a time-efficiency claim: time-discounted likelihood fell 17.93%, 19.74% and 41.05% versus Original for Single/Multi/Surface. Relative to the old ideal-sensing radiation-aware result, T08 improved Single but traded outcomes in Multi and Surface; in particular Multi/Surface total dose was 18.08%/21.30% higher than the old ideal-sensing routes even though both remained lower than Original. The old planner result therefore cannot be carried over unchanged; physical 50 m sensing makes parameter choice materially important.

## Output control

The Original route was generated once for the nine formal tests and kept only in a temporary directory. Original radiation evaluations were likewise computed once per source morphology and reused. Each parameter group retained one concise log; per-run routes, truth arrays, estimator arrays, plots, detailed CSVs and JSON summaries were deleted with the temporary workspace after their metrics were appended atomically to the master CSV.

Permanent output files: **14**. Sweep directory size: **443185 bytes (0.423 MiB)**. The target of fewer than 20 files is satisfied.
