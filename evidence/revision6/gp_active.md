# Revision 6: enhanced-computation GP acquisition comparator

Completed 2026-10-01. All three original evaluation sites and all eighteen physical instruments were retained. The chronological correction replay contains 23 deployment/channel streams (Manchester 21, London 1, York 1); aggregation remains physical-instrument equal. This is a retrospective, non-selected computational-budget sensitivity, not a new independent confirmation or an optimized GP upper bound. No frozen script, previous prediction or previous result was modified.

## Question and controlled change

Does the previous sparse GP planning approximation conceal materially better calibration when its planning computation is increased?

| Planning component | Previous GP comparator | Enhanced sensitivity |
|---|---:|---:|
| Candidate cadence | 12 h | 6 h |
| Future evaluation offsets | 9 offsets: 6,12,24,48,72,96,120,144,168 h | 28 offsets: every 6 h through 168 h |
| Context scenarios | One path at lag 192 h | Three paths at lags 192,216,288 h |
| Planning horizon | 168 h | 168 h |
| Threshold multiplier | 1.25 | 1.25 |
| Kernel length scale / noise | 2.0 / 0.1 | 2.0 / 0.1 |
| Correction memory | 2160 h / most recent 512 pairs | Unchanged |
| Resource contract | B48 per 672-h block, four starts, nominal 12 h | Unchanged |

Each historical path must end strictly before the current decision. A missing historical input uses the current context. Before any complete path exists, one frozen-current-context scenario is used. Scenario covariance reductions are averaged over available paths. No future sensor features or unacquired reference concentrations enter acquisition. Initial commissioning is a known shared schedule and conditions both covariance paths after the corresponding current product prediction. The three changes are assessed together; this experiment does not attribute their effects separately.

No hyperparameter grid or test-driven choice was made. The enhanced setting was specified before this new run; both settings remain reported irrespective of direction. Both share the original frozen GP correction and source support. Only immediate delivery with shared commissioning is examined here.

## Empirical results

Primary support is identical across five methods: eighteen physical instruments, three sites, 62,652 instrument-hours. Instrument-equal RMSE and MAE are ppb.

| GP calendar | Instrument-equal RMSE | Instrument-equal MAE | Site-equal RMSE |
|---|---:|---:|---:|
| Original GP active | 4.344329 | 3.120208 | 6.114091 |
| Enhanced GP active | 4.345341 | 3.197004 | 6.586592 |
| AR fixed-risk transferred to the same GP | 4.282259 | 3.137214 | 6.098307 |
| AR variable-risk transferred to the same GP | 4.320825 | 3.189184 | 6.167841 |
| Periodic | 4.393575 | 3.219291 | 6.206613 |

The enhanced-minus-comparator paired differences and conditional 95% percentile intervals use the existing synchronized site-quarter circular seven-day block bootstrap, 1,000 replicates, seed 20261008. They condition on these observed streams, sites, fitted corrections and settings; they do not include retraining or policy-selection uncertainty. These exploratory sensitivity intervals carry no confirmatory p-value or multiple-comparison claim.

| Comparator | Instrument-equal RMSE difference | Conditional 95% interval | Site-equal RMSE difference | Conditional 95% interval |
|---|---:|---:|---:|---:|
| Periodic | −0.048234 | [−0.148649, 0.049607] | 0.379979 | [0.129628, 0.629594] |
| Original GP active | 0.001012 | [−0.088103, 0.090729] | 0.472501 | [0.251489, 0.717474] |
| AR fixed-risk transferred GP | 0.063082 | [−0.042582, 0.162089] | 0.488285 | [0.222842, 0.763730] |

The near-zero instrument-equal difference conceals site differences. Manchester improves (3.897091 versus 3.990376 ppb), while London worsens (4.641872 versus 4.181457 ppb) and York worsens (11.220812 versus 10.170440 ppb). Manchester supplies sixteen of eighteen instruments, while London and York each supply one; the higher site-equal RMSE therefore matters and must remain visible. Site comparisons describe observed instrument/site combinations, not separable city effects.

## Decisions and computation

Paid-hour overlap excludes common commissioning. Jaccard = shared paid hours / union of paid hours. Reference budgets are exactly equal at each site: Manchester 807, London 450, York 912 attempted hours. All block quotas are fulfilled and at most four planned starts occur in each block.

| Site | Enhanced vs original GP Jaccard | Enhanced vs AR fixed-risk Jaccard | Enhanced vs AR variable-risk Jaccard |
|---|---:|---:|---:|
| Manchester | 0.132632 | 0.060447 | 0.038610 |
| London | 0.200000 | 0.070155 | 0.041667 |
| York | 0.101449 | 0.108815 | 0.024144 |

Pooled across the three disjoint site calendars, Jaccard is 0.132341 against original GP, 0.082335 against fixed-risk AR and 0.033103 against variable-risk AR. These are calendar diagnostics, not predictive accuracy measures.

Enhanced/previous planned starts are 68/64 (Manchester), 36/38 (London) and 74/74 (York). Fractions of enhanced starts within twelve hours of any original-GP start are 0.397059, 0.527778 and 0.391892, respectively. Median distance from an enhanced start to the nearest original-GP start is 24, 12 and 24 hours. Against fixed-risk AR, the corresponding median distances are 37, 50 and 35 hours; against variable-risk AR, 45, 59 and 43 hours. Stored start masks from the actual AR source calendars are used and their paid masks are asserted identical to the transferred-GP runs. Periodic start diagnostics instead use starts of contiguous paid runs, as disclosed in the CSV.

| Site | Original total wall seconds | Enhanced total wall seconds | Original acquisition-score seconds | Enhanced acquisition-score seconds |
|---|---:|---:|---:|---:|
| Manchester | 68.658204 | 77.180431 | 1.755126 | 9.500343 |
| London | 1.870537 | 2.220540 | 0.125333 | 0.632084 |
| York | 4.024112 | 5.349207 | 0.277474 | 1.592019 |

Total wall time is 74.552853 versus 84.750178 s; covariance-acquisition scoring is 2.157933 versus 11.724447 s. Enhanced scoring evaluates 6,839 channel-level candidates and 20,517 historical scenarios versus 4,009 candidates/scenarios originally. Timings come from separate executions on the same host and are descriptive, not an isolated hardware benchmark. Most wall time comes from correction refitting.

Interpretation: Increasing planning computation substantially changes the access calendar, but does not improve the instrument-equal RMSE in this sensitivity and worsens the site-equal result. This resolves the specific concern that the original sparse comparator was the only tested GP planning budget. It does not prove that the original setting is optimal, that other GP interfaces cannot improve performance, or that GP acquisition in general is inferior to AR planning. Posterior/context approximations, horizon choices and delayed delivery remain limitations of this comparator.

## Implementation verification

Five synthetic checks pass: hidden-label value/missingness invariance; current acquired target cannot change the current finalized prediction or request; future-feature invariance; prediction parity with the independent frozen GP correction; and exact budgets/start caps/commissioning exclusion. These are implementation tests, not empirical sensor evidence.

All 23 real deployment/channel streams were also replayed independently with the enhanced fixed access masks through the existing frozen GP correction. Maximum absolute prediction discrepancies are 2.49223e−12 ppb (Manchester), 5.18696e−13 ppb (London) and 5.18696e−13 ppb (York), within the specified relative/absolute 1e−12 numerical tolerance. All scored outputs are finite; all source arrays and support masks are asserted unchanged. Exact attempted budgets pass at all sites. See `verification.json` and `smoke_checks.json`.

## Main-text wording

> A computational-budget sensitivity doubled GP decision frequency, expanded the quadrature from nine to 28 offsets and averaged three strictly historical contexts. It changed paid-hour calendars substantially (Jaccard 0.10–0.20 against the original GP schedule) but left instrument-equal RMSE almost unchanged: the difference was 0.001 ppb (conditional 95% interval −0.088 to 0.091). Site-equal RMSE increased by 0.473 ppb (0.251 to 0.717), with deterioration at London and York. These findings apply to the tested posterior interface and approximations; they do not establish that more fully optimized GP acquisition cannot differ.

## Reproduction and outputs

- Runner: `scripts/revision6_gp_active.py`; actions `plan`, `smoke`, `real-smoke`, `run`, `summarize`, `verify`.
- Read-only intervals runner: `scripts/revision6_gp_intervals.py`; no arguments.
- Frozen inputs: `results/main/*_B48_periodic.npz`, original GP active streams, same-parameter shared-selected GP streams, cached public QUANT data.
- New outputs: `results/revision6/gp_active/plan.json`, three `streams/*_B48_gp_enhanced.npz` plus timing/provenance JSONs, aggregate/instrument/site/month/day metrics, `calendar_overlap.csv`, `runtime.csv`, `source_index.json`, `paired_contrasts.csv`, `metric_intervals.csv`, `bootstrap_diagnostics.json`, `smoke_checks.json`, `verification.json`, `real_smoke_Manchester.json`.
- No external upload or DOI action performed by this subtask.
