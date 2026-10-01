# Frozen-stream statistical analysis

Status: implementation and synthetic verification complete. No held-out main NPZs have been analyzed by this task, and no main empirical performance claims follow from the synthetic tests. Only `scripts/statistical_analysis.py` and this note were edited.

## Entry points and expected input

Run `python scripts/statistical_analysis.py` after the main prediction files are complete and frozen. Defaults read `results/main/{site}_B{budget}_{method}.npz`, write CSVs to `results/tables`, and write the numeric ledger to `results/statistics_summary.json`. `--source`, `--output`, `--replicates`, `--sensitivity-replicates`, and `--seed` are available. Calling `analyze_directory(...)` also permits explicit required sites, budgets, and methods.

Expected arrays: `pred` with shape D by T (already clipped to zero), `y` with shape T, hourly `times`, Boolean `score_masks` D by T, additional-attempt `query` T, `initmask` T, and scalar `metadata_json` encoding a list of D records. Metadata needs site, physical instrument, and family; device/deployment/channel and start/end fields identify row order. Optional arrays are `warning_mask`, and `y_duplicate` with an explicit QC-valid Boolean mask named `duplicate_valid`, `y_duplicate_valid`, or `duplicate_qc_valid`. Conflicting duplicate masks are rejected. Finite duplicate values alone do not establish QC validity.

The main required acquisition methods are periodic, random1/2/3, risk, information_gain, uncertainty, risk_independent, risk_fixed, window_uncertainty, and noAR. Other model names are automatically loaded. Missing required files are recorded, not silently imputed. Per-site/instrument metrics remain available for incomplete method sets, but overall bootstrap comparisons only include methods present at all configured sites. An aggregate row states whether all required site inputs exist. Unconfigured site names require an explicit `required_sites` argument so a point estimate and its interval cannot accidentally use different sites.

The loader rejects:

- nonhourly or duplicated timestamps;
- mismatched reference arrays, row identities/order, score masks, or initialization calendars between methods at the same site/budget;
- nonfinite or negative predictions on the fixed scoring masks;
- nonfinite primary references included in a scoring mask;
- additional paid requests overlapping initialization access;
- inconsistent instrument families or ambiguous duplicate row identities.

Core acquisition policies must have equal additional attempted reference hours at the same site/budget. noAR, hgb_stream, ewrls, gp_stream, gp_residual, and ridge_stream, when supplied, must use the exact periodic query mask for engine comparisons. Other names are not assumed to share that schedule; their access counts are still reported. Finite-reference hours are reported separately from attempted access hours and are not used to equalize budgets after observing labels.

## Primary estimand

Let C(i,t) be the valid, scored channels/deployment rows for physical instrument i at hour t. For method m, first form the instrument-hour loss

    L2(m,i,t) = mean over c in C(i,t) of [prediction(m,c,t) - reference(t)]².

The corresponding absolute and signed losses use the same channel mean. For each instrument, average its hourly losses over its scored clock hours, then take the square root for RMSE. The primary overall metric is the equally weighted mean of physical-instrument RMSEs. It is not the square root of a pooled fleet MSE and is not the mean of channel/deployment RMSEs.

Concurrent IMB channels contribute total weight one per instrument-hour. Nonoverlapping replacement segments are joined under the same physical instrument identity. A physical instrument moving between sites is similarly pooled across its nonoverlapping scored hours before its overall RMSE is computed; overlapping scored times under one identity at distinct sites are rejected. No nominal sample size of 18 is hard-coded; observed instrument counts are reported.

Secondary site-balanced RMSE first averages instrument RMSE within each site, then gives each observed site equal weight. Site-specific instruments and physical instruments pooled across sites have separate tables. Family summaries use the pooled physical-instrument metric, with equal instrument weights. A pooled site's long-lived instrument does not receive extra weight in the macro metric solely because it has more hours.

Instrument MAE and bias use the same instrument-hour weighting. For p90 absolute error, each channel absolute error has weight 1 / number of active channels for that instrument-hour. The p90 is the inverse weighted empirical CDF without interpolation. Thus every instrument-hour contributes total weight one, while p90 remains a distribution of channel absolute errors in ppb; it is not the percentile of channel-averaged absolute errors. The definition is deliberately explicit because those alternatives answer different questions.

Daily and monthly tables report each site's physical-instrument RMSE, MAE, bias, and scored hours. These are diagnostics; short periods with fewer hours are not independent experimental replicates. Site and family tables include their instrument and hour counts.

## Paired temporal uncertainty

The default is 1,000 replicates of a paired circular calendar-block bootstrap with 7-day blocks. Each site's hourly calendar is stratified by calendar quarter. Within a quarter, block starts are sampled uniformly, blocks wrap within that quarter, and the assembled draw is trimmed to the original number of quarter hours. Missing/unscored calendar hours remain in the timeline. This preserves observed seasonal quarter lengths and helps retain short deployments rather than treating all years as exchangeable.

One timestamp multiplicity vector is used simultaneously for every method, channel, and instrument at a site. Sites receive independent draws. The same replicate combines the resulting per-site loss sums and observed-hour counts, reconstructs each instrument RMSE, and calculates both macro estimands. Methods are never resampled independently. The bootstrap samples loss streams, not already-computed daily RMSEs.

A replicate with zero sampled scored hours for any observed site-instrument is rejected jointly for every method. Requested, attempted, accepted, and rejected counts are recorded. At most 20 times the requested number of draws are attempted; fewer accepted replicates are explicitly marked incomplete. No missing instrument RMSE is replaced by zero. This rejection conditions on inclusion of the observed instruments, which is appropriate to the stated conditional estimand but should be disclosed, especially if rejection is frequent. A quarter shorter than the requested block length is treated as one circular block; such a stratum has no count variability when sampled in full.

The intervals are 2.5th and 97.5th percentiles of paired RMSE differences. The point estimate is computed from the original observed stream, not the mean bootstrap difference. Negative `method minus comparator` favors the named method. Relative differences are descriptive percentages relative to the comparator point estimate; no percentage confidence interval is claimed. No formal p-values are produced.

The base seed is 20260930. Each budget/block configuration uses `base_seed + 10000 * budget + block_days`, which is recorded in the ledger. Sensitivity analyses use 14- and 28-day blocks with 300 draws each. These assess interval sensitivity to temporal dependence assumptions, not alternate opportunities to choose significance. Missing methods can change the dimensional output but do not change the random-number draws when sites/timelines are otherwise identical.

These intervals condition on the observed sites, physical instruments, scoring masks, quarter lengths, and realized prediction streams. Models/controllers are not retrained or rerun on bootstrap histories. Accordingly, the intervals do not capture model-selection uncertainty, adaptive policy learning under alternative histories, or uncertainty over new geographic sites. Independent site draws also do not preserve common calendar shocks across cities. With only three observed sites, they must not be described as population-level geographic confidence intervals. Random1/2/3 remain separate realized schedules; temporal resampling does not replace analysis of schedule-to-schedule variation.

## Comparisons and optional validation

Every complete method is contrasted with periodic. When present, additional reversed comparisons report periodic dynamic AR minus hgb_stream, ewrls, gp_stream, gp_residual, ridge_stream, hgb_initial, and ridge_initial. This makes the engine comparison orientation explicit. noAR minus periodic is the prespecified AR ablation when their reference masks match. Claims about engine quality and acquisition scheduling must remain distinct.

Duplicate-reference validation reuses frozen predictions and restricts the original scoring masks to QC-valid duplicate-reference hours. It does not change fitting, acquisition, or hyperparameter selection. The separate `reference=duplicate` rows provide instrument/site/family and temporal metrics; bootstrap intervals are currently for the primary reference only. This is an independent measurement-instrument check, not an independent set of atmospheric hours or evidence that the duplicate itself is error-free.

When warning flags are present, `reference=no_warning` rows repeat metrics after excluding warned sensor observations. Despite the table label's common column name, the reference concentration is unchanged; this is a sensor-QC sensitivity subset. Primary metrics retain all observations in the predefined primary scoring masks. Restricting a subset is not allowed to alter the main metric or select the favored method.

## Outputs

- instrument_metrics.csv and site_instrument_metrics.csv;
- aggregate_metrics.csv, site_metrics.csv, and family_metrics.csv;
- daily_metrics.csv and monthly_metrics.csv;
- reference_access.csv;
- paired_rmse_differences.csv and bootstrap_metric_intervals.csv;
- statistics_summary.json with missing-input report, access checks, point estimates, differences/intervals, seeds, and achieved replicate counts.

The summary includes the prediction-file manifest paths. It does not create a second prediction version or modify any model/data file. The main run's separate manifest should carry content hashes and the frozen configuration.

## Verification performed

All tests below used explicitly synthetic data or in-memory serialized synthetic NPZs. No study prediction outputs were generated by these tests.

1. Two simultaneous channels, a replacement segment, and a separate physical instrument produced the analytically expected mean instrument RMSE. The concurrent channels did not double the first instrument's hourly weight.
2. A method identical to periodic had exactly zero paired bootstrap differences; a method with all errors multiplied by 0.5 had exactly half every replicate's RMSE.
3. Repeating a seed reproduced bootstrap arrays exactly; each site's quarter sample counts equaled its observed quarter lengths.
4. A synthetic very short instrument yielded 120 accepted draws from 176 attempts, with 56 explicit joint rejections, and preserved the analytic paired result.
5. Nonfinite scored predictions, different score masks, and different primary references were rejected. Duplicate-reference results included only the explicitly QC-valid hours.
6. The NPZ schema round-tripped in memory; the top-level analysis path and three block settings ran with output methods mocked, and correctly reported an omitted required method.
7. One physical instrument observed sequentially at two sites remained one instrument in the pooled primary metric.
8. A performance check with synthetic one-year hourly streams at three sites, 18 instruments, 12 methods, and 1,000 replicates completed the bootstrap in approximately 0.64 seconds on this machine with BLAS threads limited to two. This benchmarks computation only and says nothing about empirical study performance.

The timestamp check was corrected during synthetic verification to compare timedeltas directly, avoiding pandas datetime-unit assumptions. Main-run validation and actual numerical results remain pending.
