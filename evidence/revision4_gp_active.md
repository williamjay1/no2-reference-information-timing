# Fourth-review independent GP reference-window comparator

Status: **independent GP-active three-site evaluation, summary and original-GP prediction verification completed; PASS**. The subsequently authorized fixed-calendar 72-hour GP extension is also completed and verified separately; it is not part of the GP-active policy.

## Route and purpose

SCI applied measurement/evaluation for IEEE Sensors Journal. The contribution tested here is whether a reference-window policy driven by the correction GP's own posterior can improve commercial-report correction under the same bounded information contract. Primary standards are permitted-information order, equal attempted access, identical correction settings/support, and truthful uncertainty. No causal-identification standard, physical electrode identification, field deployment or novel-GP/ALC theory is claimed. The applicable experiment-design, paper-verification and quantitative-analysis modules support this bounded revision rather than restart the full research workflow.

## Recorded design

The machine-readable protocol is `results/revision4/gp_active/plan.json`, created before this experiment. This is a retrospective review extension after the earlier evaluation results were known, not preregistration or a new independent confirmation dataset. The new comparator's settings will not be selected on the evaluation period.

| Policy | Correction parameters | Reference calendar | New computation |
|---|---|---|---|
| GP periodic | Existing early-Manchester shared-selected GP: RBF length 2, noise variance 0.1 | Original periodic B48 | Reuse immutable streams |
| GP AR fixed | Same GP | AR fixed-risk B48, transferred unchanged | Reuse immutable streams |
| GP AR variable | Same GP | AR variable-risk B48, transferred unchanged | Reuse immutable streams |
| GP active | Same GP | Independently generated GP variance-reduction calendar | New three-site replay |

Each comparison uses instantaneous delivery, shared commissioning, B48 additional attempted site-hours per 672-hour block, at most four starts, common preprocessing and the same scored support. The new GP policy commits nominal twelve-hour windows. Common final-start and catch-up enforcement may extend a nominal window, while attempts pause during external commissioning. A missing target or invalid input does not refund an attempted site-hour. Partial blocks have proportional integer feasible quotas. All commissioning access is charged in the same site-hour union as the existing experiments.

The GP uses the existing initial median imputation, missingness indicators, feature standardization and target mean/scale; transformations remain fixed. It refits after each usable acquired pair using the latest ninety calendar days and at most 512 pairs. Hyperparameters are fixed during replay. The selected length/noise were established from Manchester observations strictly before 2021 under shared commissioning; no new hyperparameter search is performed here.

## Actual GP objective and approximation

For a finite context set U and permitted training inputs X, latent covariance is C = k(U,U) − k(U,X)[k(X,X)+νI+αI]⁻¹k(X,U), using the fitted GP's RBF signal kernel, noise ν=0.1 and numerical diagonal regularization α=10⁻⁶. A hypothetical observation at context a changes C by the rank-one reduction C[:,a]C[a,:]/(C[a,a]+ν+α). Its unknown value is integrated out: covariance conditioning does not need a future target. Alpha is not independent physical measurement-noise evidence. This is implemented from the GP Cholesky posterior, not an AR state or saved AR calendar. The source reviewer checked the GP conditioning relation against Rasmussen and Williams, GPML §2.2 equations (2.22)–(2.24), and confirmed the signal/WhiteKernel distinction against sklearn source.

Two hypothetical paths begin with that same posterior. Known future external commissioning observations condition both paths after their current prediction. Additional attempted observations condition only the candidate path, with the paid prefix pausing during commissioning. Utility averages candidate-minus-baseline variance decrease before future updates, converts standardized variance to ppb squared using the fixed commissioning target scale, and aggregates ready channels using the original physical-instrument weights. It is latent variance reduction, not a calibrated probability of actual error reduction or sampling at the largest current predictive variance.

Computation is bounded in advance:

- Candidate starts are checked every twelve site-clock hours relative to the common block origin, while committed windows and mandatory catch-up execute hourly. The original AR planner considers candidates hourly; this cadence difference is explicitly part of the new policy and prevents interpreting their comparison as a pure covariance-family ablation.
- One complete historical context scenario begins at t−192 and ends at t−24 for H=168. Its initial context is the current input; missing historical inputs use the current context. If no complete path exists, all contexts use the frozen current input. The score never uses future actual inputs, availability or reference values.
- Future pre-update latent variance is integrated at offsets 6, 12, 24, 48, 72, 96, 120, 144 and 168 hours, with interval-length weights summing to one. All hypothetical candidate and known-commissioning observations needed for those evaluations are retained in sequential conditioning.
- The current fitted GP posterior is held as the planning baseline; the utility approximation does not simulate future ninety-day expiry or 512-pair eviction. Actual mean correction follows the original memory/refitting rules.
- Length-plus-quota normalization, historical 336-hour score quantiles, threshold scale 1.25 and budget completion follow the fixed-risk resource design. The new score is not an exact reproduction of the complete Cohn et al. pipeline, and no claim of a new acquisition principle is made.

## Information order and verification

At each hour, all ready instruments issue their predictions; the GP score and request action are then computed; only an authorized commissioning/requested target's value and finiteness can then be inspected. A usable pair updates the corresponding GP afterward. Initial fits of later deployments cannot contribute to earlier decisions because both prediction and utility require current readiness and valid current context.

Synthetic smoke passed with saved receipt `results/revision4/gp_active/smoke_checks.json`:

1. Changing hidden target values and missingness leaves the calendar and predictions unchanged.
2. Changing a current acquired target cannot alter that hour's estimate or decision, but can alter a later estimate.
3. Future-feature perturbations leave earlier actions and predictions unchanged.
4. Duplicating a physical instrument as identical half-weighted channels leaves its calendar unchanged.
5. Given an external mask, the new GP correction matches the original implementation using relative and absolute tolerances of 10⁻¹².
6. Latent covariance plus independent observation noise matches sklearn's predictive covariance within 10⁻¹⁰.
7. A constant-process analytical example verifies the sequential variance-reduction calculation including free shared access.
8. Feasible attempted budgets, four-start limits and commissioning exclusion hold.

Formal validation will separately regenerate every new stream's predictions using the unchanged original GP implementation on the generated calendar, check identical source/score/commissioning arrays and attempted counts, and check block/start constraints. This tests the correction independently from the new GP scheduling code; it does not certify real-time archive availability or field deployment.

That independent validation is now complete for all 23 channel/deployment segments in three sites. Largest absolute differences between single-hour and original block-vectorized GP predictions are 2.9132×10⁻¹² ppb in Manchester, 1.7053×10⁻¹³ in London and 3.4106×10⁻¹³ in York. All satisfy the declared relative/absolute 10⁻¹² checks. Source times, references, score masks, commissioning calendar and metadata remain exact; budgets, four-start caps and setup exclusion pass. Recorded maximum training age at refit is 2,160 event-time hours and maximum pair count 512. The receipts and checksums are in `verification.json` and `source_index.json`. These are numerical parity within stated tolerance, not a claim of bitwise equality of differently vectorized GP predictions.

## Resources and analysis

Only the owned script `scripts/revision4_gp_active.py`, the new `results/revision4/gp_active` directory and this evidence file are written. D has approximately 179 GiB free at the check; streams, logs and summary tables are expected to remain below 50 MB. Data are read from the existing D feature/reference caches; the F raw store is untouched. TEMP/TMP and NUMBA cache point to D, and BLAS/OpenMP use two threads. No GPU or API calls are needed. The existing same-parameter GP evaluation used about 110 seconds in aggregate fit/predict time. Early-Manchester real smoke covered the first 1,680 hours in 136.95 seconds, with 96 additional attempted hours and four starts in each of two feasible blocks. Its many known shared commissioning updates dominate fit time; all GP scores together took approximately 0.7 seconds. No smoke RMSE was used for policy selection. These measurements support a rough 2–4 minute formal-run estimate plus approximately two minutes for independent original-GP replay, with concurrent CPU work affecting wall time.

Primary comparisons are GP-active minus each of the three same-parameter GP policies. Physical-instrument RMSE and MAE pool simultaneous-channel losses within instrument-hour and nonoverlapping replacements within instrument, then give each physical instrument one vote. Seven- and fourteen-day synchronized circular site-quarter blocks each use 1,000 resamples. Intervals condition on these observed instruments/sites and realized streams, without retraining or policy-selection uncertainty. Per-device values, access counts, merged episodes, forced-hour fractions and wall time are retained. Neither lower posterior variance nor a positive result is assumed in advance; negative or heterogeneous outcomes will be reported.

Execution entry points are `--stage smoke`, `real-smoke`, `run`, `summarize`, and `verify`. The formal matrix contains three new site streams and nine immutable comparator streams. Source indices and checksums are saved with the summary. The original and revision freeze sources are not overwritten.

## Completed independent-policy results

Every cell contains the same eighteen physical instruments, 62,652 instrument-hours and 68,526 channel observations. Each requires 4,031 attempted reference site-hours: 1,862 commissioning plus 2,169 additional. The same-parameter comparator streams are the previously shared-selected GP outputs; all three GP-active streams are new.

| GP calendar | Instrument-macro RMSE (ppb) | Instrument-macro MAE (ppb) |
|---|---:|---:|
| Periodic | 4.393575 | 3.219291 |
| AR-transferred fixed risk | 4.282259 | 3.137214 |
| AR-transferred variable risk | 4.320825 | 3.189184 |
| Independent GP variance reduction | 4.344329 | 3.120208 |

| Active minus comparator | RMSE difference (ppb) | Seven-day paired interval | Fourteen-day paired interval |
|---|---:|---|---|
| Periodic | −0.049246 | [−0.150589, +0.052292] | [−0.175186, +0.053709] |
| AR-transferred fixed | +0.062069 | [−0.049075, +0.163181] | [−0.071282, +0.168206] |
| AR-transferred variable | +0.023504 | [−0.130228, +0.164568] | [−0.156216, +0.177275] |

Nine of eighteen instruments have lower active-policy RMSE than periodic GP; twelve have lower RMSE than GP on the transferred fixed calendar. These are descriptive counts. The extended-record active-minus-periodic contrasts are −0.432010 ppb for Zep188, −0.678656 for Zep344, −0.229145 for London Zep311 and −0.007830 for York Zep309. NS2 worsens by +0.630956 ppb. Individual values for every instrument and method remain in `instrument_metrics.csv`; the above selected examples do not replace the complete table or identify a physical cause.

GP-active has 180 merged commissioning-plus-paid episodes, versus 183 for periodic and 173 for both transferred calendars, at equal site-hour attempts. Mandatory/extended completion accounts for 195 of 2,169 additional hours (9.0%). Active and AR-fixed masks differ on 1,266 Manchester, 770 London and 1,654 York clock hours, confirming that the policy did not relabel a transferred AR calendar. Actual wall time is 68.66, 1.87 and 4.02 seconds by site, or 74.55 seconds total; fitted posterior scoring consumes approximately 2.16 seconds. Timing records are computational diagnostics from this run, not a controlled efficiency study or measured service savings.

The prespecified GP approximation provides a genuine learner-driven acquisition comparator, but its small point improvement over periodic GP is imprecise and it does not establish superiority over either transferred calendar. It cannot justify claiming that an added GP policy solves covariance ranking, that every GP variance selector fails, or that the original AR-timing result generalizes under delivery delay. No additional configuration was selected after inspecting these results.

## Main-text method and contribution facts (nine)

1. A separate GP posterior, with fixed early-Manchester-selected length 2/noise 0.1, chooses its own reference windows; it is not an AR calendar renamed as GP active learning.
2. Correction features, initialization transforms, ninety-day/512-pair memory, current-prediction ordering, B48 attempts, commissioning sharing and physical-instrument score support match the same-parameter GP comparisons.
3. Utility conditions the fitted signal-kernel latent covariance on a hypothetical twelve-hour paid prefix and measures future pre-update variance decrease; shared known commissioning conditions both paths.
4. One fully historical path, nine fixed quadrature offsets and a twelve-hour candidate grid bound computation. The grid and posterior-memory approximation differ from the original hourly AR planner and must be disclosed.
5. All eighteen instruments and three sites are tested; total attempted access remains 4,031 hours including 1,862 commissioning hours.
6. GP-active RMSE is 4.344 ppb, compared with 4.394 periodic, 4.282 transferred fixed and 4.321 transferred variable. The −0.049 ppb active/periodic interval crosses zero under both block lengths.
7. Device heterogeneity persists: only nine instruments improve over periodic GP, and the extended-record directions differ materially from the original AR magnitude pattern. This is descriptive behavior, not a classified device boundary or an electrode explanation.
8. Target/future-feature information tests, analytical covariance checks, exact resources and independent original-GP prediction replay pass. Posterior variance decrease is not equated with realized error reduction.
9. The experiment strengthens the method comparison and demonstrates bounded independent learner scheduling; it neither creates a new GP principle nor shows uniform or economically measured benefit.

## Authorized nonlinear delivery extension

The parent additionally requested a low-cost selected-GP comparator at 72-hour delivery. Its separate runner is `scripts/revision4_gp_delay.py`, outputs are `results/revision4/gp_delay`, and its plan was written before execution. It retains the existing periodic and regenerated AR fixed-risk delay72 calendars, with the exact original common delay support, 59,674 instrument-hours/64,886 channel observations. It is not a delayed GP-active optimization.

At prediction time t, only acquired events strictly earlier than t−72 have been assimilated. Event u arrives after prediction u+72; its original input is added to the training set afterward. Own commissioning readiness is postponed by 72 hours. Refits retain at most 512 delivered pairs in the ninety-day event-time range ending at the newest delivered event+1; the additional delivery latency is not silently converted into a different input or feature timestamp. Transformations and target scale remain commissioning fitted.

Delayed synthetic smoke passes hidden target value/missingness, arrival-boundary ordering, next-hour effect, future-feature invariance, postponed readiness, zero-delay original-GP parity and independently reconstructed delivered histories. Six formal 72-hour streams have now completed. Independent direct GP fits at the first/middle/last common scored hour of every segment and both calendars verify 138 predictions with maximum absolute difference zero. Source times, references, commissioning/attempt calendars, metadata and common masks are exact; each fit respects ninety event-time days and 512 pairs. This checks selected points by independent index reconstruction, not every delayed prediction by a second full delivery replay. Receipts are in `results/revision4/gp_delay/verification.json` and `smoke_checks.json`.

| Same selected GP on transferred AR calendars | Periodic RMSE (ppb) | Fixed-risk RMSE (ppb) | Fixed minus periodic |
|---|---:|---:|---:|
| Zero-delay, restricted common support | 4.401837 | 4.295776 | −0.106061 |
| 72-hour delivery, same common support | 4.553951 | 4.560466 | +0.006515 |

| GP direct contrast | Seven-day paired interval | Fourteen-day paired interval |
|---|---|---|
| Fixed minus periodic at zero delay: −0.106061 ppb | [−0.180469, −0.032452] | [−0.180879, −0.031594] |
| Fixed minus periodic at 72 hours: +0.006515 ppb | [−0.079122, +0.098205] | [−0.091987, +0.099783] |
| Change in timing advantage (72 minus zero): +0.112576 ppb | [+0.033666, +0.187959] | [+0.038071, +0.181767] |

All four comparisons retain eighteen instruments, 59,674 instrument-hours and 64,886 scored channel observations. The equivalent site-balanced change is +0.160649 ppb, with intervals [−0.039538,+0.373505] and [−0.064717,+0.412904], so the device-equal attenuation must not be generalized to a city population. The six delayed GP runs required approximately 173.89 seconds in aggregate measured replay time. Complete device and access-source identities remain in the delayed output tables/index.

The direct four-stream increase supports attenuation for the same-parameter GP on the specified periodic/regenerated AR-fixed calendars. The 72-hour within-delay interval includes zero; this does not establish equal performance, the absence of useful delayed selection, a universal cutoff, or failure of a delay-optimized GP policy. The experiment extends the correction-family evidence beyond AR without independently optimizing GP reference timing under delayed delivery. It supplies no GP variable-calendar delay cell and no combined nonlinear delay/sharing factorial.

An English main-text insertion supported by these completed outputs is:

> A separate selected-GP replay extends the delivery test beyond the dynamic engine. It uses the existing periodic and regenerated AR fixed-risk calendars, with 72-hour delivery after the current estimate, postponed commissioning readiness and the same 59,674 instrument-hours. The fixed-minus-periodic GP contrast changes from −0.106 ppb at immediate delivery to +0.007 ppb at 72 hours. Its direct increase is +0.113 ppb, with paired intervals of +0.034 to +0.188 using seven-day blocks and +0.038 to +0.182 using fourteen-day blocks. Thus, attenuation also occurs for this nonlinear correction under the tested calendars. This does not establish failure of a delayed GP-active policy; its acquisition rule was not optimized for delivery, and the site-balanced change remains uncertain.

## Final execution receipt

The original frozen source hashes still match both recorded plans. Three independent-policy streams and six delayed fixed-calendar streams exist and load through the unchanged statistical validator. All four paired bootstraps accepted 1,000 replicates; rejected zero-support draws were 0/1 for GP-active seven/fourteen-day blocks and 1/6 for delayed-GP seven/fourteen-day blocks. Rejection applies to every method together, rather than selecting a favorable contrast.

- Active runner SHA-256: `321efc2dbae4c1cc2689c1ea69fa4c626a00958a2b6de24e9d2d885f0ac50ef5`
- Delayed runner SHA-256: `76ebfba2a082098fa2ab6b04e8d9f6a57b302b49f0c3a87c50259e4378dd4805`
- Active plan SHA-256: `ed3b8e226eb7276d0b140cee3ae2a8c065d547ee11f5a8e552d891a0f75e789c`
- Delayed plan SHA-256: `8827e6833b9bb83f84d257300e7713408a5491c621647bdd86b2c09751d07d9a`
- Active source index SHA-256: `f3d1ecfd2de631f9811d76bcc8b7b869d86b0e565bdf7b4bb23e51869966cee7`
- Delayed source index SHA-256: `b81eb79421a76899e4457ca9470fbcfeb0010bf4fbe68e82c6d7f1a9b6b9b2e4`

The source indices enumerate every output/comparator file and its checksum. In the delayed index, zero-delay predictions remain immutable originals and are rescored by the runner on the explicitly recorded common delay mask. No source output is silently overwritten to replace its wider original mask.

**Execution and verification PASS.** Supported manuscript claims are the measured independent-policy comparison and selected-GP delay attenuation under the specified calendars. Uniform superiority, physical identifiability, new acquisition theory, exact Cohn reproduction, optimized delayed GP-active scheduling, economic savings and prospective deployment remain unsupported.
