# Independent implementation and research audit

Date: 2026-09-30. Stage: development, before freezing and evaluating post-2020 prediction performance. Scope: data screening, site construction, common engine, fixed schedules, acquisition controller, empirical baselines and development selection. No held-out predictive errors were calculated or inspected by this audit. Source and prediction code outside the auditor's acquisition module were read only. The subsequently authorized V2 acquisition repair is separately identified below.

## Decision and immediate actions

**CONDITIONAL: ready for a fresh development replay after the repaired controller passes its tests, not yet a submission or superiority decision.** A code-level privacy defect was not found in the examined query paths. The significant planning/execution mismatch discovered in V1 has been repaired and tested. Existing V1 policy results must be treated as provisional and replaced by a version-consistent development rerun before hyperparameters are frozen.

Required before final performance claims:

1. Rerun the development acquisition grid after the V2 free-initialization/calendar repair. Do not mix V1 and V2 results in selection tables.
2. Complete the promised common guarded-input empirical baseline comparison. Current `run_baseline_pilot.py` was still using unguarded inputs and q=1e-4 when read, whereas the current site engine uses guarded inputs and q=1e-3. Their scores cannot yet support a fair current-engine adequacy claim.
3. Freeze the loss aggregation for held-out evaluation: equal physical instrument versus equal channel/segment versus pooled hours. Report site-specific estimates. Development happens to contain 14 distinct instruments, so its current deployment mean does not overweight an IMB device; post-2020 data do contain multi-channel instruments.
4. Add an invariant to the production scorer: every compared method's count of finite evaluated predictions must equal the prespecified common scoring mask count. `metrics` otherwise silently drops nonfinite predictions. The baseline pilot already asserts this; `run_policy_development.summarize` did not yet do so when read.
5. Keep a ledger of actual initialization hours, extra requested hours, successful requests, paid-window initiations and forced-window hours. Nominal equal budgets are insufficient when an initialization calendar can make some paid allocations infeasible.

The root reported that initial development adaptive performance did not beat periodic access. That is a preliminary message, not an independently verified result of this audit; it must not be converted into a favorable finding by wording. The repaired rerun can change the comparison. If the result remains weak, narrow the contribution to an honest comparison of reference-access designs or reconsider the method.

## Checks actually performed

| Check | Observed result | Meaning |
|---|---|---|
| Site-hour reference consistency after `read_data` | 0 site-hour groups with inconsistent reference values, including missingness | `drop_duplicates(timestamp)` currently selects the same y across device rows; add an assertion for future source versions. |
| Duplicate site/device/hour rows | 0 | Current local reindexing has unambiguous observations. |
| Infinite primary sensor values | 0 | The current guard does not accidentally turn a recorded infinity into a valid finite training value in this dataset. |
| Reference version among finite screened y | Ratified only, 149229 joined rows | Version screening operates as intended. This is a joined-row count, not independent reference hours. |
| Sensor flags among finite screened x | Empty: 161998; Warning: 2702; Error: 0 | Error rows are removed from available sensor input; Warning rows remain intentionally. |
| Source flag counts | 2435 retrospective-collection rows; 222 manufacturer-rebase rows | Product/history caveats need explicit reporting and a relevant sensitivity analysis. |
| Development identities | 14 models, 14 different instruments | Current development macro averaging is not inflated by multi-channel duplication. |
| IMB endpoint matching in saved held-out eligibility | IMB1's three channels share both endpoints; IMB2's three channels also share endpoints | The present endpoint-based weighting rule gives one-third weights correctly for this dataset. |
| IMB contemporaneous nonmissing primary channels | IMB1: all 1772 recorded hours have 3 valid channels; IMB2: all 1811 recorded hours have 3 | Static one-third weights do not underweight an instrument through single-channel dropouts in the observed source rows. Missing whole-site-grid hours remain all-invalid. |
| Fixed schedule synthetic calendars | 36 combinations of duration, budget and phase passed exact feasible-hour and <=4-start checks | The routine covers tested ordinary initialization blocks and short final blocks. |
| Deliberately fragmented free calendar | Raises `Initialization calendar cannot accommodate fixed windows` | Fixed equal-length continuous episodes require geometric feasibility, not only a sufficient number of free slots. This is an explicit boundary, not hidden budget spending. |
| Acquisition V2 synthetic/analytic suite | PASS, final extended suite 23.69 seconds including JIT | Covers budgets, start caps, explicit shortfall, hidden labels/missingness, current labels, future features/validity, fixed-mask parity, analytic risk/IG with free observations, instrument duplication weights and H=168 causality. |

No new raw data were downloaded, modified, moved or deleted. Read-only data checks used the existing D: derived table and selected raw CSV headers. The only raw headers inspected were the primary reference file and duplicate-reference file.

## Data screening and reference identity

`study_data.read_data` uses the unfiltered out-of-box joined table, rather than dropping every row without a reference before replay. It hides Error sensor rows, excludes non-Ratified reference values, and screens T/RH by their own quality flags and stated physical ranges. This preserves the distinction between visible sensor context and evaluable reference labels. The temperature interpretation is based on the CSV's Celsius-scale observations; the source-export Kelvin discrepancy remains a documented provenance point to explain, not silently resolve by adding 273.15.

The main `QUANT_Reference_hourly.csv` header contains time, location, version, measurand and measurement. It has no `qc_flag`. The independent `QUANT_DuplicateRef_hourly.csv` does have `qc_flag`. Therefore the main reference preparation has not accidentally discarded an available primary-reference QC column. Any duplicate-reference agreement analysis must, however, explicitly use its separate QC field; finite duplicate values alone are insufficient.

`data_audit.py` asserts the source reference key and prohibits competing NO2 reference versions at the same site/hour before the many-to-one join. A direct inspection of the derived data additionally found zero reference inconsistencies across replicated device rows. The site constructor's `drop_duplicates(timestamp)` is therefore safe for this snapshot, but an invariant immediately before dropping duplicates would prevent a future silent conflict.

The signals are vendor concentration products. They are not raw electrode voltage, and the reference-mask replay does not undo vendor processing. The local first-party QUANT descriptor distinguishes out-of-box from separately delivered cal1/cal2 products; this supports exclusion of retrospectively calibrated versions. It does not justify claiming that every out-of-box hour was free from manufacturer rebasing, communication recovery or quality review. The retained flag fields allow transparent sensitivity checks. Do not remove flagged events selectively after seeing which policy wins.

The primary reference is ratified archival data. Its final ratification and publication timestamps need not equal the original observation timestamp. The model uses this as an idealized on-demand hourly calibration target with success discovered after request. Describe a quality-screened retrospective reference-access replay; do not assert demonstrated real-time target availability, instrument duty-cycle feasibility, or physical maintenance savings. A delay sensitivity is valuable if claims depend on immediate operational calibration, but this audit does not require changing the research into a hardware study.

## Site formation, initialization and generalization

The initialization is local to each deployment segment: first 336 calendar rows, at least 72 valid initial pairs. The site-ready index offsets this by the local segment start. The already-fitted state is not advanced or scored before readiness; future initialization knowledge in a later device's private model cannot affect earlier policy scores because only current ready, valid models enter the score. Grid padding retains calendar elapsed time.

The site-wide initialization mask is the union of these local setup periods. Any reference hour obtained for a newly initializing device is legitimately available to every already-ready valid device. Its cost is separate from the additional query budget. This is implemented in both fixed and adaptive fleet replay. The total initialization cost must be counted once per site hour, not 336 multiplied by every overlapping channel.

The union schedule is treated as externally known. Future entries are used to count feasible paid slots and, after V2, to propagate the known common free updates in both planning paths. This assumption is explicit and does not authorize future sensor feature, target, target-missingness or unplanned hardware-availability access. If actual deployment dates were not known, exact budget guarantees and calendar-aware planning would need a different formulation.

Ending replay at the last ratified sensor pair and requiring at least 672 later scoreable pairs are retrospective design/eligibility choices. They apply equally across policies and do not use error magnitude, but they are not prospective enrollment rules. Disclose selection based on archival coverage and give the resulting sample flow. The controller knows the study calendar horizon, which is part of the experimental allocation design.

London's eligible held-out segment is Zep311 s2, and York's is Zep309 s1. The wide Manchester fleet supplies most instrument diversity; the other two sites each supply one eligible instrument segment. Some instrument identities were observed earlier in Manchester. Thus three sites do not imply three large independent fleets or entirely unseen hardware. State the unit and scope when claiming transfer, and do not create independent sample size by counting hourly records or replacement segments.

## Acquisition time order and contexts

All ready device predictions and predictive variances are stored before any reference value at the same hour is read. A request is selected from current/past visible features, states learned from available labels and the known setup calendar. Success is checked after the request. A missing or withheld reference cannot enter policy selection through a `y.notna()` eligibility mask. Conditional changes after an observed request failure are legitimate.

Historical context eligibility is checked with `origin >= 0` and `origin + H < t`; rollout indices run only through `origin + H`, strictly before the decision time. The first step uses the actual current feature. Missing historical context uses a frozen current feature. After the long-horizon repair, the lags are derived from max(72,H+24), so increasing H to 168 does not silently eliminate every historical scenario or allow future rows.

The current implementation selects and commits a window length at its start. It does not repeatedly solve a stopping problem during the window and has no global optimality guarantee. The appropriate description is a causal joint timing-and-duration heuristic. Sequential IG, predictive-variance selection and risk selection share the common pacing and budget machinery. The risk-independent alias `classical_a` denotes an independence-planning ablation with the same dynamic correction engine; it is not a literal implementation of a separate static A-optimal algorithm.

V1 omitted known free initialization from both covariance paths and treated paid window length as clock length despite execution pausing the paid counter during free hours. This could exaggerate near-initialization gains. V2 adds known common free observations to both paths after their predictions and skips those hours in the paid prefix count. The scalar analytic test independently checks posterior precision counts, paid information and baseline uncertainty with free observations at offsets 2, 3 and 5. The known schedule does not reveal whether the future reference succeeds: the planner assumes planned observations succeed, and unexpected failure is handled in actual replay only.

The risk calculation is a working-model covariance prediction under historical context approximations. It is not verified actual future error reduction and it does not prove that all model misspecification, drift or clipping losses are reducible. If final predictions are clipped at zero, the variance criterion remains an approximation to the loss of those clipped predictions. Keep that boundary explicit.

## Budgets, fixed schedules and weighting

`fixed_schedules.py` uses only the known initialization calendar and its own previously allocated windows. Periodic phase choices and random seeds do not inspect future signals or labels. It removes overlap with initialization, separates fixed paid windows by a gap, preserves the same prorated block target and caps starts. A greedy arrangement can fail on an artificially fragmented initialization calendar even when the number of available hours is sufficient; that failure is reported rather than silently exceeding a budget. Check the actual known study calendars before freezing rather than changing the algorithm after seeing errors.

Adaptive scheduling charges attempted hours even when the current fleet has no valid signal or the reference subsequently fails. This is necessary for honest exact attempt budgets without knowledge of future availability. Fixed scheduling similarly may waste an attempted hour; both need counts of useful and failed attempts. A zero-slot block reports explicit shortfall. Initialization hours are never reclassified as extra expenditure.

Adaptive `starts` are paid-window initiations, not physical reference on/off events. A free initialization interval can connect reference access physically while the paid counter pauses. The `forced_query` flag marks every hour in an episode whose duration was overridden by the last-start/catch-up rule, not just the few excess hours beyond the originally preferred length. Report this as the share of hours in forced/overridden windows; a claim of exact marginal forced hours would require another counter.

Instrument weights are normalized over currently valid and ready model entries. The observed IMB channels have matching endpoints and simultaneous validity, so static one-third channel weights produce the intended unit weight here. If this code is reused with unequal concurrent-channel availability, weights would need dynamic normalization within instrument or the method should explicitly retain fixed channel weights. Replacement segments that do not overlap correctly retain unit weight.

Prediction-error aggregation must preserve the distinction between channel, instrument and site. First averaging RMSEs is not the same as pooling squared errors then taking one square root; neither should be substituted silently. Prespecify the primary aggregate, retain physical-unit per-site/instrument summaries, and pair identical scoring hours across policies. Serial and cross-device dependence must be reflected in uncertainty intervals; three cities are not a large geographic sample.

## Guarded engine and empirical baselines

The support guard fits its median and MAD on the first 336 sensor hours only and clips the primary sensor input rather than excluding difficult scored hours. It therefore has a causal definition. Infinities require care in general because clipping can turn them finite, but none was found in this derived sensor column. Physical T/RH screens operate separately. Initial all-missing auxiliary columns generate NumPy warnings and are subsequently assigned finite fallback medians/scales by the current engine; these warnings are not by themselves evidence of a failed fit, but metadata should make the missing-weather condition visible.

A guarded current engine cannot be fairly contrasted against an unguarded empirical competitor solely on a score difference. Pass the same guarded current features into Ridge/HGB and the same shared site reference access history, retaining the separate initial versus daily-update models. The daily refit baselines intentionally differ from the hourly filter update frequency and are an engine-adequacy sensitivity, not proof of an acquisition-policy effect. The two tasks must not be merged in the results narrative.

The local baseline module's synthetic tests verified hidden labels, queried-hour timing, future feature independence, calendar-based 90-day expiry and missing data handling. Those tests validate the implementation boundary; they do not establish competitive QUANT accuracy. A bounded development hyperparameter search and the same score-mask assertion remain appropriate before drawing a comparative conclusion.

## Remaining claim boundaries

- Standard covariance risk reduction, Bayesian information gain and model-state uncertainty are recognized criteria; a new acronym is not the contribution.
- A causal replay can be correct while showing no advantage over periodic reference access. Preserve that result if it remains after repairs.
- A high forced-window fraction can make a nominal adaptive policy mainly a budget-completion rule. Report it and inspect whether discretionary choices add value.
- Hours of archived reference revealed are not measured instrument energy, staff effort, field returns or avoided redeployments.
- WUDDR-style thresholding is an explicitly adapted comparator; do not state exact reproduction without its complete original parameters and protocol.
- Reference-product, flag and manufacturer-processing limitations should be stated once clearly and addressed with relevant sensitivity analyses, not hidden by a generic claim of raw public data.

This audit permits further development execution after the code repair. It does not certify novelty, held-out success, submission readiness or likely journal acceptance.
