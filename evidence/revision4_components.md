# Fourth-revision components and instrument fragility

Completed and independently verified 2026-10-01. All new outputs are under
results/revision4/components; no old predictions, manuscript files, other
scripts, or F/E raw data were modified. This is a post-review extension after
earlier main/white/memory results were inspected, not preregistration or an
untouched validation cohort. The protocol records a diagnostic-keyword repair
after analysis aborted, later input-support disclosure, and verification-only
extensions. Factors, parameters, calendars, predictions, masks, seeds and
interval definitions were not altered after results.

## Design and completed checks

Twelve streams implement coefficients adaptive/frozen × residual persistent/
white at the original B48 fixed-risk calendar. All cells share initialization
m/P, R, guard and q. White sets rho=0 and residual innovation variance to its
original stationary variance; beta initialization/uncertainty/noise are
unchanged. Frozen sets beta process noise and beta mean gain to zero, using
residual-only Joseph/Schmidt updates while retaining initial beta uncertainty
and cross-covariance. The covariance formula was independently compared with
(I−Kh)P(I−Kh)' + RKK'; it preserves beta means and the beta covariance block.
Existing retuned white and overwrite-latest sensor-decay are auxiliary
comparisons, not factorial cells. The calendar is replayed, not reoptimized.
These are computational contrasts, not physical causal decomposition.

All 12 streams passed. Adaptive/persistent (AP) predictions and every prior/
posterior state exactly match the original saved arrays. Hidden labels are
invariant, and a changed acquired current label does not change that hour's
prediction. Weekly covariance samples are positive semidefinite.

Main support is 62,652 instrument-hours (38,840 on four long Zep instruments;
23,812 on fourteen short Manchester instruments). Simultaneous channel squared
losses are averaged before physical-instrument RMSE. The ≥168-hour off-request
common mask is anchored to the last completed merged request episode,
including attempted hours; its 18,240 instrument-hours split 9,745 long/8,495
short, exactly matching the old verified memory support.

All six pairs and AP−AW−FP+FW interaction are retained for all/long/short,
all-scored/long-gap supports, macro-instrument/site-equal endpoints, and both
7/14-day blocks. Bootstrap uses 1,000/300 accepted paired circular within-site
quarter blocks, seed 20261001. Any zero-instrument-support sample is jointly
rejected. Uncertainty is conditional on observed cohort, calendar and
availability, not independent hours or city-population sampling.

## Component results (macro physical-instrument RMSE, ppb)

| Support | AP | AW | FP | FW |
|---|---:|---:|---:|---:|
| All eighteen | 4.161416 | 4.217215 | 6.375795 | 6.594411 |
| Four long | 5.422510 | 5.685251 | 11.343016 | 11.590453 |
| Fourteen short | 3.801103 | 3.797777 | 4.956588 | 5.166970 |
| ≥168 h, all | 4.801168 | 4.616212 | 6.839465 | 6.839465 |
| ≥168 h, long | 5.940896 | 6.192332 | 12.706677 | 12.706677 |
| ≥168 h, short | 4.475531 | 4.165893 | 5.163119 | 5.163119 |

Direct differences and 7-day 95% paired intervals:

- All AP−FP −2.214379 [−2.749195,−1.659343]; AW−FW −2.377195 [−2.949087,−1.724201].
- All AP−AW −0.055800 [−0.268403,0.097846]; FP−FW −0.218616 [−0.353695,−0.117278].
- All interaction 0.162816 [−0.015250,0.300443].
- Long AP−FP −5.920506 [−6.273150,−5.609350]; AP−AW −0.262741 [−0.336101,−0.184136].
- Short AP−FP −1.155485 [−1.834353,−0.470246]; AP−AW 0.003326 [−0.275889,0.201460].
- Long ≥168 h AP−FP −6.765781 [−7.961649,−5.506645]; AP−AW −0.251436 [−0.403027,−0.090023].
- Short ≥168 h AP−FP −0.687588 [−1.031548,−0.354398]; AP−AW 0.309639 [−0.024435,0.505812].

The qualitative boundaries hold with 14-day blocks. Coefficient adaptation
clearly matters. Residual persistence has no resolved uniform advantage with
adaptive coefficients over all devices; the four long records favor it, short
records do not. At ≥168 h, FP and FW differ by about 10^−11 ppb: practically
identical. Tiny intervals around this floating-point remainder must never be
called substantive significance. The remaining long-gap AP/AW difference
concerns learned coefficient history, not appreciable direct residual
extrapolation for seven days. No additive physical mechanism percentages are
justified. All 168 interval rows and all favorable/unfavorable cases are saved
in component_paired_intervals.csv.

## Full four-coefficient trajectories and uncertainty

All 23 state rows have four physical affine coefficients, prior/posterior
marginal SD, weekly means/SD, total variation, net/variation ratio, initial
departure and acquired-update contraction. Intercept transformation is
beta0−sum(beta_j median_j/scale_j); slopes are beta_j/scale_j. SD uses the full
covariance including intercept cross-terms. The 92 summary cells, weekly data
and hourly archives are supplied in coefficient_variation_uncertainty.csv,
weekly_coefficients_uncertainty.csv and three coefficient trajectory NPZs.

Long-record pollutant gains (first→last; total variation):

- Manchester Zep188 1.233534→1.289130; 8.024796.
- Manchester Zep344 first segment 1.479404→1.459730; 7.461254.
- Manchester Zep344 replacement 0.696675→0.294425; 0.715411.
- London Zep311 0.675623→0.502494; 3.934307.
- York Zep309 3.081617→0.097652; 17.078018.

York baseline changes 33.977979→6.908597 ppb, T coefficient
−1.096708→0.148477 ppb/°C, RH coefficient −0.310096→0.011407 ppb/%RH.
York gain's absolute net change is 17.47% of total variation; London's is
4.40%. Trajectories show updates/reversals, not monotonic aging.

Seven of 23 rows have no observed T/RH throughout. Zep344 replacement has 331
initial and 681 later T/RH observations but only one distinct value in each
variable. Its T/RH mean slopes remain zero; nominal T SD is about 34.4 ppb/°C
and physical intercept SD about 1,328 ppb. These unsupported directions and
prior-driven SD are not identified environmental effects or calibrated
coverage. Counts/distinct/ranges/SD are supplied in
reference_and_processed_product_context.csv. Processed manufacturer
concentration products do not identify raw-electrode aging or cross-sensitivity.

Only acquired usable innovations and same-hour coefficient jumps are
associated in the 92-row update-association file. Kalman jumps algebraically
depend on innovations; dependent-series rank correlations get no
independent-hour p-values and cannot establish a physical mechanism.

## Mean fragility and reference quality

All eighteen fixed/periodic errors are retained: twelve improve, six worsen
(IMB1/2, NS1/2/3, London Zep311). Every leave-one-physical-instrument-out mean
remains negative, ranging −0.466131 to −0.318063 ppb. All eighteen 7-day and
14-day intervals remain below zero. This is aggregation sensitivity with
unchanged predictions/schedule, not model retraining or independent
instrument/cohort validation.

| Support/endpoint | Difference | 7-day interval | 14-day interval |
|---|---:|---|---|
| All macro instrument | −0.421065 | [−0.762840,−0.101090] | [−0.672495,−0.127437] |
| Long macro instrument | −0.563128 | [−0.759778,−0.351980] | [−0.779948,−0.379650] |
| Short macro instrument | −0.380475 | [−0.813882,0.021223] | [−0.695963,−0.022631] |
| All site equal | −0.548248 | [−0.798813,−0.284480] | [−0.812532,−0.312167] |
| Clean flags macro | −0.419990 | [−0.763416,−0.095916] | [−0.676925,−0.119854] |
| Manchester duplicate macro | −0.445911 | [−0.822809,−0.052510] | [−0.775821,−0.087601] |

Short-record uncertainty crosses zero with 7-day blocks. Descriptive family
means improve for AP/AQM/SA/Zep and worsen for IMB/NS. Product, duration and
site are confounded and groups are small; no product-population tests are made.

On the exact same Manchester 29,832 instrument-hours (sixteen instruments),
primary-reference periodic/fixed RMSEs are 4.094129/3.684361, duplicate-reference
RMSEs 4.040314/3.594403. London/York have no duplicate. Clean flags alter
scoring support without refitting or changing initialization/label filtering.

London/York paired hours are 5,041/6,312 and 12,647/12,777 post-initialization;
reference missing hours 835/115, product missing hours 438/15. Paired reference
means are 7.809380/19.353425 ppb, SD 6.009608/13.260118, 5th–95th ranges
1.912500–20.139000/4.259261–46.111087. Median absolute consecutive-hour
reference changes are 1.007667/3.017355 ppb; manufacturer product minus
reference biases 0.337831/−9.897806 ppb. Both have zero scored guard-clipped
hours. These are confounded recorded contexts, not causes of schedule success.
Available-time and gap summaries retain their previously verified provenance;
no traffic or maintenance records were invented.

## Final verification

verification.json is PASS: 12 replay streams; exact original mean/prediction
parity; Joseph matrix and sampled PSD checks; 288 independently reaggregated
instrument rows; 24 aggregate rows; 92 physical coefficient/SD cells; all 168
interval estimates/quantiles; exact full-cohort 7/14-day bootstrap parity with
the existing reviewed implementation. Source hashes are unchanged; every
output hash is recorded. Outputs occupy approximately 24 MB, CPU threads two,
all cache/TEMP on D, no GPU/API/downloads.
