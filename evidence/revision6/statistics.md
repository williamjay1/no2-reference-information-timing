# Revision 6 statistical evidence

This analysis is retrospective. The later-period records were examined in previous revisions. The expanded grid and principal contrast family were fixed before these new computations, but this does not create independent confirmation.

## Existing evidence and necessary additions

- Common support: earlier delayed comparisons already used ready+168 and 59,674 instrument-hours; the new complete matrix extends that exact support to every policy, delay, and sharing condition. The original immediate full-support contrast is -0.421065 ppb on 62,652 hours; the matched-support contrast is -0.436060 ppb on 59,674 hours.
- The complete adaptation-by-residual 2x2 and seven-/fourteen-day interaction intervals already exist. They are reused, not duplicated. Equations and constraint implementation are documented in factorial_cell_equations.txt. White residuals preserve stationary marginal variance; frozen means retain coefficient uncertainty through a constrained Joseph update.
- The principal family newly receives centered paired bootstrap p-values and Holm adjustment across seven contrasts. Site-quarter circular blocks preserve observed quarter lengths. All eleven required policy streams share each bootstrap draw. Intervals are marginal percentile intervals, not simultaneous family intervals. No hour-IID test is used.
- Earlier full-support draws already contain all 18 device RMSEs. Device confidence intervals are extracted from those saved paired draws, without repeating training or claiming prediction intervals for a new device. London and York each remain confounded with their single physical instrument.
- Existing Gaussian coverage/variance-error diagnostics and the 208-case local window diagnostic are reused. New matched-support diagnostics and an explicitly offline actual-input and availability covariance rollout help distinguish forecasting mismatch from quota coupling. No future target values enter the oracle covariance score; future features and availability are used only in this offline diagnosis.

## Principal seven-day results

| Delay (h) | Contrast | RMSE difference (ppb) | Marginal 95% CI | Centered p | Holm p |
| --- | --- | ---: | --- | ---: | ---: |
| 0 | original minus periodic | -0.436060 | [-0.824645, -0.085350] | 0.016397 | 0.065587 |
| 24 | aware_retained minus original | -0.182405 | [-0.411451, 0.065908] | 0.160768 | 0.160768 |
| 24 | aware_retained minus periodic | -0.343918 | [-0.673672, -0.045911] | 0.025995 | 0.065587 |
| 72 | aware_retained minus original | -0.599328 | [-0.884433, -0.346099] | 0.000200 | 0.001400 |
| 72 | aware_retained minus periodic | -0.396234 | [-0.736216, -0.064994] | 0.016997 | 0.065587 |
| 168 | aware_retained minus original | -0.458120 | [-0.819952, -0.212996] | 0.007998 | 0.039992 |
| 168 | aware_retained minus periodic | -0.453694 | [-0.631989, -0.188093] | 0.000200 | 0.001400 |

The p-values approximate centered sampling distributions conditional on this observed cohort and fixed policies. They do not eliminate retrospective design decisions or establish performance in a new deployment. Fourteen-day sensitivity changes which marginal thresholds cross 0.05: notably retained minus original at 168 h is 0.039992 after Holm with seven-day blocks, but 0.063957 with fourteen-day blocks. Do not choose a block length based on this threshold.

## Device intervals

18 conditional device contrasts; 8 intervals entirely below zero, 0 entirely above, 10 spanning zero. These are descriptive marginal intervals and are not 18 independent hypothesis tests.

## Local window-ranking diagnosis

| Offline criterion | Spearman vs realized MSE difference | Sign agreement | Mean chosen MSE |
| --- | ---: | ---: | ---: |
| predicted_difference | 0.004494 | 0.524038 | 50.953007 |
| oracle_context_variance_difference_horizon | 0.088435 | 0.552885 | 50.287832 |
| oracle_context_variance_difference_scored | 0.085983 | 0.552885 | 50.287832 |

Both local branches spend exactly 12 attempted reference hours. Failure of their value ranking therefore cannot be attributed solely to the fleet quota constraint. The input/availability oracle is a diagnosis, not a deployable selector; remaining mismatch can reflect mean-model error, noise realizations, clipping, and conditional covariance assumptions. No universal failure claim about information gain or acquisition is warranted.

## Preserved adverse boundaries

- Unretuned H168 at 168 h delay has a first affected output beyond the horizon. All finite candidate scores are zero and all paid attempts are quota-forced at all three sites, with and without sharing. Its RMSE is not evidence of successful value-based scheduling.
- High coefficient process noise at 72 h delay gives retained delivery-aware RMSE greater than periodic. This adverse prespecified sensitivity remains in the evidence.

## Output interfaces

CSV files in results/revision6/statistics contain the seven principal contrasts and fourteen-day sensitivity; all 40 aggregate matrix cells with forced fractions; 120 site forcing rows; 720 common-support device variance-error rows; 18 full-support device intervals at each old block length; exact common-support device intervals; existing component interactions; and all 208 augmented local diagnostic cases. verification.json hashes every external input and records the statistical scope.
