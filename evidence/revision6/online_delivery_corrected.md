# Causal online planning under random reference releases

This retrospective QUANT replay recomputes both request calendars each hour from the actual delivered history. No archived adaptive calendar is transferred. Releases are simulated per event with lags 0/24/72/168 hours and probabilities .20/.20/.35/.25 (expected72 hours), three fixed paired seeds. The estimator handles out-of-sequence arrivals by event insertion and exact chronological refiltering of a bounded suffix. Predictions and request decisions precede current-hour releases. Fitted priors and every model contribution to the utility are withheld until ready+168, using the stated contract upper bound rather than future realized delays. Constant72 validation separately uses ready+72. Shared post-ready commissioning events follow the same simulated release law.

The retained-aware planner uses H240, threshold1.25, and a working fixed72-hour contract. Its anchor is a conservative forward-filter prior at event time t−72, conditioned only on received earlier-event labels. Later-event information is omitted from that anchor and represented as common events by the working-contract rollout. Older overdue unreceived events before the anchor are omitted. This is an explicitly misspecified approximate planner, not a distribution-aware optimal controller or exact random-arrival utility. The original objective uses current legitimate covariance with H168. Both obey B48/672h, four starts and12-hour nominal windows.

Constant72-hour aware prediction, query and score parity was checked against the frozen controller, including three real site streams. Synthetic checks cover hidden nonrequested labels, future labels and future delay draws, same-hour release causality, quotas and estimator equivalence to the separately validated out-of-order fixed-calendar replay. Synthetic tests are not empirical study evidence.

All conditions score the same18 instruments and59,674 instrument-hours after ready+168. Three-seed ranges describe conditional simulation variability, not deployment confidence intervals or fresh-site confirmation. No field arrival logs or prospective service validation are claimed. The pre-fix ready+72 randomized-gate outputs remain in online_delivery as failed availability diagnostics, with their original producer snapshotted under evidence/revision6/precommissioning_fix.

## Per-seed aggregate RMSE, ppb

seed,policy,macro_instrument_rmse
610001,aware_online,4.412774751561091
610001,original_online,4.471173380373361
610001,periodic,4.926763579448756
610002,aware_online,4.4466388184737475
610002,original_online,4.46923534188003
610002,periodic,4.827342089659511
610003,aware_online,4.551507923888759
610003,original_online,4.477230464535402
610003,periodic,4.807132075884138
