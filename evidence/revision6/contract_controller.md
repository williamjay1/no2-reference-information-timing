# Independent commissioning-contract repair

Completed 2026-10-01. New controller: `scripts/revision6_contract_controller.py`. The immutable fourth-review controller and original controller are unchanged. This repair addresses the important issue identified in `method_readonly_audit.md`; full nonsharing study replays are owned by the main agent and are not claimed completed by this subtask.

## Change and API

Commissioning has two roles that must be separated. The true commissioning allocation mask always pauses a proposed paid window, because actual execution cannot charge those hours as additional paid access. The common-information mask contains commissioning hours only when sharing is enabled. Immediate and delivery-aware covariance utilities now accept these masks separately. Thus removing information sharing does not remove known access-allocation constraints from the planner.

`run_site` retains every previous argument and adds an optional final keyword `delay_aware=True`. Use `delay_aware=False` for the delay-naive original planner, and the default `True` for delivery-aware planning. Both modes use the same repaired allocation contract, correction engine and enforcement. `run_site_fixed` has the unchanged prior interface. Existing external calls remain compatible.

## Verified properties

- Eight synthetic shared-condition comparisons, crossing delay-aware/naive planning with 0/24/72/168-hour delivery, match the corresponding immutable aware/original controller arrays exactly: prediction, variance, query, success, starts, start mask, forced hours, chosen lengths, candidate scores and budgets.
- Exact hand calculation: prior scalar variance 1, noise variance 1, no process noise, a two-attempt paid prefix, future commissioning at offsets 1 and 2. The correct paid events are offsets 0 and 3. Over an immediate six-hour output horizon, the score is `(3×1/2 + 3×2/3)/6 = 7/12`. At two-hour delivery over an eight-hour horizon, the score is `(3×1/2 + 3×2/3)/8 = 7/16`; outputs at offsets 1 and 2 gain nothing. Both repaired utility functions match these values. The former conflated mask incorrectly gives `23/36` and `23/48`.
- The two repaired utility interfaces give the same score at zero delay with matching settings and allocation/information calendars.
- Nonsharing synthetic replays preserve hidden-label invariance, prediction-before-release chronology, exact attempted block budgets, the four-start cap and commissioning exclusion.
- The inherited six-policy synthetic test suite also passes, including fixed-mask parity, causal input checks, analytical covariance objectives, weights and budget shortfall handling.
- Real Manchester prefix smoke uses 21 deployment/channel streams, 1,680 evaluation hours, H240 and d72. Shared aware and naive paths match their immutable source implementations exactly. Nonshared paths each spend 96 attempted hours, satisfy each feasible block quota and start cap, and never charge commissioning as paid. In this early prefix the repaired and former query calendars happen to coincide; this smoke establishes implementation consistency and does not establish a full-study performance effect.

Synthetic tests are implementation verification, not empirical evidence of sensor-calibration benefit. The real-prefix smoke is not a full-period performance comparison.

## Reproduction

Run `revision6_contract_controller.py --self-test` for the inherited and contract-specific synthetic tests; run `--real-smoke` for the real fleet prefix. Test records are `evidence/revision6/contract_controller_checks.json` and `contract_real_smoke.json`, including the new runner SHA256. Full repaired study cells must identify this controller in their provenance and retain the original common scoring support.
