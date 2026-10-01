# Protocol snapshot

The following is the historical pre-computation protocol and its status at
creation, retained for provenance. Completed numerical evidence is distributed
in release v0.6.0; the current public source location is
https://github.com/williamjay1/no2-reference-information-timing . The original
pre-publication statements below do not describe the repository's later state.

---

# Sixth revision: scientific robustness before publication

GitHub publication is paused until this revision and its reproducibility checks are complete. No repository has been created or pushed.

The target is an original sensor calibration and reference-service methods article for IEEE Sensors Journal. The contribution is the explicit treatment of event time, delivery time, pending requests, and service budgets in acquisition planning. This is not a new state estimator or an electrode-level mechanism study.

This revision is retrospective and exploratory: later evaluation records were examined in earlier revisions. No new analysis will be called preregistered or independently confirmed. Selection for the expanded controller grid uses pre-2021 Manchester only; no evaluation outcome is used to choose parameters.

## Analysis plan fixed before new runs

1. Derive and verify a conditional linear-Gaussian variance-reduction property and exact zero-delay equivalence. Check closest active-calibration and delayed-information literature directly, without assuming all prior work requires immediate labels.
2. Evaluate all 16 early-development candidates: H in {168,192,240,336} hours and threshold scale in {0.75,1,1.25,1.5}. Select by development instrument-equal RMSE, with lexicographic tie breaking, and report seven-day calendar-block bootstrap selection frequencies. Retain the original H168/1.25 and the previously selected H240/1.25 policies as named comparators.
3. Replay periodic, original instantaneous-objective fixed-risk, unretuned delivery-aware, previously selected delivery-aware, and expanded-grid delivery-aware policies at delays 0,24,72,168 hours, with and without shared external commissioning. Use identical common support (each deployment ready+168) for all of these comparisons; also report the original full-support immediate contrast.
4. Audit the existing complete adaptation-by-residual 2x2 cross, give cell equations and an interaction contrast, and preserve marginal residual variance when removing persistence.
5. Define one principal contrast family: original fixed-risk minus periodic at 0h; retained delivery-aware minus original-objective and minus periodic at each nonzero delay. Apply Holm to centered paired seven-day calendar-block bootstrap tests, report effect sizes and percentile intervals, and label all analyses retrospective. Additional grids, site/device contrasts, component checks, and sensitivity experiments are descriptive/exploratory rather than a second collection of unadjusted significance claims.
6. Increase the independent GP planner's computational resolution, document runtime and selected-window overlap, and bound any GP equivalence claim to the actual approximations tested.
7. Estimate conditional device-level uncertainty, make single-device site confounding explicit, and avoid calling confidence intervals prediction intervals for new instruments.
8. Quantify predictive variance versus realized error and local window-ranking failures, distinguish forecast mismatch from budget coupling, and avoid generalizing failure to every acquisition objective.
9. Test a prespecified alternative residual structure and coefficient-drift sensitivity on the same support and resource contract. Preserve adverse outcomes. No sensitivity setting is selected on the evaluation records.
10. Test random event-specific delivery and assumed-versus-actual delay mismatch with a correctly ordered, predict-before-update estimator. Validate delayed-label access and no-future-information invariance before using the results.
11. Rewrite the main contribution, method summary and engineering decisions around measurable service requirements. Keep detailed diagnostics in the supplement while retaining all evidence needed to assess the main claim in the article.
12. Delete the Codex bibliography item and its bracket citation. Retain a truthful uncited AI-assistance acknowledgment. Update reproducibility sources and publish only after the revised scientific package passes verification. A GitHub URL will be added only after it exists; no archive DOI will be invented.

## Storage and preservation

New scripts, derived results, reports and packages are on D:/MLWork/SensorsWindowCalibration. Existing raw material remains immutable on F:. Earlier numerical controllers and sealed results are read-only inputs; new work uses revision6 paths. Author identity and nonmembership of IEEE remain as corrected. The existing Nature-style drawing system will be retained, with evidence-derived values updated as necessary.

## Status

Protocol written before new computations. Independent literature/theory, statistical diagnostics and delivery-engine work are delegated; the principal fixed-delay experiments and final integration remain with the root agent.
