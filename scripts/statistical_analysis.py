"""Physical-instrument metrics and paired calendar-block uncertainty.

Read-only with respect to predictions, labels, and models. Run after the main
experiment is frozen. Percentile intervals are conditional on these observed
sites/instruments and preserve each site's observed calendar-quarter lengths.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SITES = ("Manchester", "London", "York")
EXPECTED_BUDGETS = (12, 24, 48)
EXPECTED_METHODS = ("periodic", "random1", "random2", "random3", "risk",
                    "information_gain", "uncertainty", "risk_independent",
                    "risk_fixed", "window_uncertainty", "noAR")
EXPECTED_COMMON_METHODS = EXPECTED_METHODS + ("init_only", "reference_all", "ridge_stream",
                                             "hgb_stream", "gp_stream", "ewrls", "bias_ewma")
EXPECTED_B48_ONLY_METHODS = ("ridge_initial", "hgb_initial", "factory", "hgb_stream_risk",
                            "ewrls_risk", "bias_ewma_risk", "guard_removed", "weather_removed",
                            "risk_independent_matched", "risk_fixed_matched")
FILE_RE = re.compile(r"^(?P<site>.+)_B(?P<budget>\d+)_(?P<method>.+)\.npz$")


@dataclass
class Run:
    site: str
    budget: int
    method: str
    pred: np.ndarray
    y: np.ndarray
    times: pd.DatetimeIndex
    score_masks: np.ndarray
    query: np.ndarray
    initmask: np.ndarray
    metadata: list[dict]
    warning_mask: np.ndarray | None = None
    y_duplicate: np.ndarray | None = None
    duplicate_valid: np.ndarray | None = None
    path: str = "memory"
    quality_flag_mask: np.ndarray | None = None


def _bool_array(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or not np.all(np.isin(a, [0, 1])):
        raise ValueError(f"{name}: expected boolean shape {shape}, got {a.shape}")
    return a.astype(bool, copy=False)


def validate_run(run: Run) -> None:
    """Reject silent method-specific exclusions or ambiguous metadata."""
    if run.pred.ndim != 2:
        raise ValueError(f"{run.path}: predictions must have shape (D,T)")
    d, t = run.pred.shape
    if not d or not t or len(run.times) != t or run.y.shape != (t,):
        raise ValueError(f"{run.path}: empty or inconsistent dimensions")
    if run.times.hasnans or not run.times.is_unique or not run.times.is_monotonic_increasing:
        raise ValueError(f"{run.path}: timestamps must be unique, valid and increasing")
    if t > 1 and not np.all((run.times[1:]-run.times[:-1]) == pd.Timedelta(hours=1)):
        raise ValueError(f"{run.path}: resampling requires a complete hourly calendar grid")
    run.score_masks = _bool_array(run.score_masks, (d, t), "score_masks")
    run.query = _bool_array(run.query, (t,), "query")
    run.initmask = _bool_array(run.initmask, (t,), "initmask")
    if np.any(run.query & run.initmask):
        raise ValueError(f"{run.path}: additional query overlaps initialization access")
    if run.method == "factory" and (np.any(run.query) or np.any(run.initmask)):
        raise ValueError(f"{run.path}: factory must have zero query and initialization access")
    if run.method in ("ridge_initial", "hgb_initial") and np.any(run.query):
        raise ValueError(f"{run.path}: initial-only static model must have zero additional access")
    if len(run.metadata) != d:
        raise ValueError(f"{run.path}: metadata length differs from prediction rows")
    keys = []
    for row in run.metadata:
        if not all(row.get(k) is not None for k in ("site", "instrument", "family")):
            raise ValueError(f"{run.path}: metadata needs site, instrument and family")
        if row["site"] != run.site:
            raise ValueError(f"{run.path}: filename and metadata site disagree")
        keys.append(_row_key(row))
    if len(set(keys)) != len(keys):
        raise ValueError(f"{run.path}: ambiguous duplicated deployment/channel metadata")
    if np.any(run.score_masks & ~np.isfinite(run.pred)):
        raise ValueError(f"{run.path}: nonfinite predictions on the fixed scoring mask")
    if np.any(run.score_masks & ~np.isfinite(run.y)[None, :]):
        raise ValueError(f"{run.path}: nonfinite reference on the scoring mask")
    if np.any(run.pred[run.score_masks] < 0):
        raise ValueError(f"{run.path}: expected predictions already clipped at zero")
    if run.warning_mask is not None:
        run.warning_mask = _bool_array(run.warning_mask, (d, t), "warning_mask")
    if run.quality_flag_mask is not None:
        run.quality_flag_mask = _bool_array(run.quality_flag_mask, (d, t), "quality_flag_mask")
    if (run.y_duplicate is None) != (run.duplicate_valid is None):
        raise ValueError(f"{run.path}: duplicate reference requires explicit QC-valid mask")
    if run.y_duplicate is not None:
        if run.y_duplicate.shape != (t,):
            raise ValueError(f"{run.path}: duplicate reference length differs")
        run.duplicate_valid = _bool_array(run.duplicate_valid, (t,), "duplicate_valid")
        if np.any(run.duplicate_valid & ~np.isfinite(run.y_duplicate)):
            raise ValueError(f"{run.path}: QC-valid duplicate reference contains nonfinite values")


def _row_key(row):
    return tuple(str(row.get(k, "")) for k in
                 ("site", "instrument", "family", "device", "deployment", "channel", "start", "end"))


def load_run(path: str | Path) -> Run:
    path = Path(path)
    match = FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Unrecognized prediction filename: {path.name}")
    with np.load(path, allow_pickle=False) as z:
        required = {"pred", "y", "times", "score_masks", "query", "initmask", "metadata_json"}
        if required - set(z.files):
            raise ValueError(f"{path}: missing {sorted(required-set(z.files))}")
        raw_metadata = np.asarray(z["metadata_json"]).item()
        if isinstance(raw_metadata, bytes):
            raw_metadata = raw_metadata.decode("utf-8")
        dup = np.asarray(z["y_duplicate"], float) if "y_duplicate" in z else None
        flag_keys = [k for k in ("duplicate_valid", "y_duplicate_valid", "duplicate_qc_valid") if k in z]
        if len(flag_keys) > 1 and any(not np.array_equal(z[flag_keys[0]], z[k]) for k in flag_keys[1:]):
            raise ValueError(f"{path}: conflicting duplicate-reference validity masks")
        flag = np.asarray(z[flag_keys[0]]) if flag_keys else None
        run = Run(site=match["site"], budget=int(match["budget"]), method=match["method"],
                  pred=np.asarray(z["pred"], float), y=np.asarray(z["y"], float),
                  times=pd.DatetimeIndex(pd.to_datetime(z["times"].astype(str))),
                  score_masks=np.asarray(z["score_masks"]), query=np.asarray(z["query"]),
                  initmask=np.asarray(z["initmask"]), metadata=json.loads(raw_metadata),
                  warning_mask=np.asarray(z["warning_mask"]) if "warning_mask" in z else None,
                  y_duplicate=dup, duplicate_valid=flag, path=str(path),
                  quality_flag_mask=np.asarray(z["quality_flag_mask"]) if "quality_flag_mask" in z else None)
    validate_run(run)
    return run


def check_comparable(runs: Mapping[str, Run]) -> None:
    # Factory is evaluated on identical outcomes/times but consumes no labels.
    # Compare every calibrated model with a nonfactory initialization calendar,
    # including when factory happens to be first in the file/mapping order.
    first = next((r for r in runs.values() if r.method != "factory"), next(iter(runs.values())))
    for run in runs.values():
        validate_run(run)
        if (run.site, run.budget) != (first.site, first.budget):
            raise ValueError("A site bundle must have one site and one budget")
        checks = {
            "timestamps": run.times.equals(first.times),
            "reference": np.array_equal(run.y, first.y, equal_nan=True),
            "deployment/channel order": [_row_key(r) for r in run.metadata] == [_row_key(r) for r in first.metadata],
            "scoring masks": np.array_equal(run.score_masks, first.score_masks),
            "initialization calendar": run.method == "factory" or np.array_equal(run.initmask, first.initmask),
        }
        for name, same in checks.items():
            if not same:
                raise ValueError(f"{run.path}: {name} differs from {first.path}")
        if run.warning_mask is not None and first.warning_mask is not None and not np.array_equal(run.warning_mask, first.warning_mask):
            raise ValueError(f"{run.path}: warning masks differ")
    quality_masks = [r.quality_flag_mask for r in runs.values() if r.quality_flag_mask is not None]
    if quality_masks and any(not np.array_equal(mask, quality_masks[0]) for mask in quality_masks):
        raise ValueError(f"{first.site}: quality flag masks differ across methods")


def weighted_quantile(values, weights, q=.9):
    """Inverse weighted empirical CDF, with no interpolation."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not good.any():
        return float("nan")
    values, weights = values[good], weights[good]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    i = np.searchsorted(cumulative, q*cumulative[-1], side="left")
    return float(values[order[min(i, len(order)-1)]])


def build_site_bundle(runs: Mapping[str, Run], methods=None, reference="primary"):
    """Average losses, never predictions, within concurrent instrument channels."""
    check_comparable(runs)
    methods = tuple(sorted(runs) if methods is None else methods)
    first = next(iter(runs.values()))
    metadata = first.metadata
    ids = tuple(sorted({str(r["instrument"]) for r in metadata}))
    mask = first.score_masks.copy()
    y = first.y
    if reference == "duplicate":
        candidates = [r for r in runs.values() if r.y_duplicate is not None]
        if not candidates:
            return None
        dup = candidates[0]
        for run in candidates[1:]:
            if not np.array_equal(run.y_duplicate, dup.y_duplicate, equal_nan=True) or not np.array_equal(run.duplicate_valid, dup.duplicate_valid):
                raise ValueError(f"{run.site}: duplicate references differ across methods")
        y = dup.y_duplicate
        mask &= dup.duplicate_valid[None, :]
    elif reference == "no_warning":
        candidates = [r for r in runs.values() if r.warning_mask is not None]
        if not candidates:
            return None
        if any(not np.array_equal(r.warning_mask, candidates[0].warning_mask) for r in candidates):
            raise ValueError("Warning masks disagree")
        mask &= ~candidates[0].warning_mask
    elif reference == "clean_flags":
        candidates = [r for r in runs.values() if r.quality_flag_mask is not None]
        if not candidates:
            return None
        # quality_flag_mask is supplied by the data preparation stage as the
        # union of Warning, rebase and retrospective-collection indicators.
        # Only evaluation support changes: predictions/models are frozen.
        mask &= ~candidates[0].quality_flag_mask
    elif reference != "primary":
        raise ValueError("reference must be primary, duplicate, no_warning or clean_flags")
    m, i, t = len(methods), len(ids), len(y)
    se = np.zeros((m, i, t)); ae = np.zeros_like(se); signed = np.zeros_like(se)
    active = np.zeros((i, t), bool)
    channel_count = np.zeros((i, t), np.int16)
    quantile_samples = {}; families = []; primary_instrument_hours = 0
    for j, identity in enumerate(ids):
        rows = np.array([k for k, row in enumerate(metadata) if str(row["instrument"]) == identity])
        family_set = {str(metadata[k]["family"]) for k in rows}
        if len(family_set) != 1:
            raise ValueError(f"{identity}: inconsistent instrument families")
        families.append(next(iter(family_set)))
        primary_instrument_hours += int(first.score_masks[rows].any(axis=0).sum())
        masks = mask[rows]
        counts = masks.sum(axis=0)
        active[j] = counts > 0
        channel_count[j] = counts
        denom = np.maximum(counts, 1)
        for k, name in enumerate(methods):
            errors = np.where(masks, runs[name].pred[rows]-y[None, :], 0.)
            se[k, j] = (errors*errors).sum(axis=0)/denom
            ae[k, j] = np.abs(errors).sum(axis=0)/denom
            signed[k, j] = errors.sum(axis=0)/denom
            weights = np.broadcast_to(1./denom, masks.shape)[masks]
            quantile_samples[(name, identity)] = (np.abs(errors[masks]), weights)
    return dict(site=first.site, budget=first.budget, methods=methods, instruments=ids,
                families=tuple(families), times=first.times, active=active,
                channel_count=channel_count, se=se, ae=ae, signed=signed,
                quantile_samples=quantile_samples, reference=reference,
                support=dict(primary_channel_observations=int(first.score_masks.sum()),
                    included_channel_observations=int(mask.sum()),
                    excluded_channel_observations=int(first.score_masks.sum()-mask.sum()),
                    primary_instrument_hours=primary_instrument_hours,
                    included_instrument_hours=int(active.sum()),
                    included_instruments=int(active.any(axis=1).sum())))


def _finite_float(x):
    return float(x) if np.isfinite(x) else None


def metric_tables(bundles):
    """Return instrument (pooled/site), aggregate, daily and monthly tables."""
    site_rows = []; period_rows = {"daily": [], "monthly": []}; pooled = {}
    occupied = {}
    for b in bundles:
        for i, identity in enumerate(b["instruments"]):
            active = b["active"][i]
            count = int(active.sum())
            # An instrument can move between sites, but it cannot occupy two
            # distinct sites at the same scored clock hour under one identity.
            occupancy_key = (b["budget"], identity, b["reference"])
            clock = set(b["times"][active])
            previous = occupied.setdefault(occupancy_key, {})
            for other_site, other_clock in previous.items():
                if other_site != b["site"] and clock.intersection(other_clock):
                    raise ValueError(f"{identity}: overlapping scored hours at {other_site} and {b['site']}")
            previous.setdefault(b["site"], set()).update(clock)
            for k, method in enumerate(b["methods"]):
                samples = b["quantile_samples"][(method, identity)]
                sums = np.array([b[key][k, i].sum() for key in ("se", "ae", "signed")])
                row = dict(site=b["site"], instrument=identity, family=b["families"][i],
                           method=method, budget=b["budget"], reference=b["reference"],
                           instrument_hours=count, channel_observations=int(b["channel_count"][i].sum()),
                           rmse=np.sqrt(sums[0]/count) if count else np.nan,
                           mae=sums[1]/count if count else np.nan, bias=sums[2]/count if count else np.nan,
                           p90_absolute_error=weighted_quantile(*samples))
                site_rows.append(row)
                key = (b["budget"], method, identity, b["reference"])
                p = pooled.setdefault(key, dict(sums=np.zeros(3), hours=0, channels=0, samples=[],
                                               sites=set(), families=set()))
                p["sums"] += sums; p["hours"] += count
                p["channels"] += row["channel_observations"]; p["samples"].append(samples)
                p["sites"].add(b["site"]); p["families"].add(row["family"])
        for frequency, storage in (("D", "daily"), ("M", "monthly")):
            periods = b["times"].to_period(frequency)
            for period in periods.unique():
                ix = np.asarray(periods == period)
                counts = b["active"][:, ix].sum(axis=1)
                for k, method in enumerate(b["methods"]):
                    for i, identity in enumerate(b["instruments"]):
                        if counts[i]:
                            period_rows[storage].append(dict(site=b["site"], instrument=identity,
                                family=b["families"][i], method=method, budget=b["budget"],
                                reference=b["reference"], period=str(period),
                                instrument_hours=int(counts[i]),
                                rmse=float(np.sqrt(b["se"][k, i, ix].sum()/counts[i])),
                                mae=float(b["ae"][k, i, ix].sum()/counts[i]),
                                bias=float(b["signed"][k, i, ix].sum()/counts[i])))
    instrument_rows = []
    for (budget, method, identity, reference), p in pooled.items():
        if len(p["families"]) != 1:
            raise ValueError(f"Physical instrument {identity} has different families across sites")
        n = p["hours"]
        values = np.concatenate([s[0] for s in p["samples"]])
        weights = np.concatenate([s[1] for s in p["samples"]])
        instrument_rows.append(dict(instrument=identity, family=next(iter(p["families"])),
            sites="|".join(sorted(p["sites"])), method=method, budget=budget, reference=reference,
            instrument_hours=n, channel_observations=p["channels"],
            rmse=np.sqrt(p["sums"][0]/n) if n else np.nan,
            mae=p["sums"][1]/n if n else np.nan, bias=p["sums"][2]/n if n else np.nan,
            p90_absolute_error=weighted_quantile(values, weights)))
    ins = pd.DataFrame(instrument_rows); site = pd.DataFrame(site_rows)
    summary = []
    if len(ins):
        for (budget, method, reference), frame in ins.groupby(["budget", "method", "reference"], sort=True):
            s = site[(site.budget == budget) & (site.method == method) & (site.reference == reference)]
            s = s[s.instrument_hours > 0]
            valid = frame[frame.instrument_hours > 0]
            summary.append(dict(budget=int(budget), method=method, reference=reference,
                n_instruments=len(valid), n_sites=s.site.nunique(),
                instrument_hours=int(valid.instrument_hours.sum()),
                macro_instrument_rmse=valid.rmse.mean(), macro_instrument_mae=valid.mae.mean(),
                macro_instrument_bias=valid.bias.mean(),
                site_balanced_rmse=s.groupby("site").rmse.mean().mean(),
                site_balanced_mae=s.groupby("site").mae.mean().mean()))
    family = (ins[ins.instrument_hours > 0].groupby(["budget", "method", "reference", "family"], as_index=False)
              .agg(n_instruments=("instrument", "nunique"), instrument_hours=("instrument_hours", "sum"),
                   macro_instrument_rmse=("rmse", "mean"), macro_instrument_mae=("mae", "mean"),
                   macro_instrument_bias=("bias", "mean"))) if len(ins) else pd.DataFrame()
    site_aggregate = (site[site.instrument_hours > 0].groupby(["budget", "method", "reference", "site"], as_index=False)
              .agg(n_instruments=("instrument", "nunique"), instrument_hours=("instrument_hours", "sum"),
                   macro_instrument_rmse=("rmse", "mean"), macro_instrument_mae=("mae", "mean"),
                   macro_instrument_bias=("bias", "mean"))) if len(site) else pd.DataFrame()
    return dict(instrument_metrics=ins, site_instrument_metrics=site, aggregate_metrics=pd.DataFrame(summary),
                family_metrics=family, site_metrics=site_aggregate,
                daily_metrics=pd.DataFrame(period_rows["daily"]), monthly_metrics=pd.DataFrame(period_rows["monthly"]))


def _calendar_weights(times, count, block_hours, rng):
    """Circular blocks wrap within each calendar quarter, never across sites."""
    weights = np.zeros((count, len(times)), dtype=np.int32)
    quarters = times.to_period("Q")
    for quarter in quarters.unique():
        ix = np.flatnonzero(quarters == quarter)
        n = len(ix)
        length = min(block_hours, n)
        n_blocks = int(np.ceil(n/length))
        offsets = np.arange(length)
        for r in range(count):
            starts = rng.integers(0, n, size=n_blocks)
            sampled = ((starts[:, None]+offsets[None, :]) % n).reshape(-1)[:n]
            weights[r, ix] = np.bincount(sampled, minlength=n)
    return weights


def paired_calendar_bootstrap(bundles, n_replicates=1000, block_days=7,
                            seed=20260930, batch_size=64, max_attempt_factor=20):
    """Paired differences of macro instrument RMSE; methods must cover all sites.

    Replicates with zero sampled scored hours for ANY observed site-instrument
    are rejected for every method together, and rejection counts are returned.
    """
    if not bundles or n_replicates < 1 or block_days < 1:
        raise ValueError("Nonempty bundles, positive replicates and block days required")
    methods = bundles[0]["methods"]
    if any(b["methods"] != methods for b in bundles):
        raise ValueError("Bootstrap methods must be identically ordered at every site")
    identities = tuple(sorted({i for b in bundles for i in b["instruments"]}))
    index = {identity: i for i, identity in enumerate(identities)}
    rng = np.random.default_rng(seed)
    accepted_primary = []; accepted_site = []; attempted = rejected = accepted = 0
    observed_num = np.zeros((len(methods), len(identities))); observed_count = np.zeros(len(identities))
    observed_site = []
    for b in bundles:
        counts = b["active"].sum(axis=1)
        if np.any(counts == 0):
            raise ValueError("Bootstrap bundles must exclude instruments without scored hours")
        sums = b["se"].sum(axis=2)
        for j, identity in enumerate(b["instruments"]):
            observed_num[:, index[identity]] += sums[:, j]
            observed_count[index[identity]] += counts[j]
        observed_site.append(np.sqrt(sums/counts[None, :]).mean(axis=1))
    max_attempts = max_attempt_factor*n_replicates
    with threadpool_limits(limits=2):
        while accepted < n_replicates and attempted < max_attempts:
            size = min(batch_size, n_replicates-accepted, max_attempts-attempted)
            nums = np.zeros((size, len(methods), len(identities)))
            counts = np.zeros((size, len(identities)))
            site_values = []; valid = np.ones(size, bool)
            for b in bundles:
                w = _calendar_weights(b["times"], size, 24*block_days, rng).astype(float)
                n = w @ b["active"].T.astype(float)
                numerators = (w @ b["se"].reshape(-1, len(b["times"])).T).reshape(size, len(methods), -1)
                valid &= np.all(n > 0, axis=1)
                site_values.append(np.sqrt(numerators/np.maximum(n[:, None, :], 1)).mean(axis=2))
                for j, identity in enumerate(b["instruments"]):
                    nums[:, :, index[identity]] += numerators[:, :, j]
                    counts[:, index[identity]] += n[:, j]
            valid &= np.all(counts > 0, axis=1)
            primary = np.sqrt(nums/np.maximum(counts[:, None, :], 1)).mean(axis=2)
            secondary = np.stack(site_values).mean(axis=0)
            accepted_primary.append(primary[valid]); accepted_site.append(secondary[valid])
            attempted += size; rejected += int((~valid).sum()); accepted += int(valid.sum())
    primary = np.concatenate(accepted_primary, axis=0)
    secondary = np.concatenate(accepted_site, axis=0)
    if len(primary) == 0:
        raise ValueError("No valid bootstrap replicates; availability too sparse for this design")
    return dict(methods=methods, instruments=identities, primary=primary, secondary=secondary,
                observed_primary=np.sqrt(observed_num/observed_count[None, :]).mean(axis=1),
                observed_secondary=np.stack(observed_site).mean(axis=0),
                diagnostics=dict(seed=int(seed), block_days=int(block_days),
                    requested_replicates=int(n_replicates), accepted_replicates=int(len(primary)),
                    attempted_replicates=int(attempted), rejected_replicates=int(rejected),
                    complete=len(primary) == n_replicates,
                    strata="calendar quarters within each observed site",
                    scope="conditional on observed sites, physical instruments, scoring masks and quarter lengths"))


def bootstrap_tables(result, budget):
    rows = []; intervals = []; names = result["methods"]
    pairs = {(name, "periodic") for name in names if name != "periodic" and "periodic" in names}
    for variant in ("risk_independent_matched", "risk_fixed_matched"):
        if variant in names and "risk" in names:
            pairs.add((variant, "risk"))
    for model in ("hgb_stream", "ewrls", "bias_ewma"):
        if model in names and model+"_risk" in names:
            pairs.add((model+"_risk", model))
    # Engine comparisons are oriented dynamic AR minus immediately updated ML.
    for comparator in ("hgb_stream", "ewrls", "gp_stream", "gp_residual", "ridge_stream", "bias_ewma", "hgb_initial", "ridge_initial"):
        if comparator in names and "periodic" in names:
            pairs.add(("periodic", comparator))
    for metric, samples, observed in (("macro_instrument_rmse", result["primary"], result["observed_primary"]),
                                      ("site_balanced_rmse", result["secondary"], result["observed_secondary"])):
        for i, name in enumerate(names):
            ci = np.quantile(samples[:, i], [.025, .975])
            intervals.append(dict(budget=budget, method=name, metric=metric, estimate=observed[i],
                ci_low=ci[0], ci_high=ci[1], **{k: result["diagnostics"][k] for k in
                ("block_days", "accepted_replicates", "rejected_replicates", "seed")}))
        for left, right in sorted(pairs):
            a, b = names.index(left), names.index(right)
            ci = np.quantile(samples[:, a]-samples[:, b], [.025, .975])
            estimate = observed[a]-observed[b]
            rows.append(dict(budget=budget, method=left, comparator=right, metric=metric,
                difference=estimate, relative_difference_percent=100*estimate/observed[b] if observed[b] else np.nan,
                ci_low=ci[0], ci_high=ci[1], units="ppb", negative_favors="method",
                **{k: result["diagnostics"][k] for k in
                ("block_days", "accepted_replicates", "rejected_replicates", "seed")}))
    return pd.DataFrame(rows), pd.DataFrame(intervals)


def analyze_directory(source_dir=None, output_dir=None, *, n_bootstrap=1000,
                      seed=20260930, block_days=7, sensitivity_days=(14, 28),
                      sensitivity_replicates=300, required_sites=EXPECTED_SITES,
                      required_budgets=EXPECTED_BUDGETS, required_methods=None):
    source_dir = Path(source_dir or ROOT/"results/main")
    output_dir = Path(output_dir or ROOT/"results/tables")
    paths = sorted(source_dir.glob("*.npz"))
    runs = {}; ignored_files = []
    for path in paths:
        if not FILE_RE.match(path.name):
            ignored_files.append(path.name); continue
        run = load_run(path)
        key = (run.site, run.budget, run.method)
        if key in runs:
            raise ValueError(f"Duplicated run: {key}")
        runs[key] = run
    # The default formal gate includes all frozen baseline and audit-supplement
    # cells. Explicit required_methods retains the general cross-product API.
    expected = {(s, int(b), m) for s in required_sites for b in required_budgets
                for m in (required_methods if required_methods is not None else
                          EXPECTED_COMMON_METHODS + (EXPECTED_B48_ONLY_METHODS if b == 48 else ()))}
    missing = [dict(site=s, budget=b, method=m) for s, b, m in sorted(expected-set(runs))]
    unexpected = [dict(site=s, budget=b, method=m) for s, b, m in sorted(set(runs)-expected)]
    grouped = {}
    for (site, budget, method), run in runs.items():
        grouped.setdefault((site, budget), {})[method] = run
    unexpected_sites = sorted({s for s, _ in grouped}-set(required_sites))
    if unexpected_sites:
        raise ValueError(f"Unconfigured sites {unexpected_sites}; explicitly supply required_sites")
    bundles = []; optional_bundles = []; access = []; access_checks = []; subset_support = []
    for (site, budget), methods in sorted(grouped.items()):
        bundles.append(build_site_bundle(methods))
        matched_variants = ("risk_independent_matched", "risk_fixed_matched")
        acquisitions = {m: r for m, r in methods.items() if m in EXPECTED_METHODS+matched_variants}
        paid_counts = {m: int(r.query.sum()) for m, r in acquisitions.items()}
        if len(set(paid_counts.values())) > 1:
            raise ValueError(f"{site}, B{budget}: unequal additional attempted hours {paid_counts}")
        same_schedule_engines = ("noAR", "hgb_stream", "ewrls", "gp_stream", "gp_residual", "ridge_stream", "bias_ewma", "guard_removed", "weather_removed")
        checked = []
        if "periodic" in methods:
            for name in same_schedule_engines:
                if name in methods:
                    if not np.array_equal(methods[name].query, methods["periodic"].query):
                        raise ValueError(f"{site}, B{budget}: {name} does not share periodic reference schedule")
                    checked.append(name)
        risk_checked = []
        for name in ("hgb_stream_risk", "ewrls_risk", "bias_ewma_risk"):
            if name in methods and "risk" in methods:
                if not np.array_equal(methods[name].query, methods["risk"].query):
                    raise ValueError(f"{site}, B{budget}: {name} does not share risk reference schedule")
                risk_checked.append(name)
        access_checks.append(dict(site=site, budget=budget, equal_attempt_policy_counts=paid_counts,
                                  periodic_schedule_engine_checks=checked,
                                  risk_schedule_engine_checks=risk_checked,
                                  factory_zero_reference_exception="factory" in methods))
        for ref in ("duplicate", "no_warning", "clean_flags"):
            b = build_site_bundle(methods, reference=ref)
            if b is not None:
                optional_bundles.append(b)
                subset_support.append(dict(site=site, budget=budget, reference=ref, available=True, **b["support"]))
            else:
                subset_support.append(dict(site=site, budget=budget, reference=ref, available=False))
        for name, run in sorted(methods.items()):
            access.append(dict(site=site, budget=budget, method=name, hours=len(run.y),
                reference_access_role=("factory_zero_reference" if name == "factory" else
                                       "initialization_only" if not run.query.any() else "budgeted_reference"),
                initial_reference_attempts=int(run.initmask.sum()),
                additional_reference_attempts=int(run.query.sum()),
                total_reference_attempts=int((run.initmask | run.query).sum()),
                initial_finite_reference_hours=int((run.initmask & np.isfinite(run.y)).sum()),
                additional_finite_reference_hours=int((run.query & np.isfinite(run.y)).sum()),
                channel_scored_observations=int(run.score_masks.sum())))
    tables = metric_tables(bundles+optional_bundles)
    tables["reference_access"] = pd.DataFrame(access)
    contrast_frames = []; interval_frames = []; diagnostics = []; exclusions = []
    for budget in sorted({b for _, b in grouped}):
        sites = [s for s in required_sites if (s, budget) in grouped]
        absent_sites = [s for s in required_sites if (s, budget) not in grouped]
        if absent_sites:
            exclusions.append(dict(budget=budget, reason="missing required sites; no overall confidence intervals", sites=absent_sites))
            continue
        common = set.intersection(*(set(grouped[(s, budget)]) for s in sites)) if sites else set()
        if not common:
            exclusions.append(dict(budget=budget, reason="no methods common to all required sites")); continue
        for site in sites:
            omitted = sorted(set(grouped[(site, budget)])-common)
            if omitted:
                exclusions.append(dict(budget=budget, site=site, reason="method absent at another required site", methods=omitted))
        complete = [build_site_bundle(grouped[(s, budget)], methods=sorted(common)) for s in sites]
        if n_bootstrap:
            settings = [(block_days, n_bootstrap)] + [(d, sensitivity_replicates) for d in sensitivity_days if d != block_days and sensitivity_replicates]
            for days, replicates in settings:
                result = paired_calendar_bootstrap(complete, n_replicates=replicates,
                    block_days=days, seed=seed+10000*budget+days)
                contrasts, intervals = bootstrap_tables(result, budget)
                contrast_frames.append(contrasts); interval_frames.append(intervals)
                diagnostics.append(dict(budget=budget, sites=sites, methods=sorted(common), **result["diagnostics"]))
    tables["paired_rmse_differences"] = pd.concat(contrast_frames, ignore_index=True) if contrast_frames else pd.DataFrame()
    tables["bootstrap_metric_intervals"] = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()
    # Prevent an incomplete method's partial-population macro mean being mistaken
    # for an all-site comparison; retain its per-site/instrument diagnostics.
    aggregate = tables["aggregate_metrics"]
    if len(aggregate):
        aggregate["complete_required_site_inputs"] = [
            all((s, int(row.budget), row.method) in runs for s in required_sites)
            for row in aggregate.itertuples()]
    summary = dict(schema_version=1, source_directory=str(source_dir), output_directory=str(output_dir),
        base_seed=int(seed), bootstrap_seed_rule="base_seed + 10000*budget + block_days",
        input_files=[str(p) for p in paths if FILE_RE.match(p.name)], reference_access_checks=access_checks,
        prediction_files=len(runs), expected_prediction_files=len(expected),
        ignored_files=ignored_files, missing_required_inputs=missing, unexpected_inputs=unexpected,
        complete_required_inputs=not missing, matches_expected_input_matrix=not missing and not unexpected,
        bootstrap_exclusions=exclusions,
        bootstrap_diagnostics=diagnostics,
        primary_estimand="mean physical-instrument RMSE after averaging simultaneous-channel squared losses per instrument-hour",
        secondary_estimand="mean across sites of their mean physical-instrument RMSE",
        p90_definition="inverse weighted empirical CDF of channel absolute errors; each instrument-hour has total weight one",
        comparison_sign="method minus comparator; negative RMSE difference favors method",
        uncertainty_scope="conditional on observed sites/instruments and realized prediction streams; paired circular calendar blocks within each site-quarter; no retraining, geographic population inference or policy-learning uncertainty",
        duplicate_reference_role="frozen-prediction validation on QC-valid duplicate hours; not model selection",
        quality_flag_sensitivity="clean_flags excludes the supplied union of Warning, rebase and retrospective flags from evaluation only; no retraining and no change to primary metrics",
        optional_subset_support=subset_support,
        factory_reference_access="factory has zero initialization/additional reference cost; its accuracy comparison is not budget-matched calibration",
        no_p_values=True,
        aggregate_metrics=json.loads(aggregate.to_json(orient="records")) if len(aggregate) else [],
        paired_rmse_differences=json.loads(tables["paired_rmse_differences"].to_json(orient="records")) if len(tables["paired_rmse_differences"]) else [])
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir/f"{name}.csv", index=False)
    summary_path = output_dir.parent/"statistics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"results/main")
    parser.add_argument("--output", type=Path, default=ROOT/"results/tables")
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--sensitivity-replicates", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()
    summary = analyze_directory(args.source, args.output, n_bootstrap=args.replicates,
        sensitivity_replicates=args.sensitivity_replicates, seed=args.seed)
    print(json.dumps({k: summary[k] for k in ("prediction_files", "complete_required_inputs", "bootstrap_diagnostics")}, indent=2))
    print(f"Missing required inputs: {len(summary['missing_required_inputs'])}; tables: {args.output}")


if __name__ == "__main__":
    main()
