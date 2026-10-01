"""Immediately updated calibration baselines with strictly gated target access.

The caller supplies a complete hourly grid and the common guarded features.
Prediction at hour t always precedes acquisition of y[t]. Ridge/HGB/GP refit
immediately after every usable requested label. At a refit, the training set is
the acquired observations in the preceding 2160 calendar hours; HGB and GP use
at most the most recent 512 pairs. A model is retained between acquisitions.
EW-RLS forgets information each elapsed clock hour, including missing hours.
The inexpensive bias_ewma comparator holds a queried-residual offset between
queries on top of a frozen initialization Ridge prediction. gp_residual uses
that frozen Ridge as its mean, plus a GP for acquired residuals; its default
kernel amplitude is 0.1 on the initialization-standardized target scale.

No hyperparameter is optimized using held-out outcomes. GP hyperparameters are
fixed and operate on initialization-standardized features/targets. Runtime and
fit counts are exposed in LAST_RUN_DIAGNOSTICS, without changing the required
dict-of-prediction-arrays return interface.
"""
from __future__ import annotations

from pathlib import Path
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CORE_MODEL_NAMES = ("ridge_stream", "hgb_stream", "gp_stream", "ewrls")
MODEL_NAMES = CORE_MODEL_NAMES + ("bias_ewma", "gp_residual")
STREAM_NAMES = ("ridge_stream", "hgb_stream", "gp_stream", "gp_residual")
LAST_RUN_DIAGNOSTICS: dict = {}
RANDOM_STATE = 20260930
WINDOW_HOURS = 90 * 24


def _mask(value, n, name):
    a = np.asarray(value)
    if a.shape != (n,) or not np.all(np.isin(a, [0, 1])):
        raise ValueError(f"{name} must be a boolean mask of length {n}")
    return a.astype(bool, copy=True)


def fit_predict_stream_baselines(
    X, y, init_end, query_mask, valid_x=None,
    model_names=("ridge_stream", "hgb_stream", "gp_stream", "ewrls"),
    *, forgetting=.995, gp_length_scale=1., gp_noise=.1, max_pairs=512,
    bias_alpha=.2, gp_amplitude=None,
):
    """Return per-model hourly predictions; inspect diagnostics for actual costs.

    ``gp_noise`` is a fixed noise variance on the initial-standardized target
    scale; ``forgetting`` applies per clock hour, not per acquired label. Shared
    preprocessing time is recorded separately from per-model fit/predict time.
    ``max_pairs`` caps HGB/GP training only, while Ridge retains all acquired
    pairs within the 90-day calendar window.
    The four original baselines remain the default; bias_ewma and gp_residual
    are explicit optional model names. ``bias_alpha`` weights each new queried
    residual, without time decay between queries. The initial mean-model cost
    of those optional models is included in shared preprocessing diagnostics.
    """
    global LAST_RUN_DIAGNOSTICS
    total_start = perf_counter()
    features = np.asarray(X, dtype=float).copy()
    targets = np.asarray(y, dtype=float)
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError("X must have shape (T,p), p >= 1")
    n = len(features)
    if targets.shape != (n,):
        raise ValueError("y must have shape (T,)")
    if isinstance(init_end, (bool, np.bool_)) or not isinstance(init_end, (int, np.integer)) or not 1 <= init_end <= n:
        raise ValueError("init_end must be an integer in [1,T]")
    names = tuple(model_names)
    if not names or len(set(names)) != len(names) or not set(names) <= set(MODEL_NAMES):
        raise ValueError(f"model_names must be a nonempty unique subset of {MODEL_NAMES}")
    if not 0 < forgetting <= 1 or gp_length_scale <= 0 or gp_noise <= 0 or max_pairs < 2 or not 0 < bias_alpha <= 1 or (gp_amplitude is not None and gp_amplitude <= 0):
        raise ValueError("Invalid fixed baseline hyperparameters")
    requested = _mask(query_mask, n, "query_mask")
    applicable = np.ones(n, dtype=bool) if valid_x is None else _mask(valid_x, n, "valid_x")
    applicable &= np.isfinite(features[:, 0])
    features[~np.isfinite(features)] = np.nan
    predictions = {name: np.full(n, np.nan) for name in names}
    diagnostics = {name: dict(runtime_seconds=0., fits=0, updates=0,
                             max_training_pairs=0, max_training_age_hours=0)
                   for name in names}
    initial_rows = np.flatnonzero(applicable[:init_end])
    initial_pairs = initial_rows[np.isfinite(targets[initial_rows])]
    if len(initial_pairs) < 2:
        warnings.warn("Fewer than two initialization pairs", RuntimeWarning)
        LAST_RUN_DIAGNOSTICS = dict(models=diagnostics, total_seconds=perf_counter()-total_start)
        return predictions

    prep_start = perf_counter()
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    initial_imputed = imputer.fit_transform(features[initial_rows])
    scaler = StandardScaler().fit(initial_imputed)
    transformed = scaler.transform(imputer.transform(features))
    target_mean = float(np.mean(targets[initial_pairs]))
    target_scale = max(float(np.std(targets[initial_pairs])), 1.)
    frozen_mean = None
    if "bias_ewma" in names or "gp_residual" in names:
        with threadpool_limits(limits=2):
            fixed_ridge = Ridge(alpha=1., fit_intercept=True, solver="svd")
            fixed_ridge.fit(transformed[initial_pairs], targets[initial_pairs])
            frozen_mean = fixed_ridge.predict(transformed)
    prep_seconds = perf_counter() - prep_start
    models = {}

    def new_model(name):
        if name == "ridge_stream":
            return Ridge(alpha=1., fit_intercept=True, solver="svd")
        if name == "hgb_stream":
            return HistGradientBoostingRegressor(
                learning_rate=.05, max_iter=80, max_leaf_nodes=7,
                max_bins=64, min_samples_leaf=12, l2_regularization=1.,
                early_stopping=False, random_state=RANDOM_STATE,
            )
        amplitude = (.1 if name == "gp_residual" else 1.) if gp_amplitude is None else gp_amplitude
        return GaussianProcessRegressor(
            kernel=ConstantKernel(amplitude, constant_value_bounds="fixed")
                   * RBF(np.full(transformed.shape[1], gp_length_scale), length_scale_bounds="fixed")
                   + WhiteKernel(noise_level=gp_noise, noise_level_bounds="fixed"),
            alpha=1e-6, optimizer=None, normalize_y=False,
        )

    def fit_one(name, indices, next_hour):
        chosen = indices[-max_pairs:] if name in ("hgb_stream", "gp_stream", "gp_residual") else indices
        begin = perf_counter()
        model = new_model(name)
        yy = targets[chosen]
        if name == "gp_stream":
            yy = (yy-target_mean)/target_scale
        elif name == "gp_residual":
            yy = (yy-frozen_mean[chosen])/target_scale
        model.fit(transformed[chosen], yy)
        models[name] = model
        d = diagnostics[name]
        d["runtime_seconds"] += perf_counter()-begin
        d["fits"] += 1
        d["max_training_pairs"] = max(d["max_training_pairs"], len(chosen))
        d["max_training_age_hours"] = max(d["max_training_age_hours"], int(next_hour-chosen[0]))

    with threadpool_limits(limits=2):
        # All parameters and transforms use initialization information only.
        for name in names:
            if name in STREAM_NAMES:
                fit_one(name, initial_pairs, init_end)
        streaming_names = [name for name in names if name in STREAM_NAMES]
        observed = initial_pairs.tolist()
        # Fixed-model predictions are vectorized until the next requested hour.
        # These block boundaries use only the supplied schedule and x validity;
        # no target finiteness is inspected before the prediction is frozen.
        events = np.flatnonzero(requested & applicable & (np.arange(n) >= init_end))
        next_start = init_end
        for event in np.r_[events, n-1] if streaming_names else []:
            stop = int(event)+1
            if stop < next_start:
                continue
            current = np.flatnonzero(applicable[next_start:stop]) + next_start
            if len(current):
                for name in streaming_names:
                    begin = perf_counter()
                    pred = models[name].predict(transformed[current])
                    if name == "gp_stream":
                        pred = pred*target_scale+target_mean
                    elif name == "gp_residual":
                        pred = pred*target_scale+frozen_mean[current]
                    predictions[name][current] = pred
                    diagnostics[name]["runtime_seconds"] += perf_counter()-begin
            # A repeated final boundary does not cause a second acquisition.
            if event >= next_start and requested[event] and applicable[event] and np.isfinite(targets[event]):
                observed.append(int(event))
                oldest = stop-WINDOW_HOURS
                observed = [j for j in observed if j >= oldest]
                for name in streaming_names:
                    fit_one(name, np.asarray(observed, dtype=int), stop)
                    diagnostics[name]["updates"] += 1
            next_start = stop

        if "ewrls" in names:
            begin = perf_counter()
            phi = np.column_stack([np.ones(n), transformed])
            a = phi[initial_pairs]
            penalty = np.eye(a.shape[1])
            penalty[0, 0] = .01
            precision = a.T@a + penalty
            beta = np.linalg.solve(precision, a.T@targets[initial_pairs])
            covariance = np.linalg.inv(precision)
            updates = 0
            for t in range(init_end, n):
                covariance /= forgetting
                if not applicable[t]:
                    continue
                h = phi[t]
                predictions["ewrls"][t] = h@beta
                if requested[t] and np.isfinite(targets[t]):
                    ph = covariance@h
                    denom = 1.+h@ph
                    beta += ph*((targets[t]-h@beta)/denom)
                    covariance -= np.outer(ph, ph)/denom
                    covariance = .5*(covariance+covariance.T)
                    updates += 1
            diagnostics["ewrls"].update(runtime_seconds=perf_counter()-begin,
                                        fits=1, updates=updates,
                                        max_training_pairs=int(len(initial_pairs)),
                                        max_training_age_hours=None)
        if "bias_ewma" in names:
            begin = perf_counter()
            offset = 0.
            updates = 0
            for t in range(init_end, n):
                if not applicable[t]:
                    continue
                predictions["bias_ewma"][t] = frozen_mean[t]+offset
                if requested[t] and np.isfinite(targets[t]):
                    residual = targets[t]-frozen_mean[t]
                    offset = (1-bias_alpha)*offset + bias_alpha*residual
                    updates += 1
            diagnostics["bias_ewma"].update(runtime_seconds=perf_counter()-begin,
                                            fits=1, updates=updates,
                                            max_training_pairs=int(len(initial_pairs)),
                                            max_training_age_hours=None)
    LAST_RUN_DIAGNOSTICS = dict(
        models=diagnostics, shared_preprocess_seconds=prep_seconds,
        total_seconds=perf_counter()-total_start, initial_pairs=int(len(initial_pairs)),
        forgetting_per_hour=forgetting, gp_length_scale=gp_length_scale,
        gp_noise_variance=gp_noise, gp_target_mean=target_mean,
        gp_target_scale=target_scale, bounded_training_pairs=max_pairs,
        rolling_window_hours=WINDOW_HOURS,
        bias_alpha=bias_alpha, gp_amplitude_override=gp_amplitude,
        residual_gp_default_amplitude=.1,
        bias_and_residual_gp_mean="Frozen initial Ridge; its fit/predict cost is included in shared preprocessing",
        runtime_scope="Per-model fit/predict; shared preprocessing separately timed",
    )
    return predictions


def _self_test():
    begin = perf_counter()
    rng = np.random.default_rng(RANDOM_STATE)
    n, init_end = 240, 72
    X = rng.normal(size=(n, 3)); X[:, 0] += 10.
    y = 1.4*X[:, 0] + .3*X[:, 1] - .2*X[:, 2] + rng.normal(0, .1, n)
    X[3::31, 1] = np.nan
    X[15::47, 0] = np.nan
    query = np.zeros(n, bool); query[init_end+8::19] = True
    original = fit_predict_stream_baselines(X, y, init_end, query)

    def same(left, right, stop=n):
        for name in left:
            np.testing.assert_allclose(left[name][:stop], right[name][:stop], rtol=0, atol=0, equal_nan=True)

    hidden = (np.arange(n) >= init_end) & ~query
    altered = y.copy(); altered[hidden] = 1e7; altered[np.flatnonzero(hidden)[::2]] = np.nan
    same(original, fit_predict_stream_baselines(X, altered, init_end, query))
    event = init_end+8
    altered = y.copy(); altered[event] += 100
    updated = fit_predict_stream_baselines(X, altered, init_end, query)
    same(original, updated, stop=event+1)
    for name in ("ridge_stream", "ewrls"):
        assert abs(original[name][event+1]-updated[name][event+1]) > .01
    future = X.copy(); cutoff=init_end+42; future[cutoff:] = rng.normal(20, 4, (n-cutoff, 3))
    same(original, fit_predict_stream_baselines(future, y, init_end, query), stop=cutoff)
    for prediction in original.values():
        assert np.isnan(prediction[:init_end]).all()
        assert np.isnan(prediction[~np.isfinite(X[:, 0])]).all()
        assert np.isfinite(prediction[init_end:][np.isfinite(X[init_end:, 0])]).all()
    # The final-hour query must be acquired once, even when it is a sentinel.
    final_only = np.zeros(n, bool); final_only[-1] = True
    fit_predict_stream_baselines(X, y, init_end, final_only)
    assert all(v["updates"] == 1 for v in LAST_RUN_DIAGNOSTICS["models"].values())
    # Actual missing calendar hours remain in the axis; a 90-day window cannot
    # silently become the latest 2160 acquired samples.
    long_n = WINDOW_HOURS + init_end + 20
    long_X = rng.normal(size=(long_n, 3)); long_X[:, 0] += 10
    long_y = 1.4*long_X[:, 0] + rng.normal(0, .1, long_n)
    long_q = np.zeros(long_n, bool); long_q[init_end::97] = True; long_q[-2] = True
    fit_predict_stream_baselines(long_X, long_y, init_end, long_q)
    for name in CORE_MODEL_NAMES[:-1]:
        assert LAST_RUN_DIAGNOSTICS["models"][name]["max_training_age_hours"] <= WINDOW_HOURS
    extras = ("bias_ewma", "gp_residual")
    extra_base = fit_predict_stream_baselines(X, y, init_end, query, model_names=extras)
    altered = y.copy(); altered[hidden] = 1e7; altered[np.flatnonzero(hidden)[::2]] = np.nan
    same(extra_base, fit_predict_stream_baselines(X, altered, init_end, query, model_names=extras))
    altered = y.copy(); altered[event] += 100
    changed = fit_predict_stream_baselines(X, altered, init_end, query, model_names=extras)
    same(extra_base, changed, stop=event+1)
    assert abs(changed["bias_ewma"][event+1]-extra_base["bias_ewma"][event+1]) > 1.
    same(extra_base, fit_predict_stream_baselines(future, y, init_end, query, model_names=extras), stop=cutoff)
    print(f"PASS: synthetic hidden-label, same-hour, immediate-update, future-feature, final-query and calendar-age checks ({perf_counter()-begin:.2f}s)", flush=True)


def _development(names):
    # Only this CLI branch imports the real development data.
    from study_data import read_data, deployment_arrays, eligible, ROOT
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT/"cache/numba"))
    from calibration_engine import guard_features, periodic_mask, metrics
    data = read_data()
    data = data[(data.site == "Manchester") & (data.timestamp < "2021-01-01")]
    rows = []; begin = perf_counter()
    for key, group in data.groupby("deployment"):
        arr = deployment_arrays(group, weather=True)
        included, score = eligible(arr)
        if not included:
            continue
        X, guard = guard_features(arr["X"], width=8.)
        mask = periodic_mask(len(arr["y"]), 336, 48, starts=4, phase=.25)
        output = fit_predict_stream_baselines(X, arr["y"], 336, mask, model_names=names)
        diag = LAST_RUN_DIAGNOSTICS
        for name, prediction in output.items():
            result = metrics(arr["y"], np.maximum(prediction, 0.), score)
            assert result["n"] == int(score.sum())
            rows.append(dict(deployment=key, family=arr["family"], method=name,
                             **result, **diag["models"][name],
                             shared_preprocess_seconds=diag["shared_preprocess_seconds"],
                             bundle_seconds=diag["total_seconds"],
                             init_pairs=diag["initial_pairs"],
                             guard_width=8., budget=48, starts=4, block=672, phase=.25,
                             forgetting=.995, gp_length_scale=1., gp_noise=.1,
                             window_hours=WINDOW_HOURS, max_pairs=512,
                             query_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
        pd.DataFrame(rows).to_csv(ROOT/"results/stream_baseline_dev.csv", index=False)
        print(f"{key}: {diag['total_seconds']:.2f}s, elapsed={perf_counter()-begin:.2f}s", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby("method").agg(rmse=("rmse", "mean"), mae=("mae", "mean"),
                                      deployments=("deployment", "nunique"),
                                      compute_seconds=("runtime_seconds", "sum")).to_string(), flush=True)
    print(f"Completed development comparison in {perf_counter()-begin:.2f}s", flush=True)


def _development_grid(boundary=False):
    """Prespecified bounded tuning; reuse already computed default predictions."""
    from study_data import read_data, deployment_arrays, eligible, ROOT
    os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT/"cache/numba"))
    from calibration_engine import guard_features, periodic_mask, metrics
    data = read_data()
    data = data[(data.site == "Manchester") & (data.timestamp < "2021-01-01")]
    prepared = []
    for key, group in data.groupby("deployment"):
        arr = deployment_arrays(group, weather=True)
        included, score = eligible(arr)
        if included:
            X, _ = guard_features(arr["X"], width=8.)
            mask = periodic_mask(len(arr["y"]), 336, 48, starts=4, phase=.25)
            prepared.append((key, arr, X, mask, score))
    baseline = pd.read_csv(ROOT/"results/stream_baseline_dev.csv")
    assert set(baseline.deployment) == {a[0] for a in prepared}
    assert baseline.groupby("deployment").method.nunique().eq(4).all()
    destination = ROOT/"results/stream_baseline_dev_grid.csv"
    if boundary:
        previous = pd.read_csv(destination)
        gp_rows = previous[previous.method.eq("gp_stream")]
        best_json = gp_rows.groupby("parameters_json").rmse.mean().idxmin()
        best_gp = json.loads(best_json)
        configs = [("ewrls", {"forgetting": v}) for v in [.99, .97]]
        configs += [("bias_ewma", {"bias_alpha": v}) for v in [.05, .2, .5]]
        configs += [("gp_residual", {**best_gp, "gp_amplitude": .1})]
        replacing = {(name, json.dumps(params, sort_keys=True, separators=(",", ":")))
                     for name, params in configs}
        rows = [row for row in previous.to_dict(orient="records")
                if (row["method"], row["parameters_json"]) not in replacing]
    else:
        configs = [("ewrls", {"forgetting": v}) for v in [.995, .999, .9999]]
        configs += [("gp_stream", {"gp_length_scale": ell, "gp_noise": noise})
                    for ell in [.5, 1., 2.] for noise in [.1, .5]]
        rows = []
    begin = perf_counter()
    for name, params in configs:
        label = json.dumps(params, sort_keys=True, separators=(",", ":"))
        for key, arr, X, mask, score in prepared:
            reuse = (name == "ewrls" and params["forgetting"] == .995) or (
                name == "gp_stream" and params["gp_length_scale"] == 1. and params["gp_noise"] == .1)
            if reuse:
                prior = baseline[baseline.deployment.eq(key) & baseline.method.eq(name)].iloc[0]
                result = {field: prior[field] for field in ["n", "rmse", "mae", "bias", "p90"]}
                cost = {field: prior[field] for field in ["runtime_seconds", "fits", "updates", "max_training_pairs", "max_training_age_hours"]}
                shared = float(prior.shared_preprocess_seconds)
            else:
                pred = fit_predict_stream_baselines(X, arr["y"], 336, mask,
                                                    model_names=(name,), **params)[name]
                result = metrics(arr["y"], np.maximum(pred, 0.), score)
                cost = LAST_RUN_DIAGNOSTICS["models"][name]
                shared = LAST_RUN_DIAGNOSTICS["shared_preprocess_seconds"]
            assert int(result["n"]) == int(score.sum())
            rows.append(dict(deployment=key, family=arr["family"], method=name,
                             parameters_json=label, **params, **result, **cost,
                             shared_preprocess_seconds=shared, reused_base=reuse,
                             guard_width=8., init_hours=336, budget=48, block=672,
                             starts=4, phase=.25, window_hours=WINDOW_HOURS, max_pairs=512,
                             query_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                             selection_criterion="mean_instrument_RMSE_on_14_Manchester_pre2021_deployments"))
            pd.DataFrame(rows).to_csv(destination, index=False)
        subset = pd.DataFrame(rows)
        subset = subset[subset.method.eq(name) & subset.parameters_json.eq(label)]
        print(f"{name} {label}: macro RMSE={subset.rmse.mean():.6f}, MAE={subset.mae.mean():.6f}; elapsed={perf_counter()-begin:.2f}s", flush=True)
    frame = pd.DataFrame(rows)
    print(frame.groupby(["method", "parameters_json"]).agg(rmse=("rmse", "mean"),
                  mae=("mae", "mean"), deployments=("deployment", "nunique"),
                  model_seconds=("runtime_seconds", "sum")).to_string(), flush=True)
    print(f"Bounded grid completed in {perf_counter()-begin:.2f}s (base configurations reused)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--development-grid", action="store_true")
    parser.add_argument("--development-boundary", action="store_true")
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(CORE_MODEL_NAMES))
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    if args.development:
        _development(tuple(args.models))
    if args.development_grid:
        _development_grid()
    if args.development_boundary:
        _development_grid(boundary=True)
    if not any([args.self_test, args.development, args.development_grid, args.development_boundary]):
        parser.error("Use --self-test, --development, --development-grid, or --development-boundary")
