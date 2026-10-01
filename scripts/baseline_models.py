"""Causal empirical calibration baselines on a complete hourly grid.

All returned predictions at time t precede any use of the reference at t.
The feature layout is current low-cost NO2 response first, then optional
temperature/RH/other contemporaneously available covariates. Initialization
uses indices [0, init_end); subsequent labels are available only where the
supplied query_mask is true. No acquisition policy is implemented here.

Run the embedded, synthetic-only leakage checks with::

    python baseline_models.py --self-test
"""

from __future__ import annotations

import argparse
from time import perf_counter
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


MODEL_NAMES = (
    "ridge_initial",
    "hgb_initial",
    "ridge_rolling90d_daily",
    "hgb_rolling90d_daily",
)
ROLLING_HOURS = 90 * 24
RETRAIN_HOURS = 24
RANDOM_STATE = 20260930


def _ridge() -> Ridge:
    # Feature scaling is fitted on initialization and then frozen.
    return Ridge(alpha=1.0, fit_intercept=True, solver="svd")


def _hgb() -> HistGradientBoostingRegressor:
    # No hidden random validation split or early stopping on a time series.
    return HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=80,
        max_leaf_nodes=7,
        max_bins=64,
        min_samples_leaf=12,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def _boolean_mask(value: ArrayLike, length: int, name: str) -> NDArray[np.bool_]:
    raw = np.asarray(value)
    if raw.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {raw.shape}")
    if raw.dtype != np.bool_:
        if not np.all(np.isin(raw, [0, 1])):
            raise ValueError(f"{name} must contain only bool or 0/1 values")
    return raw.astype(bool, copy=True)


def fit_predict_baselines(
    X: ArrayLike,
    y: ArrayLike,
    init_end: int,
    query_mask: ArrayLike,
    valid_x: ArrayLike | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Fit four bounded-cost calibration baselines and replay reference access.

    Parameters
    ----------
    X : array, shape (T, p)
        Unstandardized current features. Column zero must be the low-cost
        response; rows without that response remain unpredicted. Infinities
        are treated as missing. The time axis MUST be a complete hourly grid,
        including missing calendar hours, because no timestamps are supplied.
    y : array, shape (T,)
        Paired reference values. Only initial labels and queried later labels
        are used. Missing references at requested times do not update a model.
    init_end : int
        Exclusive end of initial calibration. For 14 calendar days on a
        complete hourly grid this is 336. Initialization predictions are NaN.
    query_mask : array, shape (T,)
        Request indicator from the common acquisition policy. At t it can
        affect only later predictions. Initial entries are ignored.
    valid_x : optional bool array, shape (T,)
        Additional device-applicability mask; must not be selected using hidden
        reference values. A finite current response is required in all cases.

    Returns
    -------
    dict
        Four float arrays of length T; see MODEL_NAMES. Static models retain
        their initial fit. Rolling models refit every 24 clock hours starting
        from init_end, with acquired labels from the preceding 2160 clock
        hours. Preprocessing is fitted on initial valid feature rows and frozen.
        If no usable label remains in the rolling window, the most recent
        fitted model is retained (explicit fallback, not an extra label query).
        Fewer than two initial usable pairs yields all NaNs and a warning.

    Notes
    -----
    Predicting a 24-hour block at once is equivalent to hourly prediction
    between fixed refits: feature transforms and coefficients are already
    fixed and do not use future rows when constructing any earlier prediction.
    The query indicators for that block are processed only AFTER all its
    predictions have been frozen. HGB/BLAS work is capped at two threads.
    """
    features = np.asarray(X, dtype=np.float64).copy()
    reference = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        raise ValueError("X must have shape (T, p), with p >= 1")
    n = features.shape[0]
    if reference.shape != (n,):
        raise ValueError(f"y must have shape ({n},), got {reference.shape}")
    if isinstance(init_end, (bool, np.bool_)) or not isinstance(init_end, (int, np.integer)):
        raise ValueError("init_end must be an integer index")
    if not 1 <= init_end <= n:
        raise ValueError("init_end must satisfy 1 <= init_end <= len(X)")
    requests = _boolean_mask(query_mask, n, "query_mask")
    applicable = (
        np.ones(n, dtype=bool)
        if valid_x is None
        else _boolean_mask(valid_x, n, "valid_x")
    )
    applicable &= np.isfinite(features[:, 0])
    features[~np.isfinite(features)] = np.nan
    predictions = {name: np.full(n, np.nan, dtype=np.float64) for name in MODEL_NAMES}

    # Inspect reference finiteness only where the label is genuinely available.
    initial_indices = np.flatnonzero(applicable[:init_end])
    initial_pairs = initial_indices[np.isfinite(reference[initial_indices])]
    if len(initial_pairs) < 2:
        warnings.warn(
            "Fewer than two usable initialization pairs: all baseline predictions are NaN",
            RuntimeWarning,
            stacklevel=2,
        )
        return predictions

    # No later covariate, even an unlabeled one, contributes to preprocessing.
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    initial_imputed = imputer.fit_transform(features[initial_indices])
    scaler = StandardScaler().fit(initial_imputed)
    # Transform is row-wise with frozen parameters; batch execution is causal.
    transformed = scaler.transform(imputer.transform(features))
    observed_indices = initial_pairs.tolist()
    initial_targets = reference[initial_pairs]

    with threadpool_limits(limits=2):
        initial_ridge = _ridge().fit(transformed[initial_pairs], initial_targets)
        initial_hgb = _hgb().fit(transformed[initial_pairs], initial_targets)
        rolling_ridge = initial_ridge
        rolling_hgb = initial_hgb
        previous_fit_indices = initial_pairs.copy()

        score_indices = np.flatnonzero(applicable & (np.arange(n) >= init_end))
        if len(score_indices):
            predictions["ridge_initial"][score_indices] = initial_ridge.predict(transformed[score_indices])
            predictions["hgb_initial"][score_indices] = initial_hgb.predict(transformed[score_indices])

        for t in range(init_end, n, RETRAIN_HOURS):
            end = min(t + RETRAIN_HOURS, n)
            # observed_indices contains ONLY previous acquired labels, j < t.
            past = np.asarray(observed_indices, dtype=np.int64)
            recent = past[past >= max(0, t - ROLLING_HOURS)]
            if len(recent) and not np.array_equal(recent, previous_fit_indices):
                rolling_ridge = _ridge().fit(transformed[recent], reference[recent])
                rolling_hgb = _hgb().fit(transformed[recent], reference[recent])
                previous_fit_indices = recent.copy()

            current_indices = t + np.flatnonzero(applicable[t:end])
            if len(current_indices):
                current_X = transformed[current_indices]
                predictions["ridge_rolling90d_daily"][current_indices] = rolling_ridge.predict(current_X)
                predictions["hgb_rolling90d_daily"][current_indices] = rolling_hgb.predict(current_X)

            # Reveal labels only after every prediction in this fixed-model
            # block is frozen. They can enter the next scheduled refit only.
            acquired = t + np.flatnonzero(requests[t:end] & applicable[t:end])
            available = acquired[np.isfinite(reference[acquired])]
            observed_indices.extend(available.tolist())

    return predictions


def _assert_predictions_equal(
    left: dict[str, NDArray[np.float64]],
    right: dict[str, NDArray[np.float64]],
    stop: int | None = None,
) -> None:
    for name in MODEL_NAMES:
        np.testing.assert_allclose(left[name][:stop], right[name][:stop], rtol=0, atol=0, equal_nan=True)


def _self_test() -> None:
    """Meaningful leakage checks on clearly identified synthetic data only."""
    started = perf_counter()
    rng = np.random.default_rng(RANDOM_STATE)
    n, init_end = 440, 96
    X = rng.normal(size=(n, 3))
    X[:, 0] += 20
    y = 1.8 * X[:, 0] + 0.6 * X[:, 1] - 0.4 * X[:, 2] + rng.normal(0, 0.1, n)
    X[11::19, 1] = np.nan
    X[17::41, 0] = np.nan
    query = np.zeros(n, dtype=bool)
    query[init_end + 5::31] = True
    query[init_end + 24] = True  # Query exactly at a scheduled refit boundary.
    baseline = fit_predict_baselines(X, y, init_end, query)
    for value in baseline.values():
        assert value.shape == (n,)
        assert np.isnan(value[:init_end]).all()
        assert np.isnan(value[~np.isfinite(X[:, 0])]).all()

    # Hidden values and their missingness may change arbitrarily: no prediction
    # can change, including predictions made at acquired timestamps.
    hidden = (np.arange(n) >= init_end) & ~query
    changed_hidden = y.copy()
    changed_hidden[hidden] = 1e6 + np.arange(n)[hidden]
    changed_hidden[np.flatnonzero(hidden)[::2]] = np.nan
    _assert_predictions_equal(baseline, fit_predict_baselines(X, changed_hidden, init_end, query))

    # A label at a refit boundary is not available for that hour's refit, nor
    # for the other predictions before the next scheduled daily refit.
    reveal_at = init_end + 24
    changed_visible = y.copy()
    changed_visible[reveal_at] += 500
    changed = fit_predict_baselines(X, changed_visible, init_end, query)
    _assert_predictions_equal(baseline, changed, stop=reveal_at + RETRAIN_HOURS)
    for name in ("ridge_initial", "hgb_initial"):
        np.testing.assert_allclose(baseline[name], changed[name], rtol=0, atol=0, equal_nan=True)
    later = slice(reveal_at + RETRAIN_HOURS, n)
    assert np.nanmax(np.abs(baseline["ridge_rolling90d_daily"][later] - changed["ridge_rolling90d_daily"][later])) > 0.01

    # Future features and validity cannot affect an earlier prediction, even
    # though a fixed-model block is predicted using vectorized operations.
    cutoff = init_end + 55
    changed_X = X.copy()
    changed_X[cutoff:] = rng.normal(100, 20, (n - cutoff, 3))
    _assert_predictions_equal(baseline, fit_predict_baselines(changed_X, y, init_end, query), stop=cutoff)

    # A feature absent throughout initialization retains its position and has
    # a frozen zero-fill; later availability cannot fit its imputation value.
    empty_initial_feature = X.copy()
    empty_initial_feature[:init_end, 2] = np.nan
    empty_result = fit_predict_baselines(empty_initial_feature, y, init_end, query)
    assert all(np.isfinite(v[init_end:][np.isfinite(X[init_end:, 0])]).all() for v in empty_result.values())

    # Calendar-window expiry: after 90 elapsed days, changed initialization
    # targets must no longer affect refitted rolling models. Keeping only the
    # last 2160 acquired rows would fail this sparse-query test.
    long_n = init_end + ROLLING_HOURS + RETRAIN_HOURS + 5
    long_X = rng.normal(size=(long_n, 3))
    long_X[:, 0] += 20
    long_y = 2 * long_X[:, 0] + long_X[:, 1] + rng.normal(0, 0.1, long_n)
    long_query = np.zeros(long_n, dtype=bool)
    long_query[init_end + 13::48] = True
    long_original = fit_predict_baselines(long_X, long_y, init_end, long_query)
    long_changed_y = long_y.copy()
    long_changed_y[:init_end] += 1000
    long_changed = fit_predict_baselines(long_X, long_changed_y, init_end, long_query)
    expiry = init_end + ROLLING_HOURS
    for name in ("ridge_rolling90d_daily", "hgb_rolling90d_daily"):
        np.testing.assert_allclose(long_original[name][expiry:], long_changed[name][expiry:], rtol=0, atol=0, equal_nan=True)

    print(f"PASS: synthetic hidden-label, same-hour, future-feature, calendar-window, missingness and shape checks ({perf_counter()-started:.2f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test, or import fit_predict_baselines")
    _self_test()
