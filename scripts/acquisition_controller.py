"""Causal site-level access windows with a common calibration engine.

The risk objective is classical model-expected measurement-risk reduction,
not a claimed new statistical principle. Future contexts are historical
analogs only. Reference values are read only after frozen predictions and
an access decision. Models are copied; caller-owned fitted states are intact.
"""
from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
from numba import njit

from calibration_engine import _advance, _observe


POLICIES = {
    "risk": 0, "information_gain": 1, "ig": 1,
    "uncertainty": 2, "predictive_variance": 2,
    "risk_independent": 3, "classical_a": 3,
    "risk_fixed": 4, "window_uncertainty": 5,
}


@njit(cache=True)
def _cov_advance(P, F, Q):
    p = len(F)
    for i in range(p):
        for j in range(p):
            P[i, j] *= F[i] * F[j]
        P[i, i] += Q[i]


@njit(cache=True)
def _quadratic(P, h):
    out = 0.
    for i in range(len(h)):
        for j in range(len(h)):
            out += h[i] * P[i, j] * h[j]
    return max(out, 0.)


@njit(cache=True)
def _cov_observe(P, h, r):
    p = len(h)
    ph = np.zeros(p)
    for i in range(p):
        for j in range(p):
            ph[i] += P[i, j] * h[j]
    s = r
    for i in range(p):
        s += h[i] * ph[i]
    s = max(s, 1e-12)
    for i in range(p):
        for j in range(p):
            P[i, j] -= ph[i] * ph[j] / s
    return .5 * np.log(max(s / r, 1.))


@njit(cache=True)
def _window_utilities(phi, valid, ready, Pnow, Fs, Qs, rs, weights, initmask,
                      t, lengths, horizon, policy):
    """Exact covariance rollout under specified causal context scenarios."""
    nd, _, features = phi.shape
    p = features + 1
    utilities = np.zeros(len(lengths))
    active_weight = 0.
    # Preserve the H=48 default origins while keeping every scenario complete
    # and historical for longer horizons such as H=168.
    lag0 = max(72, horizon + 24)
    lag_values = (lag0, lag0 + 24, lag0 + 96)
    for d in range(nd):
        if t < ready[d] or not valid[d, t]:
            continue
        active_weight += weights[d]
        F = Fs[d].copy()
        Q = Qs[d].copy()
        if policy == 3:
            # Only the planner changes; actual prediction/update engine does not.
            Q[-1] = Q[-1] / max(1. - F[-1] * F[-1], 1e-6)
            F[-1] = 0.
        scenarios = 0
        for lag in lag_values:
            origin = t - lag
            if origin >= 0 and origin + horizon < t:
                scenarios += 1
        # Early replay: a frozen current context is a valid causal fallback.
        nscenarios = max(scenarios, 1)
        for scenario in range(nscenarios):
            origin = -1
            seen = 0
            for lag in lag_values:
                candidate = t - lag
                if candidate >= 0 and candidate + horizon < t:
                    if seen == scenario:
                        origin = candidate
                    seen += 1
            base = Pnow[d].copy()
            alternatives = np.empty((len(lengths), p, p))
            paid_counts = np.zeros(len(lengths), np.int64)
            for a in range(len(lengths)):
                alternatives[a] = Pnow[d]
            h = np.ones(p)
            for j in range(horizon + 1):
                if j > 0:
                    _cov_advance(base, F, Q)
                    for a in range(len(lengths)):
                        _cov_advance(alternatives[a], F, Q)
                for v in range(features):
                    # Every historical index accessed is strictly before t.
                    if j > 0 and origin >= 0 and valid[d, origin + j]:
                        h[v] = phi[d, origin + j, v]
                    else:
                        h[v] = phi[d, t, v]
                vb = _quadratic(base, h)
                # This is the known external initialization calendar, not
                # future sensor or reference availability. Free references
                # update BOTH compared paths after their current prediction.
                free_access = t + j < len(initmask) and initmask[t + j]
                for a in range(len(lengths)):
                    if (policy == 0 or policy == 3 or policy == 4) and j > 0:
                        va = _quadratic(alternatives[a], h)
                        utilities[a] += weights[d] * max(vb - va, 0.) / (horizon * nscenarios)
                    paid_access = not free_access and paid_counts[a] < lengths[a]
                    if free_access or paid_access:
                        if policy == 2 and paid_access:
                            utilities[a] += weights[d] * (vb + rs[d]) / nscenarios
                        ig = _cov_observe(alternatives[a], h, rs[d])
                        if policy == 1 and paid_access:
                            utilities[a] += weights[d] * ig / nscenarios
                        if paid_access:
                            paid_counts[a] += 1
                if free_access:
                    _cov_observe(base, h, rs[d])
    if active_weight:
        utilities /= active_weight
    return utilities


@njit(cache=True)
def _past_quantile(values, t, lookback, quantile, default):
    start = max(0, t - lookback)
    store = np.empty(t - start)
    count = 0
    for j in range(start, t):
        if np.isfinite(values[j]):
            store[count] = values[j]
            count += 1
    if count == 0:
        return default
    return np.quantile(store[:count], quantile)


@njit(cache=True)
def _fleet_fixed(phi, valid, ready, ms, Ps, Fs, Qs, rs, y, initmask, query):
    nd, n, features = phi.shape
    pred = np.full((nd, n), np.nan)
    variance = np.full((nd, n), np.nan)
    success = np.zeros(n, np.bool_)
    h = np.ones(features + 1)
    for t in range(n):
        for d in range(nd):
            if t < ready[d]:
                continue
            _advance(ms[d], Ps[d], Fs[d], Qs[d])
            if valid[d, t]:
                h[:-1] = phi[d, t]
                pred[d, t] = h @ ms[d]
                variance[d, t] = _quadratic(Ps[d], h) + rs[d]
        # No y access occurs until all current predictions have been frozen.
        if initmask[t] or query[t]:
            success[t] = np.isfinite(y[t])
            if success[t]:
                for d in range(nd):
                    if t >= ready[d] and valid[d, t]:
                        h[:-1] = phi[d, t]
                        _observe(ms[d], Ps[d], h, y[t], rs[d])
    return pred, variance, success


@njit(cache=True)
def _fleet_adaptive(phi, valid, ready, ms, Ps, Fs, Qs, rs, y, initmask,
                    weights, block_budgets, remaining_slots, origin, block, max_starts,
                    policy, horizon, threshold_scale, fixed_length):
    nd, n, features = phi.shape
    p = features + 1
    pred = np.full((nd, n), np.nan)
    variance = np.full((nd, n), np.nan)
    query = np.zeros(n, np.bool_)
    success = np.zeros(n, np.bool_)
    start_mask = np.zeros(n, np.bool_)
    forced = np.zeros(n, np.bool_)
    chosen_lengths = np.zeros(n, np.int64)
    candidate_scores = np.full(n, np.nan)
    uncertainty = np.full(n, np.nan)
    high_flags = np.zeros(n)
    spent = np.zeros(len(block_budgets), np.int64)
    starts = np.zeros(len(block_budgets), np.int64)
    h = np.ones(p)
    remaining_window = 0
    forced_window = False
    cooldown = False
    for t in range(n):
        active = 0
        active_weight = 0.
        current_uncertainty = 0.
        for d in range(nd):
            if t < ready[d]:
                continue
            _advance(ms[d], Ps[d], Fs[d], Qs[d])
            if valid[d, t]:
                h[:-1] = phi[d, t]
                pred[d, t] = h @ ms[d]
                variance[d, t] = _quadratic(Ps[d], h) + rs[d]
                current_uncertainty += weights[d] * variance[d, t]
                active_weight += weights[d]
                active += 1
        if active:
            uncertainty[t] = current_uncertainty / active_weight
            boundary = _past_quantile(uncertainty, t, 72, .75, uncertainty[t])
            high_flags[t] = 1. if uncertainty[t] > boundary else 0.

        if t >= origin:
            bindex = (t - origin) // block
            if (t - origin) % block == 0:
                remaining_window = 0
                forced_window = False
                cooldown = False
            remaining_budget = block_budgets[bindex] - spent[bindex]
            available_slots = remaining_slots[t]
            # Initial access is a supplied external schedule and is never
            # reclassified as an additional paid query.
            if not initmask[t] and remaining_budget > 0:
                execute = False
                must_catch_up = remaining_budget >= available_slots
                if remaining_window > 0:
                    execute = True
                elif must_catch_up:
                    remaining_window = remaining_budget
                    forced_window = True
                    chosen_lengths[t] = remaining_window
                    if starts[bindex] < max_starts:
                        start_mask[t] = True
                        starts[bindex] += 1
                    execute = True
                elif cooldown:
                    # Distinguish committed episodes: at least one unrequested
                    # non-initialization hour lies between completed windows.
                    cooldown = False
                elif starts[bindex] < max_starts and active:
                    if policy == 4 or policy == 5:
                        lengths = np.array([min(fixed_length, remaining_budget)])
                    else:
                        lengths = np.array([min(1, remaining_budget),
                                            min(3, remaining_budget),
                                            min(6, remaining_budget),
                                            min(12, remaining_budget),
                                            min(24, remaining_budget)])
                    # Acquisition utility is independent of unqueried y.
                    if policy == 5:
                        start72 = max(0, t - 71)
                        raw = np.mean(high_flags[start72:t + 1])
                        utilities = np.full(len(lengths), raw * lengths[0])
                    else:
                        utilities = _window_utilities(phi, valid, ready, Ps, Fs,
                                                      Qs, rs, weights, initmask, t, lengths,
                                                      horizon, policy)
                    startup_shadow = block_budgets[bindex] / max(max_starts, 1)
                    best = -1.
                    chosen = 0
                    for a in range(len(lengths)):
                        score = utilities[a] / (lengths[a] + startup_shadow)
                        if score > best:
                            best = score
                            chosen = a
                    candidate_scores[t] = best
                    rate = min(.95, max(.001, remaining_budget / max(available_slots, 1)))
                    threshold = _past_quantile(candidate_scores, t, 336,
                                               1. - rate, best)
                    if best >= threshold_scale * threshold and best > 0.:
                        ell = lengths[chosen]
                        if starts[bindex] == max_starts - 1:
                            # A final start cannot leave unspendable budget.
                            ell = remaining_budget
                            forced_window = ell != lengths[chosen]
                        else:
                            forced_window = False
                        remaining_window = min(ell, available_slots)
                        chosen_lengths[t] = remaining_window
                        start_mask[t] = True
                        starts[bindex] += 1
                        execute = True
                if execute:
                    query[t] = True
                    forced[t] = forced_window
                    spent[bindex] += 1
                    remaining_window -= 1
                    if remaining_window == 0:
                        cooldown = True
                        forced_window = False

        # Request success/value are first inspected here, after scoring/action.
        if initmask[t] or query[t]:
            success[t] = np.isfinite(y[t])
            if success[t]:
                for d in range(nd):
                    if t >= ready[d] and valid[d, t]:
                        h[:-1] = phi[d, t]
                        _observe(ms[d], Ps[d], h, y[t], rs[d])
    return (pred, variance, query, success, starts, start_mask, forced,
            chosen_lengths, candidate_scores, spent)


def _prepare(models, y, ready_indices, initialization_mask):
    if not models:
        raise ValueError("At least one device model is required")
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a nonempty common hourly vector")
    n = len(y)
    ready = np.asarray(ready_indices, dtype=np.int64)
    if ready.shape != (len(models),) or np.any(ready < 0) or np.all(ready >= n):
        raise ValueError("One nonnegative ready index per model is required; at least one must be < T")
    initmask = np.asarray(initialization_mask, dtype=bool)
    if initmask.shape != (n,):
        raise ValueError("initialization_mask must have length T")
    phi = np.stack([np.asarray(m["phi"], dtype=float) for m in models])
    valid = np.stack([np.asarray(m["valid"], dtype=bool) for m in models])
    if phi.ndim != 3 or phi.shape[1] != n or valid.shape != (len(models), n):
        raise ValueError("All models must share one complete hourly grid and feature dimension")
    valid &= np.isfinite(phi).all(axis=2)
    ms = np.stack([np.asarray(m["m"], dtype=float) for m in models]).copy()
    Ps = np.stack([np.asarray(m["P"], dtype=float) for m in models]).copy()
    Fs = np.stack([np.asarray(m["F"], dtype=float) for m in models])
    Qs = np.stack([np.asarray(m["Q"], dtype=float) for m in models])
    rs = np.asarray([m["r"] for m in models], dtype=float)
    dim = phi.shape[2] + 1
    if ms.shape != (len(models), dim) or Ps.shape != (len(models), dim, dim):
        raise ValueError("State dimensions must equal phi feature dimension + one AR residual")
    if Fs.shape != ms.shape or Qs.shape != ms.shape or np.any(rs <= 0):
        raise ValueError("Diagonal F/Q vectors and strictly positive observation variances are required")
    return phi, valid, ready, ms, Ps, Fs, Qs, rs, y, initmask


def _access_summary(result, initmask, y, runtime):
    result["all_query"] = result["query"] | initmask
    result["initialization_query"] = initmask.copy()
    result["initialization_attempts"] = int(initmask.sum())
    result["initialization_successes"] = int(result["success"][initmask].sum())
    result["additional_attempts"] = int(result["query"].sum())
    result["additional_successes"] = int((result["success"] & result["query"]).sum())
    result["runtime_seconds"] = runtime
    return result


def run_site_fixed(models, y, ready_indices, initialization_mask, query_mask):
    """Replay an externally supplied site mask, including free initialization.

    This is the nonadaptive common-engine interface for periodic/random/full
    access controls. All decisions are supplied; same-hour predictions remain
    pre-update. Additional mask entries during initialization are cleared.
    """
    started = perf_counter()
    packed = _prepare(models, y, ready_indices, initialization_mask)
    initmask = packed[-1]
    query = np.asarray(query_mask, dtype=bool).copy()
    if query.shape != initmask.shape:
        raise ValueError("query_mask must have length T")
    query &= ~initmask
    query[:int(np.min(packed[2]))] = False
    pred, variance, success = _fleet_fixed(*packed, query)
    start_mask = query & ~np.r_[False, query[:-1]]
    result = dict(pred=pred, var=variance, query=query, success=success,
                  start_mask=start_mask, starts=int(start_mask.sum()))
    return _access_summary(result, initmask, packed[-2], perf_counter() - started)


def run_site(models, y, ready_indices, initialization_mask, budget=48,
             block=672, max_starts=4, policy="risk", horizon=48,
             threshold_scale=1., fixed_length=12, weights=None):
    """Run shared site-reference acquisition with strict attempted-hour budgets.

    Initial calibration access is an externally supplied, known deployment
    schedule. Its future mask is used only to count remaining paid-access slots;
    future phi/valid/reference values never enter a score or feasibility test.
    Each full block receives `budget` extra request hours; the last partial
    block receives floor(budget * duration / block). A block with fewer
    noninitial hours than this target is marked infeasible via budget_shortfall.

    Adaptive starts choose a committed prefix length. The last available start
    consumes remaining budget; final catch-up is common to all policies. Forced
    reference attempts may have no valid device and are still charged. This is
    a transparent exact-budget replay, not an energy/redeployment-cost claim.
    """
    started = perf_counter()
    if policy not in POLICIES:
        raise ValueError(f"Unknown policy {policy!r}; choose from {tuple(POLICIES)}")
    if not (isinstance(block, int) and block > 0 and isinstance(budget, int) and 0 <= budget <= block):
        raise ValueError("Require integer 0 <= budget <= block with block > 0")
    if max_starts < 1 or horizon < 24 or fixed_length < 1 or threshold_scale <= 0:
        raise ValueError("Require max_starts >= 1, horizon >= 24, fixed_length >= 1, threshold_scale > 0")
    if fixed_length > horizon:
        raise ValueError("fixed_length may not exceed the common planning horizon")
    packed = _prepare(models, y, ready_indices, initialization_mask)
    if weights is None:
        weights = [m.get("weight", 1.) for m in models]
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(models),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("weights must contain one finite positive value per model")
    n = len(packed[-2])
    initmask = packed[-1]
    origin = int(np.min(packed[2]))
    nb = (n - origin + block - 1) // block
    target = np.zeros(nb, dtype=np.int64)
    feasible = np.zeros(nb, dtype=np.int64)
    remaining_slots = np.zeros(n, dtype=np.int64)
    for k in range(nb):
        left, right = origin + k * block, min(n, origin + (k + 1) * block)
        target[k] = int(np.floor(budget * (right - left) / block))
        slots = (~initmask[left:right]).astype(np.int64)
        remaining_slots[left:right] = np.cumsum(slots[::-1])[::-1]
        feasible[k] = min(target[k], int(slots.sum()))
    output = _fleet_adaptive(*packed, weights, feasible, remaining_slots, origin, block,
                              int(max_starts), POLICIES[policy], int(horizon),
                              float(threshold_scale), int(fixed_length))
    names = ("pred", "var", "query", "success", "starts", "start_mask",
             "forced_query", "chosen_lengths", "candidate_scores", "block_spent")
    result = dict(zip(names, output))
    result.update(policy=policy, block_origin=origin, block_target=target,
                  block_feasible=feasible, budget_shortfall=target-result["block_spent"])
    if not np.array_equal(result["block_spent"], feasible):
        raise RuntimeError("Internal scheduling error: feasible exact budget was not spent")
    if np.any(result["starts"] > max_starts):
        raise RuntimeError("Internal scheduling error: start cap exceeded")
    return _access_summary(result, initmask, packed[-2], perf_counter() - started)


def _self_test():
    """Synthetic checks; these are not results from the sensor study."""
    from calibration_engine import initialize
    rng = np.random.default_rng(20260930)
    n, initial = 510, 96
    X = rng.normal(size=(n, 3))
    X[:, 0] += 20
    y = 2 * X[:, 0] + .5 * X[:, 1] + rng.normal(0, 1, n)
    models = [initialize(X, y, initial), initialize(X * [1.1, 1, 1], y, initial)]
    initmask = np.arange(n) < initial
    initmask[270:280] = True
    settings = dict(budget=12, block=168, horizon=48, max_starts=4)
    started = perf_counter()
    for policy in ("risk", "information_gain", "uncertainty", "risk_independent", "risk_fixed", "window_uncertainty"):
        result = run_site(models, y, [initial, initial], initmask, policy=policy, **settings)
        assert np.array_equal(result["block_spent"], result["block_feasible"])
        assert np.all(result["starts"] <= 4)
        assert not np.any(result["query"] & initmask)
        assert np.isnan(result["pred"][:, :initial]).all()

    base = run_site(models, y, [initial, initial], initmask, **settings)
    hidden = ~base["all_query"]
    changed_y = y.copy()
    changed_y[hidden] = 1e7
    changed_y[np.flatnonzero(hidden)[::2]] = np.nan
    changed = run_site(models, changed_y, [initial, initial], initmask, **settings)
    np.testing.assert_array_equal(base["query"], changed["query"])
    np.testing.assert_allclose(base["pred"], changed["pred"], equal_nan=True, rtol=0, atol=0)

    at = int(np.flatnonzero(base["query"])[0])
    changed_y = y.copy()
    changed_y[at] += 100
    changed = run_site(models, changed_y, [initial, initial], initmask, **settings)
    np.testing.assert_allclose(base["pred"][:, :at+1], changed["pred"][:, :at+1], equal_nan=True, rtol=0, atol=0)
    assert np.nanmax(np.abs(base["pred"][:, at+1:] - changed["pred"][:, at+1:])) > .01

    cut = initial + 91
    altered_models = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()} for m in models]
    for m in altered_models:
        m["phi"][cut:] = rng.normal(20, 5, m["phi"][cut:].shape)
        m["valid"][cut::3] = False
    changed = run_site(altered_models, y, [initial, initial], initmask, **settings)
    np.testing.assert_array_equal(base["query"][:cut], changed["query"][:cut])
    np.testing.assert_allclose(base["pred"][:, :cut], changed["pred"][:, :cut], equal_nan=True, rtol=0, atol=0)

    fixed = run_site_fixed(models, y, [initial, initial], initmask, base["query"])
    np.testing.assert_allclose(base["pred"], fixed["pred"], equal_nan=True, rtol=0, atol=0)
    np.testing.assert_allclose(base["var"], fixed["var"], equal_nan=True, rtol=0, atol=0)

    # Analytic Gaussian case: one constant coefficient, no drift, prior var=1,
    # observation var=1. This checks the actual criterion and its t+1 origin,
    # not just consistency between two copies of the same replay loop.
    plain_phi = np.ones((1, 350, 1))
    plain_valid = np.ones((1, 350), dtype=np.bool_)
    plain_P = np.array([[[1., 0.], [0., 0.]]])
    plain_F = np.array([[1., 0.]])
    plain_Q = np.zeros((1, 2))
    lens = np.array([1, 3, 6, 12, 24])
    args = (plain_phi, plain_valid, np.array([0]), plain_P, plain_F,
            plain_Q, np.ones(1), np.ones(1), np.zeros(350, dtype=np.bool_),
            200, lens, 48)
    risk = _window_utilities(*args, 0)
    expected_risk = np.array([np.mean([1 - 1/(1 + min(j, ell)) for j in range(1, 49)]) for ell in lens])
    np.testing.assert_allclose(risk, expected_risk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_window_utilities(*args, 1), .5 * np.log1p(lens), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_window_utilities(*args, 2), 2 * lens, rtol=1e-12, atol=1e-12)

    # Known free observations must reduce both paths' uncertainty, and a paid
    # prefix must resume after free clock hours rather than lose paid labels.
    free = np.zeros(350, dtype=np.bool_)
    free[[202, 203, 205]] = True
    free_lens = np.array([1, 3, 6])
    free_args = (plain_phi, plain_valid, np.array([0]), plain_P, plain_F,
                 plain_Q, np.ones(1), np.ones(1), free, 200, free_lens, 12)
    expected_free_risk = []
    expected_free_ig = []
    expected_free_uncertainty = []
    for ell in free_lens:
        count_base = count_alt = count_paid = 0
        risk_sum = ig_sum = uncertainty_sum = 0.
        for j in range(13):
            vbase, valt = 1 / (1 + count_base), 1 / (1 + count_alt)
            if j > 0:
                risk_sum += (vbase - valt) / 12
            if free[200 + j]:
                count_base += 1
                count_alt += 1
            elif count_paid < ell:
                ig_sum += .5 * np.log1p(valt)
                uncertainty_sum += vbase + 1
                count_alt += 1
                count_paid += 1
        expected_free_risk.append(risk_sum)
        expected_free_ig.append(ig_sum)
        expected_free_uncertainty.append(uncertainty_sum)
    np.testing.assert_allclose(_window_utilities(*free_args, 0), expected_free_risk, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_window_utilities(*free_args, 1), expected_free_ig, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_window_utilities(*free_args, 2), expected_free_uncertainty, rtol=1e-12, atol=1e-12)

    # Exact budgets remain honest when an entire block is free initialization;
    # impossible extra spending must become a reported shortfall.
    crowded_init = initmask.copy()
    crowded_init[initial:initial+168] = True
    no_slots = run_site(models, y, [initial, initial], crowded_init, **settings)
    assert no_slots["block_spent"][0] == 0
    assert no_slots["budget_shortfall"][0] == 12

    # Replicating a physical device as two identical half-weighted channels
    # cannot alter its aggregate utility or selected schedule.
    one = run_site([models[0]], y, [initial], initmask, weights=[1.], **settings)
    duplicated = run_site([models[0], models[0]], y, [initial, initial], initmask,
                          weights=[.5, .5], **settings)
    np.testing.assert_array_equal(one["query"], duplicated["query"])
    np.testing.assert_allclose(one["pred"][0], duplicated["pred"][0], rtol=0, atol=0, equal_nan=True)

    # Long-horizon contexts remain fully historical rather than silently
    # becoming future paths when the planning horizon is increased.
    long_settings = dict(settings, horizon=168)
    long_base = run_site(models, y, [initial, initial], initmask, **long_settings)
    long_models = [{k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in m.items()} for m in models]
    long_cut = 350
    for m in long_models:
        m["phi"][long_cut:] += 100
        m["valid"][long_cut::2] = False
    long_changed = run_site(long_models, y, [initial, initial], initmask, **long_settings)
    np.testing.assert_array_equal(long_base["query"][:long_cut], long_changed["query"][:long_cut])
    np.testing.assert_allclose(long_base["pred"][:, :long_cut], long_changed["pred"][:, :long_cut], rtol=0, atol=0, equal_nan=True)
    print(f"PASS synthetic site controller: six policies, exact budgets/start caps/shortfall, hidden/current-label and future-feature invariance, fixed-mask parity, analytic risk/IG including free-calendar updates, instrument weights ({perf_counter()-started:.2f}s including JIT)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Import run_site/run_site_fixed, or use --self-test")
    _self_test()
