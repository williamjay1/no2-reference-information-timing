"""Causal hourly sensor correction. Targets are exposed only at query times.

Initialization is handled before replay. All predictions precede target access.
The numba kernel never uses an unqueried target in state or policy calculations.
"""
import numpy as np
from numba import njit


def guard_features(X, init_end=336, width=12.):
    """Bound only the main sensor input using initialization-only support.

    Keeps every scorable hour. This limits extrapolation, not outlier removal.
    Weather has separate documented physical screens in the data loader.
    """
    Z=np.asarray(X,dtype=float).copy()
    a=Z[:init_end,0]
    center=float(np.nanmedian(a))
    mad=1.4826*float(np.nanmedian(np.abs(a-center)))
    span=max(width*mad,30.)
    Z[:,0]=np.clip(Z[:,0],center-span,center+span)
    return Z,dict(center=center,mad_scale=mad,width=width,span=span)


def initialize(X, y, init_end, q_scale=1e-4, ar_share=0.8,
               ridge=1.0, use_ar=True):
    X = np.asarray(X, dtype=float)
    train = X[:init_end]
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    mean = med.copy()
    sd = np.nanstd(train, axis=0)
    sd = np.maximum(np.where(np.isfinite(sd), sd, 1.), [1., 1., 5.][:X.shape[1]])
    # Missing weather does not discard otherwise valid pollutant measurements.
    Z = np.where(np.isfinite(X), X, med)
    Z = (Z - mean) / sd
    phi = np.column_stack([np.ones(len(X)), Z])
    valid = np.isfinite(X[:, 0])
    ok = valid[:init_end] & np.isfinite(y[:init_end])
    if ok.sum() < 72:
        raise ValueError('Fewer than 72 initial calibration pairs')
    a = phi[:init_end][ok]
    yy = y[:init_end][ok]
    penalty = np.eye(a.shape[1]) * ridge
    penalty[0, 0] = ridge * .01
    precision = a.T @ a + penalty
    beta = np.linalg.solve(precision, a.T @ yy)
    residual = y[:init_end] - phi[:init_end] @ beta
    variance = max(float(np.mean(residual[ok]**2)), 1.)
    pairs = ok[1:] & ok[:-1]
    rho = 0.
    if use_ar and pairs.sum() >= 24:
        prev, cur = residual[:-1][pairs], residual[1:][pairs]
        denom = prev @ prev
        rho = float(np.clip(prev @ cur / max(denom, 1e-8), 0., .95))
    dim = a.shape[1] + 1
    m = np.r_[beta, 0.]
    P = np.zeros((dim, dim))
    inflation = min((1 + rho) / max(1 - rho, .05), 20.)
    P[:-1, :-1] = variance * inflation * np.linalg.inv(precision)
    P[-1, -1] = variance * ar_share if use_ar else 0.
    F = np.ones(dim)
    F[-1] = rho
    Q = np.zeros(dim)
    Q[:-1] = variance * q_scale * np.r_[1., np.full(dim-2, .05)]
    Q[-1] = variance * ar_share * (1-rho*rho) if use_ar else 0.
    r = max(variance * (1-ar_share if use_ar else 1.), .2)
    return dict(phi=phi, valid=valid, m=m, P=P, F=F, Q=Q, r=r,
                variance=variance, rho=rho, init_pairs=int(ok.sum()),
                med=med, scale=sd, init_end=int(init_end))


@njit(cache=True)
def _advance(m, P, F, Q):
    d = len(m)
    for i in range(d):
        m[i] *= F[i]
        for j in range(d):
            P[i, j] *= F[i]*F[j]
        P[i, i] += Q[i]


@njit(cache=True)
def _observe(m, P, h, y, r):
    ph = P @ h
    S = r + h @ ph
    innovation = y - h @ m
    m += ph * (innovation / S)
    P -= np.outer(ph, ph) / S
    # Equivalent rank-one Joseph update for the scalar observation.
    P[:] = .5*(P+P.T)


@njit(cache=True)
def replay_fixed(phi, valid, y, mask, m0, P0, F, Q, r, init_end):
    n = len(y)
    pred = np.full(n, np.nan)
    var = np.full(n, np.nan)
    m, P = m0.copy(), P0.copy()
    for t in range(init_end, n):
        _advance(m, P, F, Q)
        if valid[t]:
            h = np.empty(len(m))
            h[:-1] = phi[t]
            h[-1] = 1.
            pred[t] = h @ m
            var[t] = h @ P @ h + r
            if mask[t] and np.isfinite(y[t]):
                _observe(m, P, h, y[t], r)
    return pred, var


def periodic_mask(n, init_end, budget=48, block=672, starts=4, phase=0):
    mask = np.zeros(n, dtype=bool)
    for origin in range(init_end, n, block):
        count = min(block, n-origin)
        b = int(np.floor(budget*count/block))
        k = min(starts, b)
        if k < 1:
            continue
        lengths = np.full(k, b//k, dtype=int)
        lengths[:b%k] += 1
        for j, ell in enumerate(lengths):
            start = origin + int((j+phase)*count/k)
            start = min(max(start, origin), origin+count-ell)
            mask[start:start+ell] = True
    return mask


def random_mask(n, init_end, budget=48, block=672, starts=4, seed=1):
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    for origin in range(init_end, n, block):
        count = min(block, n-origin)
        b = int(np.floor(budget*count/block))
        k = min(starts, b)
        if k < 1:
            continue
        lengths = np.full(k, b//k, dtype=int)
        lengths[:b%k] += 1
        # Random phase in each disjoint bin prevents overlaps and excess starts.
        for j, ell in enumerate(lengths):
            low = origin + int(j*count/k)
            high = origin + int((j+1)*count/k) - ell
            start = int(rng.integers(low, max(low, high)+1))
            mask[start:start+ell] = True
    return mask


def run_fixed(model, y, mask):
    return replay_fixed(model['phi'], model['valid'], np.asarray(y, float),
                        np.asarray(mask, bool), model['m'], model['P'],
                        model['F'], model['Q'], model['r'], model['init_end'])


def metrics(y, pred, score_mask):
    ok = score_mask & np.isfinite(y) & np.isfinite(pred)
    e = pred[ok]-y[ok]
    if not len(e):
        return dict(n=0, rmse=np.nan, mae=np.nan, bias=np.nan, p90=np.nan)
    return dict(n=len(e), rmse=float(np.sqrt(np.mean(e*e))),
                mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
                p90=float(np.quantile(np.abs(e), .9)))
