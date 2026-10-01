"""Fair development comparison after the common initialization-only input guard.

Every calibrated model receives guard_features(width=8) output. Factory output
retains the original input. No observations are excluded by the guard, and no
2021+ performance is examined. The four baseline methods share an implementation
and their measured bundle runtime is explicitly identified rather than invented
per-method timing.
"""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / "cache/numba"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from time import perf_counter
import hashlib
import numpy as np
import pandas as pd

from baseline_models import fit_predict_baselines
from calibration_engine import initialize, periodic_mask, run_fixed, metrics, guard_features
from study_data import read_data, deployment_arrays, eligible


def main():
    started = perf_counter()
    data = read_data()
    data = data[(data.site == "Manchester") & (data.timestamp < "2021-01-01")]
    rows = []
    destination = ROOT / "results/development_guard_baselines.csv"
    for key, group in data.groupby("deployment"):
        arr = deployment_arrays(group, weather=True)
        included, score = eligible(arr, init_end=336, minimum=672)
        if not included:
            continue
        original, y = arr["X"], arr["y"]
        X, guard = guard_features(original, init_end=336, width=8.)
        clipped = np.isfinite(original[:, 0]) & (original[:, 0] != X[:, 0])
        assert np.array_equal(np.isfinite(X), np.isfinite(original))
        mask = periodic_mask(len(y), 336, budget=48, starts=4, phase=.25)
        base = dict(
            deployment=key, family=arr["family"],
            grid_start=str(arr["times"][0]), grid_end=str(arr["times"][-1]),
            grid_hours=len(y), init_hours=336,
            init_pairs=int((np.isfinite(X[:336, 0]) & np.isfinite(y[:336])).sum()),
            scored_pairs=int(score.sum()), weather=True,
            requested_hours=int(mask.sum()),
            acquired_pairs=int((mask & np.isfinite(X[:, 0]) & np.isfinite(y)).sum()),
            mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            budget=48, block=672, starts=4, phase=.25,
            guard_center=guard["center"], guard_mad_scale=guard["mad_scale"],
            guard_width=guard["width"], guard_span=guard["span"],
            guard_count_all=int(clipped.sum()), guard_count_init=int(clipped[:336].sum()),
            guard_count_scored=int((clipped & score).sum()),
            score_x_min=float(original[score, 0].min()),
            score_x_median=float(np.median(original[score, 0])),
            score_x_p99=float(np.quantile(original[score, 0], .99)),
            score_x_max=float(original[score, 0].max()),
            score_y_min=float(y[score].min()),
            score_y_median=float(np.median(y[score])),
            score_y_p99=float(np.quantile(y[score], .99)),
            score_y_max=float(y[score].max()),
        )

        def record(name, prediction, seconds, scope, guarded=True, q=np.nan, rho=np.nan):
            result = metrics(y, np.maximum(prediction, 0.), score)
            assert result["n"] == int(score.sum()), (key, name, result["n"], int(score.sum()))
            rows.append(dict(**base, method=name, **result, rho=rho, q_scale=q,
                             guarded=guarded, runtime_seconds=seconds, runtime_scope=scope))

        record("factory_out_of_box", original[:, 0], 0., "no_fitting", guarded=False)
        begin = perf_counter()
        predictions = fit_predict_baselines(X, y, 336, mask)
        baseline_seconds = perf_counter() - begin
        for name, prediction in predictions.items():
            record(name, prediction, baseline_seconds, "joint_four_baseline_bundle")
        for ar in [False, True]:
            begin = perf_counter()
            model = initialize(X, y, 336, q_scale=1e-3, use_ar=ar, ar_share=.8)
            prediction, _ = run_fixed(model, y, mask)
            elapsed = perf_counter() - begin
            record(f"common_engine_q1e-3_ar{int(ar)}", prediction, elapsed,
                   "single_engine_fit_and_replay", q=1e-3, rho=model["rho"])
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(f"{key}: n={score.sum()}, guard_scored={int((clipped & score).sum())}, baseline_bundle={baseline_seconds:.2f}s, elapsed={perf_counter()-started:.2f}s", flush=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No development deployments passed availability criteria")
    summary = frame.groupby("method").agg(
        macro_rmse=("rmse", "mean"), macro_mae=("mae", "mean"),
        deployments=("deployment", "nunique"),
    ).sort_values("macro_rmse")
    print(summary.to_string(), flush=True)
    print(f"Completed {frame.deployment.nunique()} development deployments in {perf_counter()-started:.2f}s; {destination}", flush=True)


if __name__ == "__main__":
    main()
