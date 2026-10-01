"""Development-only calibration comparison under one fixed query schedule.

Never accesses 2021+ outcomes. All models use the same input deployment grids,
initial 336 hours, and 48-hour/four-window schedule per 672-hour block. Raw
factory output, four empirical baselines, and two common-engine variants are
scored with the same post-initialization pairs, after clipping predictions at 0.
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
from calibration_engine import initialize, periodic_mask, run_fixed, metrics
from study_data import read_data, deployment_arrays, eligible


def main():
    started = perf_counter()
    data = read_data()
    data = data[(data.site == "Manchester") & (data.timestamp < "2021-01-01")]
    rows = []
    destination = ROOT / "results/development_baselines.csv"
    for key, group in data.groupby("deployment"):
        arr = deployment_arrays(group, weather=True)
        included, score = eligible(arr, init_end=336, minimum=672)
        if not included:
            continue
        X, y = arr["X"], arr["y"]
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
        )

        def record(name, prediction, seconds, scope, rho=np.nan):
            result = metrics(y, np.maximum(prediction, 0.), score)
            assert result["n"] == int(score.sum()), (key, name, result["n"], int(score.sum()))
            rows.append(dict(**base, method=name, **result, rho=rho,
                             runtime_seconds=seconds, runtime_scope=scope))

        record("factory_out_of_box", X[:, 0], 0., "no_fitting")
        begin = perf_counter()
        predictions = fit_predict_baselines(X, y, 336, mask)
        elapsed = perf_counter() - begin
        for name, prediction in predictions.items():
            record(name, prediction, elapsed, "joint_four_baseline_bundle")
        for ar in [False, True]:
            begin = perf_counter()
            model = initialize(X, y, 336, q_scale=1e-4, use_ar=ar)
            prediction, _ = run_fixed(model, y, mask)
            elapsed = perf_counter() - begin
            record(f"common_engine_q1e-4_ar{int(ar)}", prediction,
                   elapsed, "single_engine_fit_and_replay", rho=model["rho"])
        frame = pd.DataFrame(rows)
        # Checkpoint complete deployments; no other task owns this output file.
        frame.to_csv(destination, index=False)
        print(f"{key}: n={score.sum()}, four-baseline bundle={base['grid_hours']}h/{rows[-3]['runtime_seconds']:.2f}s, total_elapsed={perf_counter()-started:.2f}s", flush=True)
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
