"""Frozen held-out baseline runs using the root experiment's exact label access.

This script does not select parameters or evaluate comparative performance.
Run only after the root has frozen the protocol and generated periodic/risk
source NPZs. --run-frozen and a frozen JSON manifest are required explicitly.
Resume is permitted only when manifest, inputs, code and settings all match.
"""
from __future__ import annotations

from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT/"cache/numba"))
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import hashlib
import json
import sys
from time import perf_counter, sleep
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import competitive_stream_baselines as streaming
from calibration_engine import guard_features
from site_data import make_site

STREAM_METHODS = ("ridge_stream", "hgb_stream", "gp_stream", "ewrls", "bias_ewma")
RISK_METHODS = ("hgb_stream", "ewrls", "bias_ewma")
STATIC_METHODS = ("ridge_initial", "hgb_initial", "factory")
PARAMETERS = dict(init_hours=336, guard_width=8., q_scale=.001, ar_share=.8,
                  weather=True, gp_length_scale=2., gp_noise=.5, forgetting=.995,
                  bias_alpha=.05, max_pairs=512, window_hours=2160,
                  prediction_floor=0., stage="test")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _identity(manifest):
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("The frozen manifest must be a JSON object")
    for key in ("gp_length_scale", "gp_noise", "forgetting", "bias_alpha", "max_pairs"):
        if payload.get("baseline", {}).get(key) != PARAMETERS[key]:
            raise ValueError(f"Baseline parameter {key} differs from the frozen manifest")
    for key in ("q_scale", "ar_share", "weather", "guard_width"):
        if payload.get("calibrator", {}).get(key) != PARAMETERS[key]:
            raise ValueError(f"Calibrator parameter {key} differs from the frozen manifest")
    code = ["run_main_baselines.py", "competitive_stream_baselines.py",
            "calibration_engine.py", "site_data.py", "study_data.py"]
    data = ["no2_hourly_oob.parquet", "reference_no2_ratified.parquet"]
    return dict(manifest_sha256=_hash(manifest), parameters=PARAMETERS,
                code_sha256={name: _hash(ROOT/"scripts"/name) for name in code},
                data_sha256={name: _hash(ROOT/"datasets"/name) for name in data},
                runtime=dict(python=sys.version.split()[0], numpy=np.__version__,
                             pandas=pd.__version__, sklearn=sklearn.__version__))


def _load_source(path, fleet):
    deadline = perf_counter()+180.
    announced = False
    while not path.exists():
        if not announced:
            print(f"Waiting up to 180s for root source: {path.name}", flush=True)
            announced = True
        if perf_counter() >= deadline:
            raise FileNotFoundError(f"Root experiment output required first: {path}")
        sleep(2.)
    with np.load(path, allow_pickle=False) as archive:
        source = {name: archive[name].copy() for name in archive.files}
    required = {"pred", "y", "times", "score_masks", "query", "initmask", "metadata_json"}
    if not required.issubset(source):
        raise ValueError(f"{path}: missing {sorted(required-set(source))}")
    text = np.asarray(source["metadata_json"]).item()
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    metadata = json.loads(text)
    times = pd.DatetimeIndex(pd.to_datetime(source["times"].astype(str)))
    if not times.equals(fleet["times"]):
        raise ValueError(f"{path}: source and frozen fleet times differ")
    if not np.array_equal(source["y"], fleet["y"], equal_nan=True):
        raise ValueError(f"{path}: source and independent reference cache differ")
    by_key = {row["deployment"]: j for j, row in enumerate(fleet["metadata"])}
    if len(metadata) != len(by_key) or len({r["deployment"] for r in metadata}) != len(metadata):
        raise ValueError(f"{path}: inconsistent deployment count or duplicates")
    order = [by_key[row["deployment"]] for row in metadata]
    if not np.array_equal(source["score_masks"], fleet["score_masks"][order]):
        raise ValueError(f"{path}: source and frozen scoring masks differ")
    if not np.array_equal(source["initmask"], fleet["initialization_mask"]):
        raise ValueError(f"{path}: initialization calendar differs")
    if np.any(source["query"].astype(bool) & source["initmask"].astype(bool)):
        raise ValueError(f"{path}: additional access overlaps setup access")
    local = []
    for row, j in zip(metadata, order):
        expected = fleet["metadata"][j]
        for key in ("site", "deployment", "instrument", "device", "family", "start", "end", "local_offset"):
            if row[key] != expected[key]:
                raise ValueError(f"{path}: deployment metadata mismatch for {key}")
        start = int(row["local_offset"])
        stop = int(times.searchsorted(pd.Timestamp(row["end"]), side="right"))
        if start < 0 or stop > len(times) or stop-start < PARAMETERS["init_hours"]:
            raise ValueError("Invalid physical deployment span")
        if times[start] != pd.Timestamp(row["start"]) or times[stop-1] != pd.Timestamp(row["end"]):
            raise ValueError("Deployment endpoints are not exact source hours")
        original = np.asarray(fleet["Xs"][j][start:stop], float)
        guarded, _ = guard_features(original, init_end=336, width=8.)
        if not np.array_equal(np.isfinite(original), np.isfinite(guarded)):
            raise ValueError("Guard altered feature availability")
        local.append(dict(row=row, start=start, stop=stop, original=original,
                          X=guarded, valid=np.isfinite(original[:, 0])))
    if source["pred"].shape != (len(local), len(times)):
        raise ValueError("Source prediction shape and metadata disagree")
    return source, local


def _provenance(identity, source_path, source_hash, method):
    return dict(**identity, method=method, source_npz=source_path.name,
                source_npz_sha256=source_hash)


def _resumable(path, provenance, overwrite):
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            prior = json.loads(np.asarray(saved["baseline_provenance_json"]).item())
            complete = not np.any(saved["score_masks"].astype(bool) & ~np.isfinite(saved["pred"]))
        if prior == provenance and complete:
            print(f"Verified resume: {path.name}", flush=True)
            return True
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        pass
    if not overwrite:
        raise RuntimeError(f"Existing output has incompatible provenance: {path}. Use --overwrite only for an intended recomputation.")
    return False


def _shared_arrays(source):
    keep = {"y", "times", "score_masks", "query", "initmask", "metadata_json"}
    # Copy source quality and duplicate-reference masks, never an AR variance
    # or policy-specific state trajectory as if it belonged to a new model.
    keep.update(key for key in source if key.startswith(("duplicate", "y_duplicate", "quality_", "sensor_quality", "sensor_flag"))
                or key == "warning_mask")
    return {key: source[key].copy() for key in keep if key in source}


def _save(path, source, prediction, provenance, records, method):
    if np.any(source["score_masks"].astype(bool) & ~np.isfinite(prediction)):
        raise ValueError(f"{method}: missing predictions on the common scoring mask")
    output = _shared_arrays(source)
    output["pred"] = np.maximum(prediction, 0.)
    if method in STATIC_METHODS:
        output["query"] = np.zeros_like(source["query"], dtype=bool)
    if method == "factory":
        output["initmask"] = np.zeros_like(source["initmask"], dtype=bool)
    runtime = float(sum(row["runtime_seconds"]+row.get("shared_preprocess_seconds", 0.) for row in records))
    finite_reference = np.isfinite(output["y"])
    output["runtime_seconds"] = np.asarray(runtime)
    output["manifest_sha256"] = np.asarray(provenance["manifest_sha256"])
    output["access_json"] = np.asarray(_json(dict(
        initialization_attempts=int(output["initmask"].sum()),
        initialization_successes=int((output["initmask"] & finite_reference).sum()),
        additional_attempts=int(output["query"].sum()),
        additional_successes=int((output["query"] & finite_reference).sum()),
        runtime_seconds=runtime,
        runtime_scope="Per-method compute plus its shared feature-preprocessing time; excludes NPZ IO")))
    output["baseline_provenance_json"] = np.asarray(_json(provenance))
    output["runtime_json"] = np.asarray(_json(records))
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output)
    temporary.replace(path)
    log = path.with_name(path.stem+"_runtime.json")
    log.write_text(json.dumps(dict(provenance=provenance, deployments=records),
                              indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {path.name}", flush=True)


def _static(X, y, valid, methods):
    begin = perf_counter()
    initial = np.flatnonzero(valid[:336])
    pairs = initial[np.isfinite(y[initial])]
    if len(pairs) < 72:
        raise ValueError("Static baseline has fewer than 72 initial pairs")
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    fitted = imputer.fit_transform(X[initial])
    scaler = StandardScaler().fit(fitted)
    transformed = scaler.transform(imputer.transform(X))
    preprocessing_seconds = perf_counter()-begin
    selected = np.flatnonzero(valid & (np.arange(len(y)) >= 336))
    output, costs = {}, {}
    with threadpool_limits(limits=2):
        for name in methods:
            begin = perf_counter()
            if name == "ridge_initial":
                model = Ridge(alpha=1., fit_intercept=True, solver="svd")
            else:
                model = HistGradientBoostingRegressor(
                    learning_rate=.05, max_iter=80, max_leaf_nodes=7,
                    max_bins=64, min_samples_leaf=12, l2_regularization=1.,
                    early_stopping=False, random_state=20260930)
            model.fit(transformed[pairs], y[pairs])
            p = np.full(len(y), np.nan)
            if len(selected):
                p[selected] = model.predict(transformed[selected])
            output[name] = p
            costs[name] = dict(runtime_seconds=perf_counter()-begin,
                               shared_preprocess_seconds=preprocessing_seconds,
                               initial_pairs=len(pairs), fits=1, updates=0)
    return output, costs


def _run_group(site, budget, policy, fleet, identity, directory, overwrite):
    source_path = directory/f"{site}_B{budget}_{policy}.npz"
    source, local = _load_source(source_path, fleet)
    source_manifest = np.asarray(source.get("manifest_sha256", "")).item()
    if source_manifest != identity["manifest_sha256"]:
        raise ValueError(f"{source_path}: source was generated under a different frozen manifest")
    digest = _hash(source_path)
    names = STREAM_METHODS if policy == "periodic" else RISK_METHODS
    mapped = {name: (name if policy == "periodic" else name+"_risk") for name in names}
    if policy == "periodic" and budget == 48:
        mapped.update({name: name for name in STATIC_METHODS})
    pending, provenance = {}, {}
    for name, output_name in mapped.items():
        path = directory/f"{site}_B{budget}_{output_name}.npz"
        expected = _provenance(identity, source_path, digest, output_name)
        if not _resumable(path, expected, overwrite):
            pending[name] = path
            provenance[name] = expected
    if not pending:
        return
    predictions = {name: np.full(source["pred"].shape, np.nan) for name in pending}
    records = {name: [] for name in pending}
    access = source["query"].astype(bool) | source["initmask"].astype(bool)
    for j, item in enumerate(local):
        start, stop = item["start"], item["stop"]
        y = np.asarray(source["y"][start:stop], float)
        query = access[start:stop]
        stream_names = tuple(name for name in names if name in pending)
        if stream_names:
            result = streaming.fit_predict_stream_baselines(
                item["X"], y, 336, query, valid_x=item["valid"],
                model_names=stream_names, forgetting=.995, gp_length_scale=2.,
                gp_noise=.5, max_pairs=512, bias_alpha=.05)
            diagnostics = streaming.LAST_RUN_DIAGNOSTICS
            for name, values in result.items():
                predictions[name][j, start:stop] = values
                records[name].append(dict(deployment=item["row"]["deployment"],
                    **diagnostics["models"][name],
                    shared_preprocess_seconds=diagnostics["shared_preprocess_seconds"],
                    bundle_seconds=diagnostics["total_seconds"],
                    clock_hours=stop-start,
                    label_access_after_initialization=int(query[336:].sum())))
        static_names = tuple(name for name in ("ridge_initial", "hgb_initial") if name in pending)
        if static_names:
            result, costs = _static(item["X"], y, item["valid"], static_names)
            for name, values in result.items():
                predictions[name][j, start:stop] = values
                records[name].append(dict(deployment=item["row"]["deployment"], **costs[name]))
        if "factory" in pending:
            begin = perf_counter()
            predictions["factory"][j, start:stop] = item["original"][:, 0]
            records["factory"].append(dict(deployment=item["row"]["deployment"],
                runtime_seconds=perf_counter()-begin, fits=0, updates=0, reference_access=0))
        print(f"{site} B{budget} {policy}: {item['row']['deployment']} complete", flush=True)
    for name, path in pending.items():
        _save(path, source, predictions[name], provenance[name], records[name], mapped[name])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-frozen", action="store_true")
    parser.add_argument("--sites", nargs="+", choices=("Manchester", "London", "York"),
                        default=["Manchester", "London", "York"])
    parser.add_argument("--budgets", nargs="+", type=int, choices=(12, 24, 48), default=[12, 24, 48])
    parser.add_argument("--directory", type=Path, default=ROOT/"results/main")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.run_frozen:
        parser.error("Explicit --run-frozen authorization is required after protocol freeze")
    if args.directory.drive.upper() != "D:":
        raise ValueError("Prediction outputs must remain on D:")
    if not args.manifest.is_file():
        raise FileNotFoundError(args.manifest)
    identity = _identity(args.manifest)
    args.directory.mkdir(parents=True, exist_ok=True)
    begin = perf_counter()
    for site in args.sites:
        fleet = make_site(site, stage="test", q_scale=.001, use_ar=True,
                          weather=True, ar_share=.8, guard_width=8.)
        for budget in args.budgets:
            _run_group(site, budget, "periodic", fleet, identity, args.directory, args.overwrite)
        if 48 in args.budgets:
            _run_group(site, 48, "risk", fleet, identity, args.directory, args.overwrite)
    print(f"Frozen baseline runs completed in {perf_counter()-begin:.2f}s", flush=True)


if __name__ == "__main__":
    main()
