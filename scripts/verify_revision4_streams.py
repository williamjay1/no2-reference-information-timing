"""Read-only, relocatable reaggregation and small code-level revision-4 checks.

Reads only --project, writes only the explicitly requested --receipt. Absolute
paths in prior provenance are identifiers remapped inside --project. No model
refitting, bootstrap generation, experiment runner import, or manifest editing.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np
import pandas as pd

SITES = ("Manchester", "London", "York")
CELLS = ("adaptive_persistent", "adaptive_white", "frozen_persistent", "frozen_white")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inside(project, identifier):
    """Resolve an old absolute identifier by its project-relative suffix."""
    parts = list(PureWindowsPath(str(identifier)).parts)
    markers = {"scripts", "results", "datasets", "evidence", "manuscript", "cache"}
    if PureWindowsPath(str(identifier)).is_absolute() or Path(str(identifier)).is_absolute():
        ix = next((i for i, value in enumerate(parts) if value in markers), None)
        if ix is None:
            raise ValueError(f"No relocatable project suffix: {identifier}")
        parts = parts[ix:]
    candidate = project.joinpath(*parts).resolve()
    if not candidate.is_relative_to(project):
        raise ValueError(f"Path escapes supplied project: {identifier}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Package missing {candidate.relative_to(project)}")
    return candidate


def close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11,
                               err_msg=label)


def load_stream(path):
    with np.load(path, allow_pickle=False) as archive:
        out = {key: archive[key].copy() for key in
               ("pred", "y", "times", "score_masks", "metadata_json", "query", "initmask")}
        out["provenance"] = {}
        for key in ("revision_provenance_json", "revision4_provenance_json"):
            if key in archive:
                out["provenance"] = json.loads(str(archive[key].item()))
        if "block_spent" in archive:
            out["block_spent"] = archive["block_spent"].copy()
        if "budget_shortfall" in archive:
            out["budget_shortfall"] = archive["budget_shortfall"].copy()
        if "starts" in archive:
            assert np.max(archive["starts"]) <= 4
    out["metadata"] = json.loads(str(out.pop("metadata_json").item()))
    pred, y, mask = out["pred"], out["y"], out["score_masks"]
    assert mask.shape == pred.shape == (len(out["metadata"]), len(y))
    assert np.isfinite(pred[mask]).all() and np.isfinite(np.broadcast_to(y, pred.shape)[mask]).all()
    assert not np.any(out["query"] & out["initmask"])
    if "block_spent" in out:
        assert int(out["block_spent"].sum()) == int(out["query"].sum())
        assert not out["budget_shortfall"].any()
        times = out["times"]
        origin = min(int(np.flatnonzero(times == m["ready"])[0]) for m in out["metadata"])
        per_block = [int(out["query"][origin+672*b:origin+672*(b+1)].sum())
                     for b in range(len(out["block_spent"]))]
        np.testing.assert_array_equal(per_block, out["block_spent"])
    return out


def aggregate(runs, scope="all_scored__all"):
    records = []
    for site, stream in runs.items():
        mask = stream["score_masks"].copy()
        if scope.startswith("gap_ge168h"):
            access = stream["query"] | stream["initmask"]
            ends = np.flatnonzero(access & ~np.r_[access[1:], False])
            # The last right-edge episode is unfinished, matching the source definition.
            ends = ends[ends < len(access) - 1]
            last = np.full(len(access), -1, dtype=int)
            last[ends] = ends
            last = np.maximum.accumulate(last)
            # Last completed hour is visible at the following output hour.
            previous = np.r_[-1, last[:-1]]
            age = np.arange(len(access)) - previous
            mask &= ((previous >= 0) & (age >= 168) & ~access)[None, :]
        ids = sorted({m["instrument"] for m in stream["metadata"]})
        for instrument in ids:
            idx = [i for i, m in enumerate(stream["metadata"]) if m["instrument"] == instrument]
            family = stream["metadata"][idx[0]]["family"]
            if scope.endswith("__long_Zep") and family != "Zep":
                continue
            if scope.endswith("__short_multi_product") and family == "Zep":
                continue
            selected = mask[idx]
            count = selected.sum(axis=0)
            active = count > 0
            assert active.any(), (site, instrument, scope)
            residual = np.where(selected, stream["pred"][idx] - stream["y"], 0.)
            hourly_se = np.square(residual).sum(axis=0) / np.maximum(count, 1)
            records.append(dict(site=site, instrument=instrument, family=family,
                                instrument_hours=int(active.sum()),
                                channel_observations=int(selected.sum()),
                                rmse=float(np.sqrt(hourly_se[active].mean()))))
    frame = pd.DataFrame(records)
    return dict(n_instruments=len(frame), instrument_hours=int(frame.instrument_hours.sum()),
                macro_instrument_rmse=float(frame.rmse.mean()),
                site_equal_rmse=float(frame.groupby("site").rmse.mean().mean())), frame


def extracts(path, names, namespace=None):
    """Execute only named numerical functions, without runner top-level side effects."""
    namespace = {"np": np} if namespace is None else namespace
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            node.decorator_list = []
            body.append(node)
    assert {node.name for node in body} == set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return namespace


def smoke(project):
    controller = project / "scripts/revision4_delay_controller.py"
    env = extracts(project / "scripts/calibration_engine.py", ("_advance", "_observe"))
    env = extracts(controller, ("_cov_advance", "_quadratic", "_cov_observe", "_delay_utilities", "_fleet_fixed"), env)
    rng = np.random.default_rng(404)
    n, t, delay, horizon, p = 800, 500, 72, 168, 5
    phi = rng.normal(size=(1, n, p - 1)); valid = np.ones((1, n), bool)
    a = rng.normal(size=(p, p)); prior = (a @ a.T)[None]
    f = np.array([[1., 1., 1., 1., .8]]); q = np.array([[.01, .01, .01, .01, .2]])
    noise = np.array([.4]); setup = np.zeros(n, bool); query = setup.copy()
    query[440:452] = True; setup[470:476] = True; setup[510:517] = True
    got = env["_delay_utilities"](phi, valid, np.array([50]), prior, f, q, noise, np.ones(1),
                                   setup, query, t, np.array([12]), horizon, delay)[0]
    expected = 0.; transition = np.diag(f[0]); project_delay = np.diag(f[0] ** delay)
    def condition(cov, h):
        ph = cov @ h
        return cov - np.outer(ph, ph) / (.4 + h @ ph)
    for lag in (horizon + 24, horizon + 48, horizon + 120):
        origin = t - lag; base = prior[0].copy(); alt = base.copy(); paid = 0
        for j in range(horizon + 1):
            if j:
                base = transition @ base @ transition.T + np.diag(q[0])
                alt = transition @ alt @ transition.T + np.diag(q[0])
            h_output = np.r_[phi[0, origin + j] if j else phi[0, t], 1.]
            if j:
                expected += max(float(h_output @ project_delay @ (base-alt) @ project_delay.T @ h_output), 0.) / (3*horizon)
            event = t-delay+j
            h_event = np.r_[phi[0, event] if event < t else phi[0, origin-delay+j] if event > t else phi[0, t], 1.]
            common = setup[event] or (event < t and query[event])
            candidate = event >= t and not setup[event] and paid < 12
            if common: base = condition(base, h_event)
            if common or candidate: alt = condition(alt, h_event)
            if candidate: paid += 1
    close(got, expected, "Independent pending/free delayed covariance")
    # Source delayed estimator: hidden labels cannot matter; arrival output is pre-update.
    n = 20; x = np.ones((1, n, 1)); v = np.ones((1, n), bool)
    ready = np.array([2]); mean = np.zeros((1, 2)); cov = np.eye(2)[None]
    f = np.array([[1., .8]]); q = np.array([[.01, .2]])
    y = np.linspace(1., 2., n); im = np.zeros(n, bool); qr = im.copy(); qr[6] = True
    def replay(target):
        return env["_fleet_fixed"](x, v, ready, mean.copy(), cov.copy(), f, q, np.array([.4]), target, im, qr, 3, True)[0]
    pred = replay(y); hidden = y.copy(); hidden[~qr] = np.nan
    np.testing.assert_allclose(pred, replay(hidden), equal_nan=True, rtol=0, atol=0)
    changed = y.copy(); changed[6] += 100.; later = replay(changed)
    np.testing.assert_allclose(pred[:, :10], later[:, :10], equal_nan=True, rtol=0, atol=0)
    assert np.max(np.abs(pred[:, 10:]-later[:, 10:])) > .01
    gp = extracts(project / "scripts/revision4_gp_active.py", ("sequential_variance_reduction",))
    free = np.zeros(6, bool); free[2] = True; paid = {0, 1}; offsets = np.arange(1, 6)
    gp_value = gp["sequential_variance_reduction"](np.ones((6, 6)), np.arange(6), offsets,
                                                    np.full(5, .2), free, paid, 1.)
    nb = na = 0; analytic = 0.
    for j in range(6):
        if j: analytic += .2*(1/(1+nb)-1/(1+na))
        if free[j]: nb += 1; na += 1
        elif j in paid: na += 1
    close(gp_value, analytic, "GP sequential latent variance analytic case")
    frozen = extracts(project / "scripts/revision4_components.py", ("frozen_observe",))
    a = rng.normal(size=(5, 5)); cov = a @ a.T; initial = cov.copy()
    mean = rng.normal(size=5); before = mean.copy(); h = rng.normal(size=5); r = .4
    ph = cov @ h; gain = np.r_[np.zeros(4), ph[-1]/(r+h@ph)]
    identity = np.eye(5)-np.outer(gain, h)
    joseph = identity @ initial @ identity.T + r*np.outer(gain, gain)
    frozen["frozen_observe"](mean, cov, h, 7., r)
    close(cov, joseph, "Frozen constrained Joseph covariance")
    np.testing.assert_array_equal(mean[:4], before[:4])
    return dict(status="PASS", delayed_pending_free_covariance=float(got),
                gp_analytic_variance_reduction=float(gp_value),
                hidden_target_and_arrival_order=True, constrained_Joseph_covariance=True,
                execution="Named functions extracted from supplied project code; decorators removed; pure NumPy execution")


def verify(project):
    root = project / "results/revision4"; checked_hashes = {}
    def check_hash(identifier, expected):
        path = inside(project, identifier); relative = path.relative_to(project).as_posix()
        actual = checked_hashes.get(relative) or digest(path)
        assert actual == expected, f"Hash mismatch {relative}: {actual} != {expected}"
        checked_hashes[relative] = actual
    receipts = {}
    d = read_json(root / "delay/independent_validation.json"); assert d["status"] == "PASS"
    check_hash("scripts/revision4_delay_controller.py", d["controller_sha256"])
    receipts["delay"] = dict(path="results/revision4/delay/independent_validation.json", sha256=digest(root/"delay/independent_validation.json"))
    for directory in ("gp_active", "gp_delay"):
        receipt = read_json(root / directory / "verification.json"); assert receipt["status"] == "PASS"
        check_hash(f"scripts/revision4_{directory}.py", receipt["runner_sha256"])
        for item in receipt["checks"]:
            assert item["status"] == "PASS"
            filename = (f'{item["site"]}_B48_gp_active.npz' if directory == "gp_active" else
                        f'{item["site"]}_B48_gp_{item["policy"]}_delay72.npz')
            check_hash(f"results/revision4/{directory}/streams/{filename}", item["output_sha256"])
        index = read_json(root / directory / "source_index.json")
        check_hash(f"results/revision4/{directory}/plan.json", index["plan_sha256"])
        check_hash(f"scripts/revision4_{directory}.py", index["runner_sha256"])
        for item in index["cells"]: check_hash(item["path"], item["sha256"])
        receipts[directory] = dict(path=f"results/revision4/{directory}/verification.json", sha256=digest(root/directory/"verification.json"))
    component = read_json(root / "components/verification.json"); assert component["status"] == "PASS"
    for name, value in component["output_sha256"].items(): check_hash(f"results/revision4/components/{name}", value)
    protocol = read_json(root / "components/protocol.json")
    code_hash = protocol["implementation_sha256"]
    for amendment in protocol.get("implementation_amendments", []):
        assert amendment["old_sha256"] == code_hash
        code_hash = amendment["new_sha256"]
    check_hash("scripts/revision4_components.py", code_hash)
    for name, value in protocol["source_sha256"].items(): check_hash(name, value)
    receipts["components"] = dict(path="results/revision4/components/verification.json", sha256=digest(root/"components/verification.json"))
    summaries = []; support = {}; actual_streams = 0; instrument_checks = 0
    designs = [
        ("delay", "delay_aware72", "{site}_B48_delay_aware72.npz", "extended"),
        ("delay", "delay_aware72_untuned", "{site}_B48_delay_aware72_untuned.npz", "extended"),
        ("gp_active", "gp_active", "streams/{site}_B48_gp_active.npz", ""),
        ("gp_delay", "gp_periodic_d72", "streams/{site}_B48_gp_periodic_delay72.npz", ""),
        ("gp_delay", "gp_fixed_d72", "streams/{site}_B48_gp_risk_fixed_delay72.npz", ""),
    ] + [("components", cell, "{site}_B48_"+cell+".npz", "") for cell in CELLS]
    for directory, method, pattern, prefix in designs:
        runs = {site: load_stream(root/directory/pattern.format(site=site)) for site in SITES}
        actual_streams += len(runs)
        for site, stream in runs.items():
            stream_path = root/directory/pattern.format(site=site)
            checked_hashes[stream_path.relative_to(project).as_posix()] = digest(stream_path)
            group = "delayed" if directory in ("delay", "gp_delay") else "immediate"
            key = (group, site)
            if key in support:
                np.testing.assert_array_equal(stream["score_masks"], support[key]["score_masks"])
                np.testing.assert_array_equal(stream["times"], support[key]["times"])
                np.testing.assert_array_equal(stream["y"], support[key]["y"])
            else: support[key] = stream
            provenance = stream["provenance"]
            for field, code in (("runner_sha256", f"scripts/revision4_{directory}.py"),
                                ("controller_sha256", "scripts/revision4_delay_controller.py")):
                if field in provenance: check_hash(code, provenance[field])
            if "plan_sha256" in provenance: check_hash(f"results/revision4/{directory}/plan.json", provenance["plan_sha256"])
        aggregate_csv = (root/directory/(prefix+"_aggregate_metrics.csv" if prefix else "aggregate_metrics.csv")) if directory != "components" else root/directory/"component_aggregate_metrics.csv"
        instrument_csv = (root/directory/(prefix+"_instrument_metrics.csv" if prefix else "instrument_metrics.csv")) if directory != "components" else root/directory/"component_instrument_metrics.csv"
        saved_aggregate = pd.read_csv(aggregate_csv); saved_instruments = pd.read_csv(instrument_csv)
        scopes = ["all_scored__all"] if directory != "components" else [b+"__"+g for b in ("all_scored", "gap_ge168h") for g in ("all", "long_Zep", "short_multi_product")]
        for scope in scopes:
            values, instruments = aggregate(runs, scope)
            selected = saved_aggregate[saved_aggregate.method.eq(method)]
            ins_selected = saved_instruments[saved_instruments.method.eq(method)]
            if directory == "components":
                selected = selected[selected.scope.eq(scope)]; ins_selected = ins_selected[ins_selected.scope.eq(scope)]
            assert len(selected) == 1
            row = selected.iloc[0]
            close(values["macro_instrument_rmse"], row.macro_instrument_rmse, f"{directory}/{method}/{scope}")
            equal_field = "site_equal_rmse" if directory == "components" else "site_balanced_rmse"
            close(values["site_equal_rmse"], row[equal_field], "site equal")
            assert values["instrument_hours"] == int(row.instrument_hours) and values["n_instruments"] == int(row.n_instruments)
            for item in instruments.itertuples():
                chosen = ins_selected[ins_selected.instrument.eq(item.instrument)]
                if "site" in chosen: chosen = chosen[chosen.site.eq(item.site)]
                assert len(chosen) == 1
                close(item.rmse, chosen.rmse.iloc[0], "physical instrument RMSE")
                assert item.instrument_hours == int(chosen.instrument_hours.iloc[0])
                if "channel_observations" in chosen:
                    assert item.channel_observations == int(chosen.channel_observations.iloc[0])
                instrument_checks += 1
            summaries.append(dict(directory=directory, method=method, scope=scope, **values))
    assert actual_streams == 27
    assert all(r["instrument_hours"] == (59674 if r["directory"] in ("delay", "gp_delay") else 62652)
               for r in summaries if r["scope"] == "all_scored__all")
    return dict(status="PASS", project=str(project), utc=datetime.now(timezone.utc).isoformat(),
                actual_npz_streams_reaggregated=actual_streams, instrument_metric_rows_checked=instrument_checks,
                aggregate_rows_checked=len(summaries), summaries=summaries,
                receipt_files_checked=receipts, file_hashes_checked=checked_hashes,
                synthetic_smoke=smoke(project),
                scope="Fresh reaggregation of all 27 new streams plus source-code small numerical checks; no refitting or bootstrap generation",
                original_absolute_paths_read=False, writes="Only the explicit --receipt destination")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    assert args.project.is_absolute() and args.receipt.is_absolute(), "Both arguments must be absolute"
    project = args.project.resolve(); receipt = args.receipt.resolve()
    assert receipt.drive.upper() == "D:", "Receipt/compute outputs must remain on D"
    assert not receipt.exists(), "Use a new receipt name; existing receipts are immutable"
    assert "manifest" not in receipt.name.lower(), "Do not overwrite a package manifest"
    started = perf_counter()
    try:
        result = verify(project)
    except Exception as error:
        result = dict(status="FAIL", project=str(project), error=repr(error))
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
        raise
    result["helper_sha256"] = digest(Path(__file__).resolve())
    result["wall_seconds"] = perf_counter()-started
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "project", "actual_npz_streams_reaggregated", "instrument_metric_rows_checked", "aggregate_rows_checked", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
