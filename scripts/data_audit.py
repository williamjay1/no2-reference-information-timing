"""Audit QUANT v1 and prepare non-retrospectively-recalibrated NO2 products.

No model fitting, gap filling, clipping, or outcome-dependent data selection.
Only out-of-box sensor values enter analysis tables; calibrated versions remain
in audit counts. Time values remain as provided, with no assumed timezone shift.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def read_sensor(path: Path) -> tuple[pd.DataFrame, dict]:
    keep = []
    count = Counter()
    total = 0
    for chunk in pd.read_csv(path, chunksize=400_000, dtype={"sensornumber": "int16", "flag": "string", "flagreason": "string"}):
        total += len(chunk)
        count.update({f"{v}|{m}": int(n) for (v, m), n in chunk.groupby(["version", "measurand"]).size().items()})
        keep.append(chunk[chunk.measurand.isin(["NO2", "Temperature", "RelHumidity"])].copy())
    data = pd.concat(keep, ignore_index=True)
    data["time"] = pd.to_datetime(data.time, errors="raise")
    data["flag"] = data.flag.fillna("")
    data["flagreason"] = data.flagreason.fillna("")
    return data, {"total_rows": total, "rows_by_version_measurand": dict(sorted(count.items()))}


def concurrent_channel_checks(sensor: pd.DataFrame) -> list[dict]:
    """Compare recorded IMB channels; counts alone cannot establish duplication."""
    result = []
    for instrument in ["IMB1", "IMB2"]:
        for version in ["out-of-box", "cal1"]:
            selected = sensor[sensor.instrument.eq(instrument) & sensor.version.eq(version) & sensor.measurand.eq("NO2")]
            wide = selected.pivot(index=["time", "location"], columns="sensornumber", values="measurement")
            for first, second in combinations(wide.columns, 2):
                pair = wide[[first, second]].dropna()
                delta = pair[first] - pair[second]
                result.append(dict(instrument=instrument, version=version,
                                   first=int(first), second=int(second),
                                   paired_hours=len(pair), exact_equal=int(delta.eq(0).sum()),
                                   max_absolute_difference=float(delta.abs().max()),
                                   correlation=float(pair[first].corr(pair[second]))))
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path(r"F:\AcademicData\SensorsWindowCalibration\raw"))
    p.add_argument("--work", type=Path, default=Path(r"D:\MLWork\SensorsWindowCalibration"))
    a = p.parse_args()
    if a.work.drive.upper() != "D:":
        raise ValueError("Derived data must be written on D")
    out = a.work / "datasets"
    res = a.work / "results"
    out.mkdir(parents=True, exist_ok=True)
    res.mkdir(parents=True, exist_ok=True)
    sensor, audit = read_sensor(a.raw / "QUANT_SensorSystems_hourly.csv")
    reference = pd.read_csv(a.raw / "QUANT_Reference_hourly.csv", parse_dates=["time"])
    duplicate = pd.read_csv(a.raw / "QUANT_DuplicateRef_hourly.csv", parse_dates=["time"])
    sensor_key = ["time", "location", "instrument", "sensornumber", "version", "measurand"]
    ref_key = ["time", "location", "version", "measurand"]
    assert not sensor.duplicated(sensor_key).any(), "Ambiguous sensor rows"
    assert not reference.duplicated(ref_key).any(), "Ambiguous reference rows"
    assert sensor.time.dt.minute.eq(0).all() and reference.time.dt.minute.eq(0).all()
    z = sensor[sensor.measurand.eq("NO2")].copy()
    version_coverage = z.groupby(["location", "instrument", "sensornumber", "version"]).agg(n_rows=("measurement", "size"), n_nonmissing=("measurement", "count"), start=("time", "min"), end=("time", "max")).reset_index()
    version_coverage.to_csv(res / "no2_all_version_coverage.csv", index=False)
    reference.groupby(["location", "version", "measurand"]).agg(n_rows=("measurement", "size"), n_nonmissing=("measurement", "count"), start=("time", "min"), end=("time", "max")).reset_index().to_csv(res / "reference_coverage.csv", index=False)
    z.groupby(["version", "flag", "flagreason"], dropna=False).size().reset_index(name="n").to_csv(res / "no2_flags_by_version.csv", index=False)
    sensor.to_parquet(out / "sensor_no2_met_all_versions.parquet", index=False)

    # One physical sensor can move between sites. Preserve a cross-site identity
    # and a deployment identity rather than treating moves as independent units.
    oob = z[z.version.eq("out-of-box")].copy()
    oob = oob.rename(columns={"time": "timestamp", "location": "site", "measurement": "x", "version": "sensor_version", "flag": "sensor_flag", "flagreason": "sensor_flagreason"}).drop(columns="measurand")
    oob["device"] = oob.instrument + "__s" + oob.sensornumber.astype(str)
    oob["deployment"] = oob.site + "__" + oob.device
    oob["family"] = oob.instrument.str.extract(r"^([A-Za-z]+)")[0]
    join_key = ["timestamp", "site", "instrument", "sensornumber"]
    for measurand, field in [("Temperature", "T"), ("RelHumidity", "RH")]:
        m = sensor[sensor.measurand.eq(measurand) & sensor.version.eq("out-of-box")].rename(columns={"time": "timestamp", "location": "site", "measurement": field, "flag": field + "_flag", "flagreason": field + "_flagreason"})
        oob = oob.merge(m[join_key + [field, field + "_flag", field + "_flagreason"]], on=join_key, how="left", validate="one_to_one")
    r = reference[reference.measurand.eq("NO2")].rename(columns={"time": "timestamp", "location": "site", "measurement": "y", "version": "reference_version"})
    assert not r.duplicated(["timestamp", "site"]).any(), "Competing reference versions at same hour"
    oob = oob.merge(r[["timestamp", "site", "y", "reference_version"]], on=["timestamp", "site"], how="left", validate="many_to_one")
    for measurand, field in [("Temperature", "T_reference"), ("RelHumidity", "RH_reference")]:
        m = reference[reference.measurand.eq(measurand) & reference.version.eq("Ratified")].rename(columns={"time": "timestamp", "location": "site", "measurement": field})
        oob = oob.merge(m[["timestamp", "site", field]], on=["timestamp", "site"], how="left", validate="many_to_one")
    oob["valid_x"] = np.isfinite(oob.x)
    oob["valid_y"] = np.isfinite(oob.y)
    oob["sensor_error"] = oob.sensor_flag.eq("Error")
    oob["sensor_warning"] = oob.sensor_flag.eq("Warning")
    oob["manufacturer_rebase_flag"] = oob.sensor_flagreason.str.contains("Rebas", case=False, na=False)
    oob["retrospective_collection_flag"] = oob.sensor_flagreason.str.contains("Retrospectively collected", na=False)
    oob["eligible_ratified"] = oob.valid_x & oob.valid_y & oob.reference_version.eq("Ratified") & ~oob.sensor_error
    oob["complete_device_met"] = np.isfinite(oob["T"]) & np.isfinite(oob["RH"]) & ~oob.T_flag.eq("Error") & ~oob.RH_flag.eq("Error")
    oob = oob.sort_values(["site", "device", "timestamp"]).reset_index(drop=True)
    assert not oob.duplicated(["site", "device", "timestamp"]).any()
    oob.to_parquet(out / "no2_hourly_oob.parquet", index=False)
    main = oob[oob.eligible_ratified].copy()
    main.to_parquet(out / "no2_hourly_oob_ratified.parquet", index=False)
    main[main.complete_device_met].to_parquet(out / "no2_hourly_oob_ratified_complete_met.parquet", index=False)
    coverage = []
    for keys, g in oob.groupby(["site", "instrument", "sensornumber", "device", "deployment", "family"]):
        span_h = int((g.timestamp.max() - g.timestamp.min()).total_seconds() / 3600) + 1
        valid = g[g.eligible_ratified]
        coverage.append(dict(zip(["site", "instrument", "sensornumber", "device", "deployment", "family"], keys)) | {
            "start": str(g.timestamp.min()), "end": str(g.timestamp.max()), "span_hours": span_h,
            "n_oob_hours": len(g), "fraction_of_span_recorded": len(g)/span_h,
            "n_ratified_eligible": int(g.eligible_ratified.sum()), "n_unratified_y": int(g.reference_version.eq("Unratified").sum()), "n_missing_reference": int(g.y.isna().sum()),
            "n_ratified_complete_device_met": int((g.eligible_ratified & g.complete_device_met).sum()),
            "n_missing_T": int(g["T"].isna().sum()), "n_missing_RH": int(g["RH"].isna().sum()),
            "n_sensor_error": int(g.sensor_error.sum()), "n_sensor_warning": int(g.sensor_warning.sum()),
            "n_rebased_or_rebasing": int(g.manufacturer_rebase_flag.sum()), "n_retrospective_collection": int(g.retrospective_collection_flag.sum()),
            "max_record_gap_hours": float(g.timestamp.diff().dt.total_seconds().div(3600).max()) if len(g)>1 else None,
            "ratified_start": str(valid.timestamp.min()), "ratified_end": str(valid.timestamp.max()),
            "ratified_distinct_days": int(valid.timestamp.dt.normalize().nunique()),
        })
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(res / "no2_oob_coverage.csv", index=False)
    oob.assign(month=oob.timestamp.dt.to_period("M").astype(str)).groupby(["site", "device", "month"]).agg(n_oob=("x", "size"), n_ratified=("eligible_ratified", "sum"), n_complete_met=("complete_device_met", "sum")).reset_index().to_csv(res / "no2_oob_monthly_coverage.csv", index=False)
    dup = duplicate[duplicate.measurand.eq("NO2")].rename(columns={"time": "timestamp", "location": "site", "measurement": "duplicate_y"}).drop(columns="measurand")
    dup = dup.merge(r[["timestamp", "site", "y", "reference_version"]], on=["timestamp", "site"], how="left", validate="many_to_one")
    dup.to_parquet(out / "duplicate_reference_no2.parquet", index=False)
    audit.update({
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_record": "https://zenodo.org/records/10775692", "source_license_metadata": "CC BY 4.0",
        "source_observations": "Hourly manufacturer concentration products, not raw electrode voltages",
        "time_policy": "Join exact provided hourly timestamps, without DST/local timezone conversion. Source export metadata uses Z and reference ingestion uses lubridate as_datetime (UTC default); plain Zenodo CSV strings themselves carry no timezone offset.",
        "units_policy": "NO2 sensor and reference ppb; RH percent. Zenodo hourly CSV temperature uses Celsius-scale values, whereas the first-party CEDA export script explicitly converts Celsius to Kelvin. Keep source CSV values unchanged. Physical-range QC is required for anomalous values.",
        "primary_version": "out-of-box", "excluded_versions": ["cal1", "cal2"],
        "primary_reference": "Ratified only; Unratified values retained in audit table but excluded from main table",
        "flags_policy": "Error rows excluded from main; Warning retained with separate flags, enabling prespecified sensitivity analysis",
        "missingness_policy": "No imputation. Absent hourly rows remain absent; span completeness reported. Local device T/RH only for T/RH; reference meteorology separately named.",
        "identity_policy": "instrument identifies a system; sensornumber distinguishes either simultaneous sensing channels (IMB has three concurrent NO2 measurements) or replacement segments (documented for Aeroqual and Zephyr). device=instrument+sensornumber is a channel/segment key, not an independent-instrument count; it remains the same across sites. deployment additionally includes site. Group uncertainty by instrument/site as appropriate; do not treat channels, replacement segments, moves, or hourly rows as independent instruments.",
        "n_no2_all_versions": len(z), "n_no2_oob": len(oob), "n_no2_ratified_eligible": len(main), "n_no2_ratified_complete_device_met": int(main.complete_device_met.sum()),
        "n_oob_instruments": int(oob.instrument.nunique()), "n_oob_device_keys": int(oob.device.nunique()), "n_oob_site_device_deployments": int(oob.deployment.nunique()),
        "no2_versions": z.version.value_counts().to_dict(), "no2_flags": records(z.groupby(["version", "flag", "flagreason"], dropna=False).size().reset_index(name="n")),
        "concurrent_channel_checks": concurrent_channel_checks(sensor),
        "coverage": coverage, "duplicate_reference_rows": len(dup), "duplicate_reference_qc_counts": duplicate.qc_flag.value_counts(dropna=False).to_dict(),
        "schema": {c: str(d) for c,d in oob.dtypes.items()},
        "no_model_results_generated": True,
    })
    with (res / "data_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)
    with (out / "data_dictionary.json").open("w", encoding="utf-8") as f:
        json.dump({k:audit[k] for k in ["source_record", "source_observations", "time_policy", "units_policy", "primary_version", "excluded_versions", "primary_reference", "flags_policy", "missingness_policy", "identity_policy", "schema"]}, f, indent=2)
    print(json.dumps({"oob_rows": len(oob), "ratified_rows": len(main), "ratified_complete_device_met": int(main.complete_device_met.sum()), "output": str(out), "long_oob_coverage": coverage_df[coverage_df.n_ratified_eligible.ge(4000)].to_dict(orient="records")}, default=str), flush=True)


if __name__ == "__main__":
    main()
