"""Cache the independent QUANT Ratified NO2 reference series on D:.

Reads the immutable reference CSV once. No join to any sensor table, no
interpolation, and no restriction to sensor-record availability is performed.
Output columns are timestamp, site, y; source version and units are retained
as explicit Parquet metadata. Source timestamps are kept unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        r"F:\AcademicData\SensorsWindowCalibration\raw\QUANT_Reference_hourly.csv"))
    parser.add_argument("--output", type=Path, default=Path(
        r"D:\MLWork\SensorsWindowCalibration\datasets\reference_no2_ratified.parquet"))
    args = parser.parse_args()
    if args.output.drive.upper() != "D:":
        raise ValueError("Derived reference cache must be stored on D:")
    # Single source read. All subsequent inspections operate on this frame.
    raw = pd.read_csv(args.source)
    expected = ["time", "location", "version", "measurand", "measurement"]
    if list(raw.columns) != expected:
        raise ValueError(f"Unexpected source columns: {list(raw.columns)}")
    selected = raw[raw.measurand.eq("NO2") & raw.version.eq("Ratified")]
    result = selected[["time", "location", "measurement"]].rename(
        columns={"time": "timestamp", "location": "site", "measurement": "y"}).copy()
    result["timestamp"] = pd.to_datetime(result.timestamp, errors="raise")
    result["site"] = result.site.astype("string")
    result["y"] = pd.to_numeric(result.y, errors="raise").astype("float64")
    assert result.timestamp.notna().all() and result.site.notna().all()
    assert not result.duplicated(["timestamp", "site"]).any(), "Ambiguous reference values"
    assert result.timestamp.eq(result.timestamp.dt.floor("h")).all(), "Not an hourly time grid"
    result = result.sort_values(["site", "timestamp"]).reset_index(drop=True)
    table = pa.Table.from_pandas(result, preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata.update({
        b"source_record": b"https://zenodo.org/records/10775692",
        b"source_release": b"QUANT v1",
        b"source_filename": args.source.name.encode("utf-8"),
        b"source_version": b"Ratified",
        b"measurand": b"NO2",
        b"y_units": b"ppb",
        b"time_policy": b"Exact source timestamps, no timezone or DST shift",
        b"formation": b"NO2 and Ratified filters only; independent of sensor availability",
    })
    table = table.replace_schema_metadata(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(args.output)
    # Verify the saved derivative; do not reopen the immutable source.
    restored = pq.read_table(args.output)
    assert restored.num_rows == len(result)
    assert restored.schema.names == ["timestamp", "site", "y"]
    assert restored.schema.metadata[b"source_version"] == b"Ratified"
    assert restored.schema.metadata[b"y_units"] == b"ppb"
    by_site = result.groupby("site").agg(
        rows=("y", "size"), finite_y=("y", lambda v: int(np.isfinite(v).sum())),
        start=("timestamp", "min"), end=("timestamp", "max"))
    print(json.dumps(dict(
        source_columns=list(raw.columns), source_rows=len(raw),
        saved_columns={c: str(d) for c, d in result.dtypes.items()},
        saved_rows=len(result), finite_y=int(np.isfinite(result.y).sum()),
        source_version="Ratified", y_units="ppb", output=str(args.output),
        by_site=by_site.reset_index().to_dict(orient="records"),
    ), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
