"""Download immutable QUANT v1 source files and record verified provenance.

Raw downloads are staged on D:, MD5 verified, then exclusively created on F:.
Existing raw files are never replaced, deleted, or modified.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

import requests

RECORD = "10775692"
EXPECTED = {
    "QUANT_SensorSystems_hourly.csv": "7f8665948c58014f4dde1122d5f438e6",
    "QUANT_Reference_hourly.csv": "51df9e9632f21e5a8fcbed11662cab45",
    "QUANT_DuplicateRef_hourly.csv": "95a7adc5e12f6b67f92baadb9a18cb2a",
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def readonly(path: Path) -> None:
    path.chmod(stat.S_IREAD)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw", type=Path, default=Path(r"F:\AcademicData\SensorsWindowCalibration\raw"))
    p.add_argument("--work", type=Path, default=Path(r"D:\MLWork\SensorsWindowCalibration"))
    a = p.parse_args()
    if a.raw.drive.upper() not in {"F:", "E:"} or a.work.drive.upper() != "D:":
        raise ValueError("Raw files must be on F/E and work files on D")
    a.raw.mkdir(parents=True, exist_ok=True)
    temp = a.work / "temp"
    temp.mkdir(parents=True, exist_ok=True)
    (a.work / "datasets").mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    response = requests.get(f"https://zenodo.org/api/records/{RECORD}", timeout=90)
    response.raise_for_status()
    record = response.json()
    files = [x for x in record["files"] if x["key"] in EXPECTED]
    if len(files) != len(EXPECTED):
        raise ValueError("Zenodo record file list changed")
    required = sum(x["size"] for x in files)
    spaces = {d: shutil.disk_usage(d).free for d in ["D:/", "E:/", "F:/"]}
    print(json.dumps({"raw_destination": str(a.raw), "free_bytes": spaces, "download_bytes": required}), flush=True)
    if spaces[a.raw.drive.upper() + "/"] < required * 1.2 or spaces["D:/"] < required * 1.2:
        raise RuntimeError("Insufficient staging or raw space")
    metadata_path = a.raw / f"QUANT_v1_zenodo_{RECORD}_metadata_{now}.json"
    with metadata_path.open("x", encoding="utf-8") as f:
        json.dump({"retrieved_at_utc": now, "api_url": response.url, "record": record}, f, indent=2, ensure_ascii=False)
    readonly(metadata_path)

    def download(item: dict) -> dict:
        key, expected = item["key"], EXPECTED[item["key"]]
        if item["checksum"] != "md5:" + expected:
            raise ValueError(f"Unexpected published checksum for {key}")
        dst = a.raw / key
        if dst.exists():
            actual = md5_file(dst)
            if actual != expected or dst.stat().st_size != item["size"]:
                raise ValueError(f"Existing immutable file mismatch: {dst}")
            print(f"Verified existing {key}", flush=True)
            return {"path": str(dst), "bytes": dst.stat().st_size, "md5": actual, "source_url": item["links"]["self"], "reused": True}
        stage = temp / f"{now}_{key}.part"
        for attempt in range(1, 4):
            try:
                h = hashlib.md5()
                size = 0
                with requests.get(item["links"]["self"], stream=True, timeout=(30, 120)) as r:
                    r.raise_for_status()
                    with stage.open("wb") as f:
                        for block in r.iter_content(4 * 1024 * 1024):
                            if block:
                                f.write(block)
                                h.update(block)
                                size += len(block)
                if size != item["size"] or h.hexdigest() != expected:
                    raise ValueError(f"Integrity mismatch for {key}: {size}, {h.hexdigest()}")
                with stage.open("rb") as src, dst.open("xb") as out:
                    shutil.copyfileobj(src, out, 8 * 1024 * 1024)
                if md5_file(dst) != expected:
                    raise ValueError(f"Copy verification failed: {dst}")
                readonly(dst)
                print(f"Downloaded verified {key}: {size} bytes", flush=True)
                return {"path": str(dst), "bytes": size, "md5": expected, "source_url": item["links"]["self"], "reused": False}
            except requests.RequestException as exc:
                print(f"Retry {attempt} {key}: {type(exc).__name__}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(2 * attempt)
        raise RuntimeError("Download exhausted")

    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(download, files))
    manifest = {"record_id": RECORD, "version": record["metadata"].get("version", "v1"), "record_doi": record["metadata"].get("doi"), "license": record["metadata"].get("license"), "downloaded_at_utc": now, "metadata_path": str(metadata_path), "files": outputs}
    raw_manifest = a.raw / f"QUANT_v1_download_manifest_{now}.json"
    with raw_manifest.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    readonly(raw_manifest)
    with (a.work / "datasets" / "source_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
