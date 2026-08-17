"""Atomic CSV append, run directories, and manifest writing.

The atomic-append pattern (temp file in the same directory + os.replace) is what makes run_grid.py
safe to SIGKILL and restart: a crash mid-write leaves the previous results.csv untouched, never a
half-written file.
"""
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Set

RESULTS_DIR = Path("results")
RUNS_DIR = RESULTS_DIR / "runs"


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_manifest(run_id: str, manifest: dict) -> Path:
    path = run_dir(run_id) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def atomic_append_csv(csv_path, row: Dict[str, object], fieldnames: Iterable[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fieldnames)

    existing_rows = []
    if csv_path.exists():
        with csv_path.open("r", newline="") as fh:
            existing_rows = list(csv.DictReader(fh))

    fd, tmp_path = tempfile.mkstemp(dir=str(csv_path.parent), prefix=".tmp-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in existing_rows:
                writer.writerow(r)
            writer.writerow(row)
        os.replace(tmp_path, csv_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def existing_run_ids(csv_path) -> Set[str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="") as fh:
        return {row["run_id"] for row in csv.DictReader(fh)}
