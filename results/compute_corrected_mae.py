#!/usr/bin/env python3
"""
compute_corrected_mae.py

Reads results/predictions/serial.json as the implementation-correctness
baseline and every other results/predictions/*.json as a parallel
observation. For each parallel file, computes

    Corrected MAE  = mean |Pred_parallel(t) - Pred_serial(t)|
    Corrected RMSE = sqrt(mean (Pred_parallel(t) - Pred_serial(t))^2)

over the canonical test_set ordering (|T| = 29,866 for the default
N=M=1000 run). Writes results/corrected_mae.csv with one row per
parallel file. Expected: 0.0000 for every parallel implementation, i.e.
every parallel version reproduces the serial prediction array exactly.

Each JSON file is a flat array of floats, in test_set traversal order,
written by the recommender binaries when the env var PRED_DUMP_PATH is
set. See dump_test_predictions_json() in the *_recommender sources.

Usage: python3 compute_corrected_mae.py [predictions_dir]
"""

import csv
import json
import math
import sys
from pathlib import Path


def parse_filename(stem: str):
    """Map a JSON filename stem (without .json) to (Version, Workers)."""
    if stem == "serial":
        return ("Serial", "1")
    if stem == "cuda":
        return ("CUDA", "GPU")
    parts = stem.split("_")
    if parts[0] == "openmp":
        return ("OpenMP", parts[1])
    if parts[0] == "pthreads":
        return ("Pthreads", parts[1])
    if parts[0] == "mpi":
        return ("MPI", parts[1])
    if len(parts) >= 3 and parts[0] == "hybrid" and parts[1] == "omp":
        return ("Hybrid MPI+OpenMP", parts[2])
    if len(parts) >= 3 and parts[0] == "hybrid" and parts[1] == "pt":
        return ("Hybrid MPI+Pthreads", parts[2])
    return (stem, "?")


def compute_corrected(parallel, serial):
    if len(parallel) != len(serial):
        raise ValueError(
            f"length mismatch: parallel={len(parallel)} vs serial={len(serial)}"
        )
    n = len(serial)
    abs_sum = 0.0
    sq_sum = 0.0
    for p, s in zip(parallel, serial):
        d = p - s
        abs_sum += abs(d)
        sq_sum += d * d
    return abs_sum / n, math.sqrt(sq_sum / n)


def main():
    pred_dir = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else Path(__file__).resolve().parent / "predictions"

    serial_path = pred_dir / "serial.json"
    if not serial_path.is_file():
        print(f"ERROR: baseline file not found: {serial_path}",
              file=sys.stderr)
        return 1

    with open(serial_path) as f:
        serial = json.load(f)
    print(f"Loaded serial baseline: {len(serial)} predictions from "
          f"{serial_path}\n")

    header = (f"{'Version':<22} {'Workers':<8} "
              f"{'Corrected MAE':>16} {'Corrected RMSE':>16} {'Match':>6}")
    print(header)
    print(f"{'-'*22} {'-'*8} {'-'*16} {'-'*16} {'-'*6}")

    rows = []
    for path in sorted(pred_dir.glob("*.json")):
        if path.name == "serial.json":
            continue
        with open(path) as f:
            parallel = json.load(f)
        version, workers = parse_filename(path.stem)
        mae, rmse = compute_corrected(parallel, serial)
        match = "YES" if mae <= 1e-9 and rmse <= 1e-9 else "NO"
        rows.append([version, workers,
                     f"{mae:.10f}", f"{rmse:.10f}", match])
        print(f"{version:<22} {workers:<8} "
              f"{mae:>16.10f} {rmse:>16.10f} {match:>6}")

    out_csv = pred_dir.parent / "corrected_mae.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Version", "Workers",
                    "Corrected_MAE", "Corrected_RMSE", "Match"])
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_csv}")

    bad = [r for r in rows if r[-1] != "YES"]
    if bad:
        print(f"WARNING: {len(bad)} parallel run(s) did NOT match the "
              f"serial baseline.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
