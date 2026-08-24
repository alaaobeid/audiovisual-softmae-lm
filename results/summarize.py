#!/usr/bin/env python3
"""
summarize.py — aggregate per-seed evaluation CSVs into a summary table.

Usage:
    python results/summarize.py results/alpha009_s*.csv
    python results/summarize.py results/          # all CSVs in dir
"""
import sys, os, glob
import pandas as pd
import numpy as np


def load_csv(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["pred_age"])
    df["error"] = (df["pred_age"] - df["true_age"]).abs()
    mae = df["error"].mean()
    coverage = len(df) / (len(pd.read_csv(path)))
    return mae, coverage, len(df), path


def main():
    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            paths.extend(sorted(glob.glob(os.path.join(arg, "*.csv"))))
        else:
            paths.extend(sorted(glob.glob(arg)))

    if not paths:
        print("Usage: python summarize.py <csv_files_or_dir>")
        sys.exit(1)

    rows = []
    for p in paths:
        try:
            mae, cov, n, path = load_csv(p)
            rows.append({"file": os.path.basename(p), "mae": mae, "coverage": cov, "n": n})
        except Exception as e:
            print(f"WARN: {p}: {e}")

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format="%.3f"))
    print(f"\nMean MAE: {df['mae'].mean():.3f} ± {df['mae'].std():.3f}  (n={len(df)} runs)")


if __name__ == "__main__":
    main()
