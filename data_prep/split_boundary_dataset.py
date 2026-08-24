#!/usr/bin/env python3
"""
split_boundary_dataset.py

Splits boundary_500.csv (produced by extract_kanav_dataset.py) into:
  - boundary_val.csv   — 100 samples (used for early stopping during training)
  - boundary_test.csv  — 400 samples (held-out test set)

Stratified by age so each age 14-21 is represented proportionally in both splits.
Seed=42 for reproducibility.

Usage:
    python data_prep/split_boundary_dataset.py \
        --boundary_csv boundary_500_dataset/boundary_500.csv \
        --output_dir   boundary_500_dataset
"""
import csv, os, random, argparse
from collections import defaultdict

random.seed(42)


def main(args):
    rows_by_age = defaultdict(list)
    with open(args.boundary_csv) as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows_by_age[int(row["text"])].append(row)

    for age in rows_by_age:
        random.shuffle(rows_by_age[age])

    # Stratified split: 20% val (100/500), 80% test (400/500)
    val_rows, test_rows = [], []
    for age in sorted(rows_by_age):
        rows = rows_by_age[age]
        n_val = max(1, round(len(rows) * 0.2))
        val_rows.extend(rows[:n_val])
        test_rows.extend(rows[n_val:])

    random.shuffle(val_rows)
    random.shuffle(test_rows)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, rows in [("boundary_val.csv", val_rows), ("boundary_test.csv", test_rows)]:
        path = os.path.join(args.output_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {path}: {len(rows)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary_csv", required=True, help="Path to boundary_500.csv")
    parser.add_argument("--output_dir",   required=True, help="Directory to write val/test CSVs")
    main(parser.parse_args())
