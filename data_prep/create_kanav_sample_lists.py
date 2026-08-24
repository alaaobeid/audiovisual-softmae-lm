#!/usr/bin/env python3
"""
create_kanav_sample_lists.py

Generates balanced and boundary sample lists from KAN-AV's sample_duration_list.json
(provided with the raw dataset under Other/).

Output:
  - balanced_1056_samples.json: exactly 12 samples/age (88 ages × 12 = 1056)
  - balanced_1000_samples.json: ~11 samples/age, capped at 1000 total
  - boundary_500_samples.json:  ~62 samples/age for ages 14-21 (500 total)

Usage:
    python create_kanav_sample_lists.py --kan_root /path/to/KAN [--output_dir /path/to/out]
"""
import json, os, random, argparse
from collections import defaultdict

random.seed(42)


def main(args):
    with open(os.path.join(args.kan_root, "Other", "sample_duration_list.json")) as f:
        all_samples = json.load(f)

    # Age is the 3rd path component: Batch{N}/Person_Name/{AGE}/{clip_name}
    by_age = defaultdict(list)
    for path, duration in all_samples:
        age = int(path.split("/")[2])
        by_age[age].append((path, duration))

    for age in by_age:
        random.shuffle(by_age[age])

    # balanced_1056: exactly 12 per age
    balanced_1056 = []
    for age in sorted(by_age):
        if len(by_age[age]) >= 12:
            balanced_1056.extend(by_age[age][:12])

    # balanced_1000: ~11 per age, capped at 1000
    balanced_1000 = []
    ages_sorted = sorted(by_age.keys())
    per_age = 1000 // len(ages_sorted)
    remainder = 1000 - per_age * len(ages_sorted)
    for i, age in enumerate(ages_sorted):
        n = per_age + (1 if i < remainder else 0)
        if len(by_age[age]) >= n:
            balanced_1000.extend(by_age[age][:n])

    # boundary_500: ages 14-21, ~62-63 per age
    boundary_ages = range(14, 22)
    boundary_500 = []
    per_b = 500 // len(boundary_ages)
    remainder_b = 500 - per_b * len(boundary_ages)
    for i, age in enumerate(boundary_ages):
        n = per_b + (1 if i < remainder_b else 0)
        samples = by_age.get(age, [])
        if len(samples) >= n:
            boundary_500.extend([(p, age) for p, _ in samples[:n]])

    out_dir = args.output_dir or args.kan_root
    os.makedirs(out_dir, exist_ok=True)
    outputs = [
        ("balanced_1056_samples.json", [[p, d] for p, d in balanced_1056]),
        ("balanced_1000_samples.json", [[p, d] for p, d in balanced_1000[:1000]]),
        ("boundary_500_samples.json",  [[p, a] for p, a in boundary_500[:500]]),
    ]
    for name, data in outputs:
        path = os.path.join(out_dir, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {path}: {len(data)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kan_root", required=True, help="Path to KAN-AV root directory")
    parser.add_argument("--output_dir", default=None, help="Output dir (default: kan_root)")
    main(parser.parse_args())
