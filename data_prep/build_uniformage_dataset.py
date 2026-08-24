#!/usr/bin/env python3
"""
build_uniformage_dataset.py

Builds uniformage_av/ — the training dataset used in the SoftMAE paper.

  Train: 3000 samples — as uniform as possible across ages 0-80
  Val:   500 samples  — as uniform as possible across ages 0-80
  Test:  500 samples  — unchanged from IDIAPAge test split

Sources:
  - IDIAPAge processed dataset (clips/ + frames/ + train/validation/test CSVs)
  - KAN-AV extracted dataset   (audio/ + frames/ + manifest CSV)

KAN clips used in the boundary_500 and balanced_1000 eval sets are excluded
from training to prevent leakage.

Usage:
    python build_uniformage_dataset.py \
        --idiapage_dir  /path/to/idiapage_dataset \
        --kanav_dir     /path/to/kanav_dataset \
        --boundary_csv  /path/to/boundary_500_dataset/boundary_500.csv \
        --balanced_csv  /path/to/balanced_1000_dataset/balanced_1000_uncropped.csv \
        --output_dir    uniformage_av
"""
import json, csv, os, random, argparse
import numpy as np
from collections import Counter, defaultdict

random.seed(42)

AGE_RANGE = list(range(0, 81))


def load_rows_by_age(csv_path, text_col="text"):
    rows_by_age = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows_by_age[int(row[text_col])].append(row)
    for age in rows_by_age:
        random.shuffle(rows_by_age[age])
    return rows_by_age


def build_uniform_split(idiap_by_age, kan_pool, total_target):
    """
    Allocate exactly total_target samples as uniformly as possible across ages 0-80.
    Fills from IDIAPAge first, then KAN. Consumes from kan_pool in-place.
    """
    max_available = {
        age: len(idiap_by_age.get(age, [])) + len(kan_pool.get(age, []))
        for age in AGE_RANGE
    }

    constrained, unconstrained = {}, set(AGE_RANGE)
    for _ in range(100):
        remaining = total_target - sum(constrained.values())
        per_age = remaining // len(unconstrained) if unconstrained else 0
        newly = [a for a in unconstrained if max_available[a] < per_age]
        if not newly:
            break
        for a in newly:
            constrained[a] = max_available[a]
            unconstrained.remove(a)

    remaining = total_target - sum(constrained.values())
    base = remaining // len(unconstrained)
    extra = remaining - base * len(unconstrained)

    allocation = {a: constrained.get(a, base) for a in AGE_RANGE}
    for a in sorted(unconstrained):
        if extra <= 0:
            break
        allocation[a] += 1
        extra -= 1

    assert sum(allocation.values()) == total_target

    idiap_out, kan_out = [], []
    for age in AGE_RANGE:
        target = allocation[age]
        idiap_rows = idiap_by_age.get(age, [])
        keep = min(len(idiap_rows), target)
        idiap_out.extend(idiap_rows[:keep])

        kan_need = target - keep
        if kan_need > 0:
            pool = kan_pool.get(age, [])
            for item in pool[:kan_need]:
                kan_out.append({"clip": item["clip"], "age": age, "duration": item["duration"]})
            kan_pool[age] = pool[kan_need:]

    return idiap_out, kan_out, allocation


def write_split_csv(idiap_rows, kan_rows, idiapage_dir, kanav_dir, output_path):
    rows, idx = [], 1
    for row in idiap_rows:
        rows.append({
            "id":         idx,
            "audio_path": os.path.join(idiapage_dir, row["audio_path"]),
            "image_path": os.path.join(idiapage_dir, row["image_path"]),
            "text":       row["text"],
            "language":   row.get("language", "en"),
            "duration":   row.get("duration", ""),
        })
        idx += 1
    for item in kan_rows:
        clip = item["clip"]
        rows.append({
            "id":         idx,
            "audio_path": os.path.join(kanav_dir, "audio", clip + ".wav"),
            "image_path": os.path.join(kanav_dir, "frames", clip + "__s001.jpg"),
            "text":       str(item["age"]),
            "language":   "en",
            "duration":   str(item["duration"]),
        })
        idx += 1
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "audio_path", "image_path", "text", "language", "duration"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load KAN pool from manifest CSV
    kan_by_age = defaultdict(list)
    with open(os.path.join(args.kanav_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            age = int(row["text"])
            kan_by_age[age].append({"clip": row["audio_path"].replace("audio/", "").replace(".wav", ""),
                                    "duration": row.get("duration", "")})
    for age in kan_by_age:
        random.shuffle(kan_by_age[age])

    # Exclude eval clips
    used_clips = set()
    for csv_path in [args.boundary_csv, args.balanced_csv]:
        if csv_path and os.path.exists(csv_path):
            with open(csv_path) as f:
                for row in csv.DictReader(f):
                    used_clips.add(row["audio_path"].replace("audio/", "").replace(".wav", ""))
    for age in kan_by_age:
        kan_by_age[age] = [c for c in kan_by_age[age] if c["clip"] not in used_clips]

    print(f"KAN pool after exclusions: {sum(len(v) for v in kan_by_age.values())} clips")

    train_by_age = load_rows_by_age(os.path.join(args.idiapage_dir, "train_formatted.csv"))
    val_by_age   = load_rows_by_age(os.path.join(args.idiapage_dir, "validation_formatted.csv"))

    kan_pool = {age: kan_by_age[age][:] for age in kan_by_age}

    train_idiap, train_kan, _ = build_uniform_split(train_by_age, kan_pool, 3000)
    val_idiap,   val_kan,   _ = build_uniform_split(val_by_age,   kan_pool, 500)

    train_clips = {i["clip"] for i in train_kan}
    val_clips   = {i["clip"] for i in val_kan}
    assert not (train_clips & val_clips), "Overlap between train and val KAN clips!"
    print("No KAN overlap between train/val: OK")

    n_train = write_split_csv(train_idiap, train_kan, args.idiapage_dir, args.kanav_dir,
                               os.path.join(args.output_dir, "train_formatted.csv"))
    n_val   = write_split_csv(val_idiap,   val_kan,   args.idiapage_dir, args.kanav_dir,
                               os.path.join(args.output_dir, "validation_formatted.csv"))

    # Test: pass through IDIAPAge test unchanged
    test_rows = []
    with open(os.path.join(args.idiapage_dir, "test_formatted.csv")) as f:
        for row in csv.DictReader(f):
            test_rows.append({
                "id":         row["id"],
                "audio_path": os.path.join(args.idiapage_dir, row["audio_path"]),
                "image_path": os.path.join(args.idiapage_dir, row["image_path"]),
                "text":       row["text"],
                "language":   row.get("language", "en"),
                "duration":   row.get("duration", ""),
            })
    with open(os.path.join(args.output_dir, "test_formatted.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "audio_path", "image_path", "text", "language", "duration"])
        w.writeheader()
        w.writerows(test_rows)

    print(f"\nDataset written to {args.output_dir}")
    for label, path in [("Train", "train_formatted.csv"), ("Val", "validation_formatted.csv"), ("Test", "test_formatted.csv")]:
        ages = Counter()
        with open(os.path.join(args.output_dir, path)) as f:
            for row in csv.DictReader(f):
                ages[int(row["text"])] += 1
        counts = [ages.get(a, 0) for a in range(0, 81)]
        nonzero = [c for c in counts if c > 0]
        arr = np.array(nonzero)
        total = sum(ages.values())
        minors = sum(c for a, c in ages.items() if a < 18)
        print(f"  {label:5s}: {total:4d} | {len(nonzero)}/81 ages | "
              f"per-age min={arr.min()} max={arr.max()} cv={arr.std()/arr.mean():.3f} | "
              f"minors={minors} ({100*minors/total:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--idiapage_dir",  required=True, help="Path to built IDIAPAge dataset")
    parser.add_argument("--kanav_dir",     required=True, help="Path to extracted KAN-AV dataset")
    parser.add_argument("--boundary_csv",  default=None,  help="boundary_500.csv to exclude from training")
    parser.add_argument("--balanced_csv",  default=None,  help="balanced_1000_uncropped.csv to exclude from training")
    parser.add_argument("--output_dir",    required=True, help="Output directory (uniformage_av)")
    main(parser.parse_args())
