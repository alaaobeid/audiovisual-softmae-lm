#!/usr/bin/env python3
"""
format_idiapage_splits.py

Converts the Zenodo IDIAPAge split CSVs (clip, age, source, split) into the
formatted CSVs expected by build_uniformage_dataset.py (id, audio_path,
image_path, text, language, duration).

For each clip, picks the first available frame from frames_uncropped/.
Clips with no extracted audio or no extracted frames are skipped.

Zenodo CSV format:
    clip,age,source,split
    LwBNhA2ks5c_0_10_age4,4,current,train

Usage:
    python data_prep/format_idiapage_splits.py \
        --zenodo_csv   /path/to/zenodo/train.csv \
        --processed_dir idiapage_processed \
        --output_csv   idiapage_processed/train_formatted.csv
"""
import os, csv, re, argparse
from collections import defaultdict


def main(args):
    # Index all available frames by clip stem
    frames_dir = os.path.join(args.processed_dir, "frames_uncropped")
    audio_dir  = os.path.join(args.processed_dir, "audio")

    frames_by_clip = defaultdict(list)
    if os.path.isdir(frames_dir):
        for fname in os.listdir(frames_dir):
            if not fname.endswith(".jpg"):
                continue
            m = re.match(r"^(.+)__s(\d+)\.jpg$", fname)
            if m:
                frames_by_clip[m.group(1)].append(fname)
    for clip in frames_by_clip:
        frames_by_clip[clip].sort()

    rows = []
    skipped = 0
    with open(args.zenodo_csv) as f:
        for idx, row in enumerate(csv.DictReader(f), 1):
            clip = row["clip"]
            age  = row["age"]

            audio_path = os.path.join("audio", clip + ".wav")
            if not os.path.exists(os.path.join(args.processed_dir, audio_path)):
                skipped += 1
                continue

            frames = frames_by_clip.get(clip, [])
            if not frames:
                skipped += 1
                continue

            first_frame = frames[0]
            image_path = os.path.join("frames_uncropped", first_frame)

            rows.append({
                "id":         idx,
                "audio_path": audio_path,
                "image_path": image_path,
                "text":       age,
                "language":   "en",
                "duration":   "",
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "audio_path", "image_path", "text", "language", "duration"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {args.output_csv}: {len(rows)} samples ({skipped} skipped — missing audio or frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zenodo_csv",    required=True, help="Zenodo split CSV (clip,age,source,split)")
    parser.add_argument("--processed_dir", required=True, help="Output of preprocess_idiapage.py")
    parser.add_argument("--output_csv",    required=True, help="Path to write formatted CSV")
    main(parser.parse_args())
