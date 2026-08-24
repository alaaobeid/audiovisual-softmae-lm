#!/usr/bin/env python3
"""
download_idiapage_segments.py

Downloads IDIAPAge YouTube video segments from the Zenodo split CSVs.
Output: video_segments/{clip}.mp4

Zenodo CSV format:
    clip,age,source,split
    LwBNhA2ks5c_0_10_age4,4,current,train

Clip stem encodes: {youtube_id}_{start}_{end}_age{N}
  e.g. LwBNhA2ks5c_0_10_age4 -> youtube_id=LwBNhA2ks5c, start=0, end=10

Requires: yt-dlp, ffmpeg

Usage:
    python data_prep/download_idiapage_segments.py \
        --splits train.csv val.csv test.csv \
        --output_dir video_segments
"""
import csv, os, re, subprocess, argparse

CLIP_PATTERN = re.compile(r"^(.+)_(\d+)_(\d+)_age\d+$")


def parse_clip(clip):
    m = CLIP_PATTERN.match(clip)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))


def download_segment(clip, out_dir):
    youtube_id, start, end = parse_clip(clip)
    if youtube_id is None:
        print(f"WARN: cannot parse clip stem: {clip}")
        return False

    out_file = os.path.join(out_dir, clip + ".mp4")
    if os.path.exists(out_file):
        return True

    tmp_vid = f"/tmp/{youtube_id}.mp4"
    if not os.path.exists(tmp_vid):
        dl = subprocess.run(
            ["yt-dlp",
             "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
             "-o", tmp_vid, f"https://www.youtube.com/watch?v={youtube_id}"],
            capture_output=True, text=True,
        )
        if not os.path.exists(tmp_vid):
            reason = dl.stderr.splitlines()[-1] if dl.stderr.strip() else "unknown"
            print(f"FAILED: {youtube_id} — {reason}")
            return False

    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_vid,
         "-ss", str(start), "-to", str(end), "-c", "copy", out_file],
        capture_output=True,
    )
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)
    return os.path.exists(out_file)


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    failed = []

    for split_file in args.splits:
        if not os.path.exists(split_file):
            print(f"Skipping missing: {split_file}")
            continue
        with open(split_file) as f:
            rows = list(csv.DictReader(f))
        print(f"\n=== {split_file}: {len(rows)} segments ===")
        for row in rows:
            if not download_segment(row["clip"], args.output_dir):
                failed.append(f"{split_file},{row['clip']}")

    print(f"\nDone. {len(failed)} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits",      nargs="+", required=True)
    parser.add_argument("--output_dir",  default="video_segments")
    main(parser.parse_args())
