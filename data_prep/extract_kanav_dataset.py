#!/usr/bin/env python3
"""
extract_kanav_dataset.py

Converts raw KAN-AV video/audio files into a flat dataset:
  - audio/{path}.wav        (copied from 02_audio_16k/)
  - frames/{path}__s001.jpg (first 1fps sample with a detected face, saved uncropped)
  - manifest CSV

Samples at 1fps (middle of each second). Saves the first frame that passes
RetinaFace detection. Clips where no 1fps sample contains a face are skipped.

Input sample list JSON format: [[relative_path, age_or_duration], ...]
  - balanced_*: second field is duration (float)
  - boundary_500: second field is age (int)

Usage:
    python extract_kanav_dataset.py \
        --kan_root    /path/to/KAN \
        --samples_json /path/to/samples.json \
        --output_dir  /path/to/output \
        --output_csv  manifest.csv
"""
import json, os, shutil, csv, subprocess, argparse
import cv2
import numpy as np
from retinaface import RetinaFace

DEFAULT_MAX_SIZE = 800


def has_face(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = DEFAULT_MAX_SIZE / max(h, w)
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
    faces = RetinaFace.detect_faces(img_rgb)
    return isinstance(faces, dict) and len(faces) > 0


def first_face_frame(video_path):
    """Return the first 1fps sample (as BGR ndarray) that contains at least one face.
    Returns None if no 1fps sample in the video passes detection."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps      = cap.get(cv2.CAP_PROP_FPS)
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total / fps) if fps > 0 else 0
    for sec in range(1, duration + 1):
        frame_idx = int((sec - 0.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if has_face(frame):
            cap.release()
            return frame
    cap.release()
    return None


def main(args):
    video_dir    = os.path.join(args.kan_root, "01_silent_video")
    audio_dir    = os.path.join(args.kan_root, "02_audio_16k")
    frames_out   = os.path.join(args.output_dir, "frames")
    audio_out    = os.path.join(args.output_dir, "audio")

    with open(args.samples_json) as f:
        samples = json.load(f)

    os.makedirs(frames_out, exist_ok=True)
    os.makedirs(audio_out,  exist_ok=True)

    rows = []
    skipped_no_face = 0
    for idx, (path, _) in enumerate(samples, 1):
        age        = int(path.split("/")[2])
        video_path = os.path.join(video_dir, path + ".avi")
        audio_src  = os.path.join(audio_dir, path + ".wav")
        frame_out  = os.path.join(frames_out, path + "__s001.jpg")
        audio_dest = os.path.join(audio_out,  path + ".wav")

        if not os.path.exists(video_path):
            print(f"WARN: video not found: {video_path}")
            continue
        if not os.path.exists(audio_src):
            print(f"WARN: audio not found: {audio_src}")
            continue

        frame = first_face_frame(video_path)
        if frame is None:
            skipped_no_face += 1
            continue

        os.makedirs(os.path.dirname(frame_out),  exist_ok=True)
        os.makedirs(os.path.dirname(audio_dest), exist_ok=True)
        cv2.imwrite(frame_out, frame)
        shutil.copy2(audio_src, audio_dest)

        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_src],
            capture_output=True, text=True,
        )
        duration = float(result.stdout.strip()) if result.stdout.strip() else 0.0

        rows.append([idx, f"audio/{path}.wav", f"frames/{path}__s001.jpg",
                     str(age), "en", f"{duration:.6f}"])

    csv_path = os.path.join(args.output_dir, args.output_csv)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "audio_path", "image_path", "text", "language", "duration"])
        w.writerows(rows)

    print(f"Done: {len(rows)} samples written to {csv_path} "
          f"({skipped_no_face} skipped — no face detected)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kan_root",     required=True)
    parser.add_argument("--samples_json", required=True)
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--output_csv",   default="manifest.csv")
    main(parser.parse_args())
