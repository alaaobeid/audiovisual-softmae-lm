#!/usr/bin/env python3
"""
preprocess_idiapage.py

Processes downloaded IDIAPAge video segments into:
  - audio/{stem}.wav                  (16kHz mono, via ffmpeg)
  - frames_uncropped/{stem}__s{K}.jpg (full uncropped frame, only if a face is detected)

Frame sampling: 1fps at the middle of each second — frame_idx = (sec - 0.5) * video_fps.
Frames with no detected face are skipped. The first passing frame is the one used
downstream by format_idiapage_splits.py.

Requires: ffmpeg, retina-face, opencv-python

Usage:
    python data_prep/preprocess_idiapage.py \
        --video_dir   video_segments \
        --output_dir  idiapage_processed
"""
import os, argparse, subprocess
import cv2
import numpy as np
from retinaface import RetinaFace
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mkv", ".avi", ".mov"}
DEFAULT_MAX_SIZE = 800


def extract_audio(video_path, out_path, sr=16000):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", str(sr), out_path],
        capture_output=True,
    )
    return result.returncode == 0


def has_face(frame_bgr):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    scale = DEFAULT_MAX_SIZE / max(h, w)
    if scale < 1.0:
        img_rgb = cv2.resize(img_rgb, (int(w * scale), int(h * scale)))
    faces = RetinaFace.detect_faces(img_rgb)
    return isinstance(faces, dict) and len(faces) > 0


def extract_frames(video_path, out_dir, stem):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0

    fps       = cap.get(cv2.CAP_PROP_FPS)
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = int(total / fps) if fps > 0 else 0

    os.makedirs(out_dir, exist_ok=True)
    count = 0
    for sec in range(1, duration_s + 1):
        frame_idx = int((sec - 0.5) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        if not has_face(frame):
            continue
        out_path = os.path.join(out_dir, f"{stem}__s{str(sec).zfill(3)}.jpg")
        cv2.imwrite(out_path, frame)
        count += 1

    cap.release()
    return count


def main(args):
    audio_dir  = os.path.join(args.output_dir, "audio")
    frames_dir = os.path.join(args.output_dir, "frames_uncropped")
    os.makedirs(audio_dir,  exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    entries = []
    for fname in os.listdir(args.video_dir):
        if fname.startswith("._"):
            continue
        stem, ext = os.path.splitext(fname)
        if ext.lower() in VIDEO_EXTENSIONS:
            entries.append((fname, stem))

    failed, no_face = [], []
    for fname, stem in tqdm(entries):
        video_path = os.path.join(args.video_dir, fname)
        audio_path = os.path.join(audio_dir, stem + ".wav")

        if not extract_audio(video_path, audio_path):
            failed.append(stem)
            continue

        n = extract_frames(video_path, frames_dir, stem)
        if n == 0:
            no_face.append(stem)

    print(f"Done. {len(entries) - len(failed)} processed. "
          f"{len(failed)} audio failures. "
          f"{len(no_face)} clips with no face detected in any frame.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir",   required=True)
    parser.add_argument("--output_dir",  required=True)
    main(parser.parse_args())
