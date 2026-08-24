"""
evaluate.py — greedy inference on a fine-tuned checkpoint with full metrics.

Metrics reported:
  - Overall MAE and coverage
  - Per-age MAE (for boundary set: ages 14-21)
  - Minor detection (binary: true_age < 18 vs >= 18):
      accuracy, precision, recall, F1, confusion matrix

Usage:
    python evaluate.py \
        --checkpoint output/run_alpha009_s7331 \
        --eval_csv   boundary_500_dataset/boundary_test.csv \
        --eval_dir   boundary_500_dataset \
        --output     results/run_alpha009_s7331_boundary_test.csv

    python evaluate.py \
        --checkpoint output/run_alpha009_s7331 \
        --eval_csv   balanced_1000_dataset/balanced_1000.csv \
        --eval_dir   balanced_1000_dataset \
        --output     results/run_alpha009_s7331_balanced.csv
"""
import os, argparse, logging, sys
from collections import defaultdict
import torch
import pandas as pd
import numpy as np
from datasets import load_dataset
from unsloth import FastModel
from utils import extract_age, apply_image_token_budget_to_processor, generate_for_audiovisual

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_TOKEN_BUDGET = 280
MINOR_THRESHOLD = 18


def compute_metrics(df):
    df = df.dropna(subset=["pred_age"]).copy()
    df["true_age"]  = df["true_age"].astype(int)
    df["pred_age"]  = df["pred_age"].astype(int)
    df["error"]     = (df["pred_age"] - df["true_age"]).abs()

    mae      = df["error"].mean()
    coverage = len(df) / len(df)  # already filtered

    # per-age MAE
    per_age = df.groupby("true_age")["error"].mean().sort_index()

    # minor detection
    df["true_minor"] = df["true_age"] < MINOR_THRESHOLD
    df["pred_minor"] = df["pred_age"] < MINOR_THRESHOLD
    tp = ((df["true_minor"]) & (df["pred_minor"])).sum()
    tn = ((~df["true_minor"]) & (~df["pred_minor"])).sum()
    fp = ((~df["true_minor"]) & (df["pred_minor"])).sum()
    fn = ((df["true_minor"]) & (~df["pred_minor"])).sum()
    acc       = (tp + tn) / len(df)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "mae": mae, "coverage": coverage, "n": len(df),
        "per_age_mae": per_age.to_dict(),
        "minor_detection": {
            "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        },
    }


def main(args):
    logger.info("Loading checkpoint: %s", args.checkpoint)
    model, processor = FastModel.from_pretrained(
        model_name=args.checkpoint, dtype=None, max_seq_length=8192,
        load_in_4bit=False, full_finetuning=False,
    )
    apply_image_token_budget_to_processor(processor, IMAGE_TOKEN_BUDGET)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eval_ds = load_dataset("csv", data_files=args.eval_csv, split="train")
    eval_ds = eval_ds.map(lambda b: {
        "image_path": [os.path.join(args.eval_dir, p) for p in b["image_path"]],
        "audio_path": [os.path.join(args.eval_dir, p) for p in b["audio_path"]],
    }, batched=True)
    logger.info("Eval set: %d samples", len(eval_ds))

    rows = []
    with torch.no_grad():
        for i, sample in enumerate(eval_ds):
            try:
                pred  = generate_for_audiovisual(model, processor, sample, device,
                                                 max_new_tokens=8,
                                                 image_token_budget=IMAGE_TOKEN_BUDGET)
                p_age = extract_age(pred)
                t_age = int(sample["text"])
                rows.append({"id": sample.get("id", i), "true_age": t_age,
                             "pred_raw": pred, "pred_age": p_age})
            except Exception as e:
                logger.warning("Sample %d failed: %s", i, e)
                rows.append({"id": sample.get("id", i), "true_age": sample["text"],
                             "pred_raw": None, "pred_age": None})

            if (i + 1) % 50 == 0:
                logger.info("  %d/%d done", i + 1, len(eval_ds))

    df = pd.DataFrame(rows)
    n_parsed = df["pred_age"].notna().sum()
    coverage = n_parsed / len(df)

    # Only report minor detection if the eval set spans the minor/adult boundary
    # (i.e. boundary-style sets with ages 14-21). For broad age-range sets like
    # balanced_1000 the metric is not meaningful.
    valid = df.dropna(subset=["pred_age"])
    age_range = valid["true_age"].astype(int)
    is_boundary_set = int(age_range.min()) >= 14 and int(age_range.max()) <= 21

    metrics = compute_metrics(df)
    if not is_boundary_set:
        metrics.pop("minor_detection", None)

    logger.info("=" * 60)
    logger.info("MAE=%.3f  coverage=%.1f%%  n=%d/%d",
                metrics["mae"], 100 * coverage, n_parsed, len(df))
    if is_boundary_set:
        logger.info("Per-age MAE: %s",
                    "  ".join(f"{a}:{v:.2f}" for a, v in metrics["per_age_mae"].items()))
    if is_boundary_set:
        md = metrics["minor_detection"]
        logger.info("Minor detection (threshold=<%d):", MINOR_THRESHOLD)
        logger.info("  Acc=%.3f  Prec=%.3f  Rec=%.3f  F1=%.3f",
                    md["accuracy"], md["precision"], md["recall"], md["f1"])
        logger.info("  TP=%d  TN=%d  FP=%d  FN=%d",
                    md["tp"], md["tn"], md["fp"], md["fn"])
    logger.info("=" * 60)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df.to_csv(args.output, index=False)
        # also save metrics json alongside
        import json
        metrics_path = args.output.replace(".csv", "_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Predictions saved to %s", args.output)
        logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_csv",   required=True)
    parser.add_argument("--eval_dir",   required=True)
    parser.add_argument("--output",     default=None)
    main(parser.parse_args())
