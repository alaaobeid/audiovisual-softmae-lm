"""
train.py — SoftMAE fine-tuning of Gemma 4 E4B for audio-visual age estimation.

Every sample is trained in all 3 modalities (AV + image-only + audio-only).
3000 samples × 3 = 9000 training examples per epoch.

Environment variables:
  LAMBDA        (default 0.9)    — weight of soft-MAE loss term (λ in paper)
  NUM_EPOCHS    (default 5)
  SEED          (default 3407)
  LEARNING_RATE (default 2e-4)
  LORA_R        (default 16)
  LORA_ALPHA    (default 32)

Usage:
    LAMBDA=0.9 SEED=7331 python train.py \
        --dataset_dir uniformage_av \
        --boundary_val_csv boundary_500_dataset/boundary_val.csv \
        --boundary_val_dir boundary_500_dataset \
        --output_dir output/run_alpha009_s7331
"""
import os, sys, logging, random, json, argparse
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

import numpy as np
import librosa
import torch
from PIL import Image
from datasets import load_dataset
from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

from utils import (
    extract_age, apply_image_token_budget_to_processor,
    install_kw_filter, generate_for_audiovisual,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)], level=logging.INFO)
logger = logging.getLogger(__name__)

# ── model ──────────────────────────────────────────────────────────────────────
MODEL_ID       = "unsloth/gemma-4-E4B-it"
MAX_SEQ_LENGTH = 8192
IMAGE_TOKEN_BUDGET = 280

ALL_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "post", "linear_start", "linear_end", "embedding_projection",
    "ffw_layer_1", "ffw_layer_2", "output_proj",
]

SYSTEM_PROMPT = (
    "You are an expert at estimating a person's age from their voice and appearance. "
    "Reply with only the age as a single integer."
)

# ── hyperparams from env ───────────────────────────────────────────────────────
LAMBDA        = float(os.environ.get("LAMBDA",        "0.9"))
NUM_EPOCHS    = int(os.environ.get("NUM_EPOCHS",    "5"))
SEED          = int(os.environ.get("SEED",          "3407"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
LORA_R        = int(os.environ.get("LORA_R",        "16"))
LORA_ALPHA_HP = int(os.environ.get("LORA_ALPHA",    "32"))


# ── SoftMAE trainer ────────────────────────────────────────────────────────────
class SoftMAETrainer(SFTTrainer):
    def __init__(self, *args, lmbda=0.9, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lmbda = lmbda
        self._tokenizer = tokenizer
        # Precompute: for each age 0-99, the first token id of its string representation
        self._age_token_ids = {
            age: tokenizer.encode(str(age), add_special_tokens=False)
            for age in range(100)
        }

    def _compute_soft_mae(self, logits, labels):
        """
        For each sample in the batch, find the position just before the age token,
        extract a closed softmax over the 100 age-token logits, compute E[age],
        and return (soft_ages, true_ages).

        Note: uses the first token of each age string as a proxy score. For single-digit
        ages, each has a unique single-token encoding, so the construction is exact
        dot-product NTL-MAE for those ages. For multi-digit ages, tokenizers that lack a
        dedicated single-token representation decompose them into individual digit tokens,
        causing multiple ages to share the same first token and receive identical scores —
        a known limitation documented in the paper.
        """
        device = logits.device
        age_values = torch.arange(100, dtype=torch.float32, device=device)
        soft_ages, true_ages = [], []

        for i in range(logits.shape[0]):
            label_mask = labels[i] != -100
            positions  = label_mask.nonzero(as_tuple=True)[0]
            if len(positions) == 0 or positions[0].item() == 0:
                continue
            valid_tokens = labels[i][label_mask]
            text = self._tokenizer.decode(valid_tokens, skip_special_tokens=True).strip()
            try:
                true_age = int(text)
                if not (0 <= true_age <= 99):
                    continue
            except ValueError:
                continue

            pred_logits = logits[i, positions[0].item() - 1]
            age_scores  = torch.tensor(
                [pred_logits[self._age_token_ids[a][0]].item() for a in range(100)],
                device=device, dtype=torch.float32,
            )
            soft_age = (torch.softmax(age_scores, dim=0) * age_values).sum()
            soft_ages.append(soft_age)
            true_ages.append(torch.tensor(true_age, device=device, dtype=torch.float32))

        if not soft_ages:
            return None, None
        return torch.stack(soft_ages), torch.stack(true_ages)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        inputs["use_cache"] = False
        outputs = model(**inputs)
        ce_loss  = outputs.loss
        mae_loss = torch.tensor(0.0, device=ce_loss.device)

        if (self.lmbda > 0 and labels is not None
                and hasattr(outputs, "logits")
                and isinstance(outputs.logits, torch.Tensor)):
            soft_ages, true_ages = self._compute_soft_mae(outputs.logits, labels)
            if soft_ages is not None:
                mae_loss = torch.nn.functional.l1_loss(soft_ages, true_ages)

        total_loss = ce_loss + self.lmbda * mae_loss
        return (total_loss, outputs) if return_outputs else total_loss


# ── boundary-val early stopping ────────────────────────────────────────────────
class BoundaryValCallback(TrainerCallback):
    def __init__(self, model, processor, boundary_val_ds, output_dir, device,
                 patience=3, min_epochs=3, min_delta_pct=0.01):
        self.model = model
        self.processor = processor
        self.boundary_val_ds = boundary_val_ds
        self.output_dir = output_dir
        self.device = device
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta_pct = min_delta_pct
        self.best_mae = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.history = []

        history_path = os.path.join(output_dir, "boundary_val_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    saved = json.load(f)
                self.history    = saved.get("history", [])
                self.best_mae   = saved.get("best_mae", float("inf"))
                self.best_epoch = saved.get("best_epoch", -1)
                for entry in reversed(self.history):
                    if entry["boundary_val_mae"] < self.best_mae * (1.0 + self.min_delta_pct):
                        break
                    self.epochs_without_improvement += 1
                logger.info("Resumed: best_mae=%.3f ep%d patience=%d/%d",
                            self.best_mae, self.best_epoch,
                            self.epochs_without_improvement, self.patience)
            except Exception as e:
                logger.warning("Could not restore boundary_val history: %s", e)

    def on_epoch_end(self, args, state, control, **kwargs):
        self.model.eval()
        errors = []
        with torch.no_grad():
            for sample in self.boundary_val_ds:
                try:
                    pred  = generate_for_audiovisual(self.model, self.processor, sample,
                                                     self.device, max_new_tokens=8,
                                                     image_token_budget=IMAGE_TOKEN_BUDGET)
                    p_age = extract_age(pred)
                    if p_age is not None:
                        errors.append(abs(p_age - int(sample["text"])))
                except Exception as e:
                    logger.warning("Boundary val sample failed: %s", e)
                    pass

        mae      = sum(errors) / len(errors) if errors else float("inf")
        coverage = len(errors) / len(self.boundary_val_ds)
        epoch    = int(state.epoch)
        self.history.append({"epoch": epoch, "boundary_val_mae": round(mae, 3),
                              "coverage": round(coverage, 3)})

        threshold = self.best_mae * (1.0 - self.min_delta_pct)
        if mae < threshold:
            self.best_mae   = mae
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            logger.info("  New best boundary_val MAE: %.3f at epoch %d (cov=%.1f%%)",
                        mae, epoch, 100 * coverage)
        else:
            self.epochs_without_improvement += 1
            logger.info("  Boundary_val MAE: %.3f ep%d (best=%.3f ep%d, patience %d/%d)",
                        mae, epoch, self.best_mae, self.best_epoch,
                        self.epochs_without_improvement, self.patience)

        with open(os.path.join(self.output_dir, "boundary_val_history.json"), "w") as f:
            json.dump({"history": self.history, "best_epoch": self.best_epoch,
                       "best_mae": round(self.best_mae, 3)}, f, indent=2)

        if epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience:
            logger.info("  Early stopping: best ep%d MAE=%.3f", self.best_epoch, self.best_mae)
            control.should_training_stop = True

        self.model.train()


# ── conversation builders ──────────────────────────────────────────────────────
def _load_audio(path, sr=16000):
    wav, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def to_av(sample, dataset_dir):
    audio = _load_audio(os.path.join(dataset_dir, sample["audio_path"]))
    image = Image.open(os.path.join(dataset_dir, sample["image_path"])).convert("RGB")
    age   = str(sample["text"]).strip()
    return {"messages": [
        {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": [
            {"type": "audio", "audio": audio},
            {"type": "image", "image": image},
            {"type": "text",  "text": "Estimate the person's age from their voice in the audio and their appearance in the image. Reply with a single integer only."},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": age}]},
    ]}


def to_audio_only(sample, dataset_dir):
    audio = _load_audio(os.path.join(dataset_dir, sample["audio_path"]))
    blank = Image.new("RGB", (1, 1), color=(0, 0, 0))
    age   = str(sample["text"]).strip()
    return {"messages": [
        {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": [
            {"type": "audio", "audio": audio},
            {"type": "image", "image": blank},
            {"type": "text",  "text": "Estimate the person's age from their voice in the audio. Reply with a single integer only."},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": age}]},
    ]}


def to_image_only(sample, dataset_dir):
    image = Image.open(os.path.join(dataset_dir, sample["image_path"])).convert("RGB")
    age   = str(sample["text"]).strip()
    return {"messages": [
        {"role": "system",    "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",      "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": "Estimate the person's age from their appearance in the image. Reply with a single integer only."},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": age}]},
    ]}


# ── main ───────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("Config: lambda=%.3f epochs=%d seed=%d lr=%g output=%s",
                LAMBDA, NUM_EPOCHS, SEED, LEARNING_RATE, args.output_dir)

    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID, dtype=None, max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=False, full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=LORA_R, lora_alpha=LORA_ALPHA_HP, lora_dropout=0, bias="none",
        random_state=SEED, target_modules=ALL_TARGET_MODULES,
    )
    apply_image_token_budget_to_processor(processor, IMAGE_TOKEN_BUDGET)
    install_kw_filter(model)
    if hasattr(model, "base_model"):
        install_kw_filter(model.base_model)
        if hasattr(model.base_model, "model"):
            install_kw_filter(model.base_model.model)

    train_ds = load_dataset("csv", data_files=os.path.join(args.dataset_dir, "train_formatted.csv"), split="train")
    logger.info("Train: %d samples × 3 modalities = %d examples", len(train_ds), len(train_ds) * 3)

    logger.info("Formatting training conversations...")
    train_conversations = (
        [to_av(s, args.dataset_dir)         for s in train_ds] +
        [to_image_only(s, args.dataset_dir) for s in train_ds] +
        [to_audio_only(s, args.dataset_dir) for s in train_ds]
    )
    random.seed(SEED)
    random.shuffle(train_conversations)

    val_ds = load_dataset("csv", data_files=os.path.join(args.dataset_dir, "validation_formatted.csv"), split="train")
    val_conversations = [to_av(s, args.dataset_dir) for s in val_ds]

    boundary_val_ds = load_dataset("csv", data_files=args.boundary_val_csv, split="train")
    boundary_val_ds = boundary_val_ds.map(lambda b: {
        "image_path": [os.path.join(args.boundary_val_dir, p) for p in b["image_path"]],
        "audio_path": [os.path.join(args.boundary_val_dir, p) for p in b["audio_path"]],
    }, batched=True)
    logger.info("Boundary val: %d samples", len(boundary_val_ds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boundary_cb = BoundaryValCallback(model, processor, boundary_val_ds, args.output_dir, device)

    trainer = SoftMAETrainer(
        model=model,
        train_dataset=train_conversations,
        eval_dataset=val_conversations,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        lmbda=LAMBDA,
        tokenizer=processor.tokenizer,
        callbacks=[boundary_cb],
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_ratio=0.03,
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            eval_strategy="epoch",
            save_total_limit=NUM_EPOCHS,
            seed=SEED,
            report_to=[],
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=MAX_SEQ_LENGTH,
        ),
    )

    logger.info("Starting training...")
    trainer.train()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Done. Best boundary_val: ep%d MAE=%.3f (lambda=%.2f)", boundary_cb.best_epoch, boundary_cb.best_mae, LAMBDA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",      required=True, help="Path to uniformage_av/")
    parser.add_argument("--boundary_val_csv", required=True, help="Path to boundary_val.csv (100 samples)")
    parser.add_argument("--boundary_val_dir", required=True, help="Root dir of boundary_500_dataset/")
    parser.add_argument("--output_dir",       required=True, help="Where to save checkpoints")
    main(parser.parse_args())
