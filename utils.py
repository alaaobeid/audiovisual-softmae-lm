"""
utils.py — inlined helpers from gemma_tuner (no external dependency required).

Provides:
  - extract_age(text) -> Optional[int]
  - apply_image_token_budget_to_processor(processor, budget)
  - install_kw_filter(module)
  - generate_for_audiovisual(model, processor, sample, device, max_new_tokens, image_token_budget)
"""
import re, ast, logging
from typing import Any, Optional

import torch
import librosa
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_GEMMA_UNEXPECTED_KWARGS = frozenset({
    "inputs_ids", "inputs_embeds", "decoder_inputs_embeds", "num_items_in_batch",
})


def extract_age(text: Optional[str]) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"\b(1[01][0-9]|120|[0-9]{1,2})\b", str(text))
    return int(m.group(1)) if m else None


def apply_image_token_budget_to_processor(processor: Any, budget: int) -> None:
    b = int(budget)
    if not hasattr(processor, "image_seq_length"):
        return
    if int(getattr(processor, "image_seq_length", 0)) == b:
        return
    processor.image_seq_length = b
    img_tok = getattr(processor, "image_token", None)
    boi     = getattr(processor, "boi_token", None)
    eoi     = getattr(processor, "eoi_token", None)
    if img_tok is not None and boi is not None and eoi is not None:
        processor.full_image_sequence = f"\n\n{boi}{''.join([str(img_tok)] * b)}{eoi}\n\n"


def install_kw_filter(module) -> None:
    try:
        _orig = module.forward
    except Exception:
        return

    def _filtered(*args, **kwargs):
        for k in _GEMMA_UNEXPECTED_KWARGS:
            kwargs.pop(k, None)
        return _orig(*args, **kwargs)

    try:
        module.forward = _filtered
    except Exception:
        logger.warning("Cannot patch .forward on %s", type(module).__name__)


def _load_audio(path: str, sr: int = 16000) -> np.ndarray:
    wav, orig_sr = librosa.load(path, sr=None, mono=True)
    if orig_sr != sr:
        wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=sr)
    return wav.astype(np.float32)


def _manual_greedy_generate(model, inputs: dict, max_new_tokens: int, eos_token_id=None):
    work = dict(inputs)
    work.pop("labels", None)
    input_ids      = work.pop("input_ids")
    attention_mask = work.pop("attention_mask", None)
    generated = []
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attention_mask, **work)
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_tok)
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        if attention_mask is not None:
            pad = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype,
                             device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, pad], dim=1)
        if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
            break
    if not generated:
        return torch.empty((0,), dtype=input_ids.dtype, device=input_ids.device)
    return torch.cat(generated, dim=1)[0]


def _prune_generate_kwargs(inputs: dict) -> None:
    for k in ("mm_token_type_ids", "image_position_ids", "input_features_mask", "is_longer"):
        inputs.pop(k, None)
    for k in list(inputs):
        if isinstance(inputs[k], bool):
            inputs.pop(k)


def generate_for_audiovisual(model, processor, sample: dict, device,
                              max_new_tokens: int = 8, image_token_budget: int = 280) -> str:
    sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    wav   = _load_audio(sample["audio_path"], sr)
    image = Image.open(sample["image_path"]).convert("RGB")

    messages = [[{
        "role": "user",
        "content": [
            {"type": "audio", "audio": wav},
            {"type": "image", "image": image},
            {"type": "text",  "text": "Estimate the person's age from their voice in the audio and their appearance in the image."},
        ],
    }]]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw_inputs = processor(text=prompt, audio=[wav], images=[[image]],
                           sampling_rate=sr, return_tensors="pt", padding=True)
    raw_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in raw_inputs.items()}

    inputs = dict(raw_inputs)
    _prune_generate_kwargs(inputs)

    eos_id = getattr(processor.tokenizer, "eos_token_id", None)
    in_len = int(inputs["input_ids"].shape[1])

    try:
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen_ids = out[0][in_len:]
    except (ValueError, AttributeError) as e:
        msg = str(e)
        # strip keys the model complained about and retry once
        m = re.search(r"\[(.*?)\]", msg)
        if m:
            try:
                for key in ast.literal_eval(f"[{m.group(1)}]"):
                    if isinstance(key, str):
                        inputs.pop(key, None)
            except Exception:
                pass
        try:
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen_ids = out[0][in_len:]
        except Exception:
            gen_ids = _manual_greedy_generate(model, raw_inputs, max_new_tokens, eos_id)

    return processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
