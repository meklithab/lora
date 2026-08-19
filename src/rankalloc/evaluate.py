"""Held-out loss and GSM8K generation evaluation. BUILD_SPEC.md §4.7."""
import re
from fractions import Fraction
from typing import Dict, List, Optional

import torch

from rankalloc.data import GenExample, TokenizedExample, collate_batch, collate_gen_batch

_NUM_RE = re.compile(r"-?[\d,]*\.?\d+")


def _normalize_number(raw: str) -> Optional[Fraction]:
    s = raw.strip().replace(",", "").replace("$", "").rstrip("%")
    if not s or s in ("-", "."):
        return None
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        try:
            return Fraction(str(float(s)))
        except ValueError:
            return None


def extract_strict(generation: str) -> Optional[Fraction]:
    if "####" not in generation:
        return None
    tail = generation.split("####")[-1]
    match = _NUM_RE.search(tail)
    return _normalize_number(match.group()) if match else None


def extract_flexible(generation: str) -> Optional[Fraction]:
    matches = _NUM_RE.findall(generation)
    return _normalize_number(matches[-1]) if matches else None


def _matches(pred: Optional[Fraction], gold: str) -> bool:
    if pred is None:
        return False
    gold_val = _normalize_number(gold)
    return gold_val is not None and pred == gold_val


@torch.no_grad()
def compute_held_out_loss(
    model,
    examples: List[TokenizedExample],
    pad_token_id: int,
    device,
    micro_batch: int = 8,
    use_amp: bool = False,
) -> Dict[str, float]:
    """loss_token_weighted = total NLL / total supervised tokens; loss_example_mean = mean of each
    example's own per-token NLL. They diverge under length imbalance -- both reported, see §4.7.
    """
    was_training = model.training
    model.eval()
    total_nll = 0.0
    total_supervised_tokens = 0
    example_losses = []
    for i in range(0, len(examples), micro_batch):
        batch = examples[i : i + micro_batch]
        input_ids, labels, attention_mask = collate_batch(batch, pad_token_id, device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        per_token_nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        supervised_mask = shift_labels != -100
        per_example_supervised = supervised_mask.sum(dim=1)
        per_example_nll = per_token_nll.sum(dim=1)
        total_nll += per_example_nll.sum().item()
        total_supervised_tokens += int(per_example_supervised.sum().item())
        for nll, n_tok in zip(per_example_nll.tolist(), per_example_supervised.tolist()):
            if n_tok > 0:
                example_losses.append(nll / n_tok)
    if was_training:
        model.train()
    return {
        "loss_token_weighted": total_nll / total_supervised_tokens if total_supervised_tokens else float("nan"),
        "loss_example_mean": sum(example_losses) / len(example_losses) if example_losses else float("nan"),
        "n_examples": len(examples),
        "supervised_tokens": total_supervised_tokens,
    }


@torch.no_grad()
def compute_gsm8k_generation(
    model,
    tokenizer,
    examples: List[GenExample],
    device,
    micro_batch: int = 8,
    max_new_tokens: int = 256,
    n_samples_to_log: int = 30,
) -> Dict:
    was_training = model.training
    model.eval()
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    strict_correct = 0
    flexible_correct = 0
    samples = []
    try:
        for i in range(0, len(examples), micro_batch):
            batch = examples[i : i + micro_batch]
            input_ids, attention_mask = collate_gen_batch(batch, tokenizer.pad_token_id, device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = generated[:, input_ids.shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for ex, text in zip(batch, texts):
                pred_strict = extract_strict(text)
                pred_flexible = extract_flexible(text)
                ok_strict = _matches(pred_strict, ex.answer)
                ok_flexible = _matches(pred_flexible, ex.answer)
                strict_correct += int(ok_strict)
                flexible_correct += int(ok_flexible)
                if len(samples) < n_samples_to_log:
                    samples.append(
                        {
                            "prompt": ex.prompt_text,
                            "generation": text,
                            "gold_answer": ex.answer,
                            "pred_strict": str(pred_strict) if pred_strict is not None else None,
                            "pred_flexible": str(pred_flexible) if pred_flexible is not None else None,
                            "correct_strict": ok_strict,
                            "correct_flexible": ok_flexible,
                        }
                    )
    finally:
        tokenizer.padding_side = original_padding_side
        if was_training:
            model.train()
    n = len(examples)
    return {
        "gsm8k_strict": strict_correct / n if n else float("nan"),
        "gsm8k_flexible": flexible_correct / n if n else float("nan"),
        "n_examples": n,
        "samples": samples,
    }
