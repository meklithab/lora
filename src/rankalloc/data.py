"""Load GSM8K (primary) and Alpaca (probe-only), build deterministic splits, and apply response-only
loss masking. See BUILD_SPEC.md §4.2.

Response-only masking uses the prefix-length trick: tokenize the prompt alone and the full
prompt+response conversation, then mask every label position covered by the prompt tokenization.
This only works if the tokenizer produces the same tokens for the shared prefix in both cases, which
we assert rather than assume: if the tokenizer's BPE merges differently across that boundary, we want
a loud AssertionError, not silently-wrong labels.
"""
import dataclasses
from dataclasses import dataclass
from typing import Dict, List, Optional

from datasets import load_dataset

from rankalloc.config import DataConfig

IGNORE_INDEX = -100

FALLBACK_TEMPLATE = "Question: {question}\nAnswer:"
FALLBACK_TEMPLATE_WITH_RESPONSE = "Question: {question}\nAnswer: {answer}"


@dataclass
class TokenizedExample:
    input_ids: List[int]
    labels: List[int]
    attention_mask: List[int]


@dataclass
class GenExample:
    prompt_input_ids: List[int]
    prompt_text: str
    answer: str


@dataclass
class TaskBundle:
    train: List[TokenizedExample]
    val_loss: List[TokenizedExample]
    val_gen: Optional[List[GenExample]]
    stats: Dict[str, object]


def _format_prompt(tokenizer, question: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return FALLBACK_TEMPLATE.format(question=question)


def _format_full(tokenizer, question: str, answer: str) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
            tokenize=False,
            add_generation_prompt=False,
        )
    return FALLBACK_TEMPLATE_WITH_RESPONSE.format(question=question, answer=answer)


def _tokenize_masked(tokenizer, question: str, answer: str, max_seq_len: int):
    prompt_text = _format_prompt(tokenizer, question)
    full_text = _format_full(tokenizer, question, answer)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    assert full_ids[: len(prompt_ids)] == prompt_ids, (
        "Chat-template tokenization is not prefix-stable: the prompt-only tokenization is not a "
        "prefix of the full-conversation tokenization. Response-only masking would be silently wrong."
    )

    truncated = False
    if len(full_ids) > max_seq_len:
        full_ids = full_ids[:max_seq_len]
        truncated = True

    labels = [IGNORE_INDEX] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids) :]
    attention_mask = [1] * len(full_ids)
    return TokenizedExample(input_ids=full_ids, labels=labels, attention_mask=attention_mask), truncated


def _split_gsm8k_answer(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip()


def _permuted_indices(n: int, seed: int) -> List[int]:
    import random as _random

    rng = _random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    return idx


def _load_gsm8k(cfg: DataConfig, tokenizer) -> TaskBundle:
    ds = load_dataset("openai/gsm8k", "main")
    train_ds = ds["train"]
    test_ds = ds["test"]

    train_order = _permuted_indices(len(train_ds), cfg.split_seed)
    test_order = _permuted_indices(len(test_ds), cfg.split_seed + 1)

    val_gen_n = min(cfg.val_gen_n, len(test_order))
    val_loss_n = min(cfg.val_loss_n, len(test_order) - val_gen_n)
    gen_idx = test_order[:val_gen_n]
    loss_idx = test_order[val_gen_n : val_gen_n + val_loss_n]

    train_keys = {f"train:{i}" for i in train_order}
    val_loss_keys = {f"test:{i}" for i in loss_idx}
    val_gen_keys = {f"test:{i}" for i in gen_idx}
    assert train_keys.isdisjoint(val_loss_keys)
    assert train_keys.isdisjoint(val_gen_keys)
    assert val_loss_keys.isdisjoint(val_gen_keys)

    truncated_count = 0
    total_tokens = 0
    supervised_tokens = 0

    train_examples = []
    for i in train_order:
        row = train_ds[i]
        ex, truncated = _tokenize_masked(tokenizer, row["question"], row["answer"], cfg.max_seq_len)
        train_examples.append(ex)
        truncated_count += int(truncated)
        total_tokens += len(ex.input_ids)
        supervised_tokens += sum(1 for l in ex.labels if l != IGNORE_INDEX)

    val_loss_examples = []
    for i in loss_idx:
        row = test_ds[i]
        ex, truncated = _tokenize_masked(tokenizer, row["question"], row["answer"], cfg.max_seq_len)
        val_loss_examples.append(ex)
        truncated_count += int(truncated)

    val_gen_examples = []
    for i in gen_idx:
        row = test_ds[i]
        prompt_text = _format_prompt(tokenizer, row["question"])
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        val_gen_examples.append(
            GenExample(
                prompt_input_ids=prompt_ids,
                prompt_text=prompt_text,
                answer=_split_gsm8k_answer(row["answer"]),
            )
        )

    stats = {
        "task": "gsm8k",
        "chat_template_used": bool(tokenizer.chat_template),
        "fallback_template": None if tokenizer.chat_template else FALLBACK_TEMPLATE_WITH_RESPONSE,
        "n_train": len(train_examples),
        "n_val_loss": len(val_loss_examples),
        "n_val_gen": len(val_gen_examples),
        "truncated_count": truncated_count,
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "max_seq_len": cfg.max_seq_len,
        "split_seed": cfg.split_seed,
    }
    return TaskBundle(train=train_examples, val_loss=val_loss_examples, val_gen=val_gen_examples, stats=stats)


def _load_alpaca(cfg: DataConfig, tokenizer) -> TaskBundle:
    ds = load_dataset("tatsu-lab/alpaca")["train"]
    order = _permuted_indices(len(ds), cfg.split_seed)

    truncated_count = 0
    total_tokens = 0
    supervised_tokens = 0
    train_examples = []
    for i in order:
        row = ds[i]
        question = row["instruction"]
        if row.get("input"):
            question = f"{question}\n\n{row['input']}"
        ex, truncated = _tokenize_masked(tokenizer, question, row["output"], cfg.max_seq_len)
        train_examples.append(ex)
        truncated_count += int(truncated)
        total_tokens += len(ex.input_ids)
        supervised_tokens += sum(1 for l in ex.labels if l != IGNORE_INDEX)

    stats = {
        "task": "alpaca",
        "chat_template_used": bool(tokenizer.chat_template),
        "fallback_template": None if tokenizer.chat_template else FALLBACK_TEMPLATE_WITH_RESPONSE,
        "n_train": len(train_examples),
        "n_val_loss": 0,
        "n_val_gen": 0,
        "truncated_count": truncated_count,
        "total_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "max_seq_len": cfg.max_seq_len,
        "split_seed": cfg.split_seed,
    }
    return TaskBundle(train=train_examples, val_loss=[], val_gen=None, stats=stats)


def load_task(cfg: DataConfig, tokenizer) -> TaskBundle:
    if cfg.task == "gsm8k":
        return _load_gsm8k(cfg, tokenizer)
    if cfg.task == "alpaca":
        return _load_alpaca(cfg, tokenizer)
    raise ValueError(f"Unknown task: {cfg.task!r}")
