"""CPU-only tests for rankalloc.train.train.

This file didn't exist before this commit. Its absence is exactly how a real defect shipped
undetected: train()'s signature was silently missing `clip_mode`/`data_order_seed` while its body
already referenced them (a partially-applied find-and-replace patch), and a full 466-test green
suite caught none of it -- because nothing in the suite ever called train(). It surfaced only when
a user ran the real smoke test on a GPU and got `TypeError: train() got an unexpected keyword
argument 'clip_mode'`.

These tests run on CPU with a tiny fake model precisely so this class of bug -- a call site and a
function signature drifting apart -- is caught by `pytest -q` before it reaches a GPU session.
"""
import json
import random
import tempfile
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from rankalloc.data import TokenizedExample, collate_batch
from rankalloc.train import train


class _Adapter(nn.Module):
    """Named submodules `<owner>.lora_A` / `<owner>.lora_B`, matching a real PEFT adapter's naming
    closely enough for train.py's `name.rsplit(".lora_", 1)[0]` per_module grouping to exercise.
    """

    def __init__(self, d):
        super().__init__()
        self.lora_A = nn.Linear(d, d, bias=False)
        self.lora_B = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.lora_B(self.lora_A(x))


class _FakeModel(nn.Module):
    """Stands in for a PEFT-wrapped causal LM: forward(input_ids, labels, attention_mask) -> an
    object with `.loss`, and two independent trainable "adapters" for clip_mode='per_module'."""

    def __init__(self, vocab=32, d=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.embed.weight.requires_grad_(False)
        self.attn = _Adapter(d)
        self.mlp = _Adapter(d)

    def forward(self, input_ids, labels, attention_mask):
        x = self.embed(input_ids)
        x = x + self.attn(x) + self.mlp(x)
        logits = x.sum(dim=(-1, -2))  # (batch,) -- collapse both feature and sequence dims
        loss = ((logits - labels.float().mean(dim=-1)) ** 2).mean()
        return types.SimpleNamespace(loss=loss)


def _examples(n=6, vocab=32, min_len=3, max_len=7, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        length = rng.randint(min_len, max_len)
        ids = [rng.randrange(vocab) for _ in range(length)]
        out.append(TokenizedExample(input_ids=ids, labels=ids, attention_mask=[1] * length))
    return out


def test_train_accepts_the_arguments_run_single_passes():
    """The direct regression test for the shipped bug: bind the exact keyword set run_single.py's
    build_allocation/train_loop call site uses against train's real signature. Raises TypeError
    immediately if any keyword is unknown, without needing to execute a training loop at all.
    """
    import inspect

    sig = inspect.signature(train)
    sig.bind(
        object(), [], 0,
        max_steps=1, lr=1e-3, warmup_ratio=0.0, max_grad_norm=1.0, micro_batch=2,
        device="cpu", run_dir="unused", eval_every=1, eval_fn=None,
        clip_mode="global", data_order_seed=None,
    )


@pytest.mark.parametrize("clip_mode", ["global", "per_module", "none"])
def test_train_runs_on_cpu_for_every_clip_mode(tmp_path, clip_mode):
    model = _FakeModel()
    result = train(
        model, _examples(), pad_token_id=0, max_steps=3, lr=1e-2, warmup_ratio=0.0,
        max_grad_norm=1.0, micro_batch=2, device="cpu", run_dir=tmp_path,
        eval_every=1, eval_fn=None, clip_mode=clip_mode,
    )
    assert result.status == "ok"
    assert result.final_step == 3
    log_path = tmp_path / "step_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:
        row = json.loads(line)
        assert set(row) >= {"step", "loss", "lr", "grad_norm", "tokens_seen", "cumulative_gpu_seconds"}


def test_train_rejects_unknown_clip_mode():
    model = _FakeModel()
    with pytest.raises(ValueError, match="clip_mode"):
        train(
            model, _examples(), pad_token_id=0, max_steps=1, lr=1e-2, warmup_ratio=0.0,
            max_grad_norm=1.0, micro_batch=2, device="cpu", run_dir="unused",
            clip_mode="bogus",
        )


def test_eval_fn_called_at_eval_every_and_final_step(tmp_path):
    calls = []

    def eval_fn(model):
        calls.append(1)
        return {"loss_token_weighted": 0.0, "loss_example_mean": 0.0}

    train(
        _FakeModel(), _examples(), pad_token_id=0, max_steps=5, lr=1e-2, warmup_ratio=0.0,
        max_grad_norm=1.0, micro_batch=2, device="cpu", run_dir=tmp_path,
        eval_every=2, eval_fn=eval_fn,
    )
    assert len(calls) == 3  # steps 2, 4, and the final step 5 (not a multiple of eval_every)


def test_data_order_seed_none_preserves_original_batch_order(monkeypatch, tmp_path):
    """I3 (compute/data-matching invariant, README L4): leaving data_order_seed unset must mean
    every run consumes identical data in identical order -- verified here by spying on the actual
    batches train() builds, not by inference from reading the source.
    """
    examples = _examples(n=6)
    seen_batches = []
    import rankalloc.train as train_mod

    real_collate = train_mod.collate_batch

    def spy_collate(batch, pad_token_id, device):
        seen_batches.append(list(batch))
        return real_collate(batch, pad_token_id, device)

    monkeypatch.setattr(train_mod, "collate_batch", spy_collate)
    train(
        _FakeModel(), examples, pad_token_id=0, max_steps=2, lr=1e-2, warmup_ratio=0.0,
        max_grad_norm=1.0, micro_batch=2, device="cpu", run_dir=tmp_path, data_order_seed=None,
    )
    assert seen_batches[0] == examples[0:2]
    assert seen_batches[1] == examples[2:4]


def test_data_order_seed_shuffles_and_is_reproducible(monkeypatch, tmp_path):
    examples = _examples(n=6)
    expected_order = list(examples)
    random.Random(11).shuffle(expected_order)

    import rankalloc.train as train_mod

    real_collate = train_mod.collate_batch
    runs = []
    for i in range(2):  # same seed, twice -- must reproduce identically
        seen = []

        def spy_collate(batch, pad_token_id, device, _seen=seen):
            _seen.append(list(batch))
            return real_collate(batch, pad_token_id, device)

        monkeypatch.setattr(train_mod, "collate_batch", spy_collate)
        train(
            _FakeModel(), list(examples), pad_token_id=0, max_steps=1, lr=1e-2, warmup_ratio=0.0,
            max_grad_norm=1.0, micro_batch=2, device="cpu", run_dir=tmp_path / str(i),
            data_order_seed=11,
        )
        runs.append(seen)

    assert runs[0] == runs[1]  # same seed -> identical batch order both times
    assert runs[0][0] == expected_order[0:2]  # matches Python's own random.Random(11).shuffle
    assert runs[0][0] != examples[0:2]  # and actually differs from the unshuffled order
