import pytest
from transformers import AutoTokenizer

from rankalloc.config import DataConfig
from rankalloc.data import IGNORE_INDEX, load_task

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@pytest.fixture(scope="module")
def gsm8k_bundle(tokenizer):
    cfg = DataConfig(task="gsm8k", max_seq_len=512, split_seed=12345, val_loss_n=10, val_gen_n=10)
    return load_task(cfg, tokenizer)


def test_at_least_20_examples_available(gsm8k_bundle):
    assert len(gsm8k_bundle.train) >= 20


def test_non_masked_count_equals_response_length(tokenizer, gsm8k_bundle):
    checked = 0
    for ex in gsm8k_bundle.train[:20]:
        non_masked = [l for l in ex.labels if l != IGNORE_INDEX]
        first_non_masked_idx = next(i for i, l in enumerate(ex.labels) if l != IGNORE_INDEX)
        response_ids = ex.input_ids[first_non_masked_idx:]
        assert non_masked == response_ids
        checked += 1
    assert checked >= 20


def test_first_non_masked_position_is_first_response_token(tokenizer, gsm8k_bundle):
    checked = 0
    for ex in gsm8k_bundle.train[:20]:
        first_non_masked_idx = next(i for i, l in enumerate(ex.labels) if l != IGNORE_INDEX)
        assert ex.labels[first_non_masked_idx] == ex.input_ids[first_non_masked_idx]
        # everything before it is masked
        assert all(l == IGNORE_INDEX for l in ex.labels[:first_non_masked_idx])
        checked += 1
    assert checked >= 20


def test_labels_same_length_as_input_ids(gsm8k_bundle):
    for ex in gsm8k_bundle.train[:20]:
        assert len(ex.labels) == len(ex.input_ids) == len(ex.attention_mask)


def test_three_way_split_disjointness(tokenizer):
    cfg = DataConfig(task="gsm8k", max_seq_len=512, split_seed=12345, val_loss_n=50, val_gen_n=50)
    bundle = load_task(cfg, tokenizer)
    assert bundle.stats["n_val_loss"] == 50
    assert bundle.stats["n_val_gen"] == 50
    # disjointness is asserted inside load_task itself; reaching here means it held
    assert len(bundle.val_gen) == 50


def test_val_gen_has_prompt_and_answer(gsm8k_bundle):
    assert gsm8k_bundle.val_gen is not None
    for ex in gsm8k_bundle.val_gen[:5]:
        assert len(ex.prompt_input_ids) > 0
        assert ex.answer != ""


def test_token_accounting_present(gsm8k_bundle):
    stats = gsm8k_bundle.stats
    assert stats["total_tokens"] > 0
    assert stats["supervised_tokens"] > 0
    assert stats["supervised_tokens"] < stats["total_tokens"]


def test_alpaca_probe_only_no_val_gen(tokenizer):
    cfg = DataConfig(task="alpaca", max_seq_len=512, split_seed=12345)
    bundle = load_task(cfg, tokenizer)
    assert len(bundle.train) >= 20
    assert bundle.val_gen is None
    assert bundle.val_loss == []


def test_deterministic_order_same_split_seed(tokenizer):
    cfg = DataConfig(task="gsm8k", max_seq_len=512, split_seed=999, val_loss_n=10, val_gen_n=10)
    b1 = load_task(cfg, tokenizer)
    b2 = load_task(cfg, tokenizer)
    ids1 = [ex.input_ids for ex in b1.train[:10]]
    ids2 = [ex.input_ids for ex in b2.train[:10]]
    assert ids1 == ids2


def test_different_split_seed_changes_order(tokenizer):
    cfg_a = DataConfig(task="gsm8k", max_seq_len=512, split_seed=1, val_loss_n=10, val_gen_n=10)
    cfg_b = DataConfig(task="gsm8k", max_seq_len=512, split_seed=2, val_loss_n=10, val_gen_n=10)
    b_a = load_task(cfg_a, tokenizer)
    b_b = load_task(cfg_b, tokenizer)
    first_ids_a = [ex.input_ids for ex in b_a.train[:5]]
    first_ids_b = [ex.input_ids for ex in b_b.train[:5]]
    assert first_ids_a != first_ids_b
