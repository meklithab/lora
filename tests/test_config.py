import textwrap

import pytest

from rankalloc.config import (
    ConfigError,
    RunConfig,
    apply_overrides,
    from_json,
    from_yaml,
    run_id,
    to_json,
)


def write_yaml(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def test_unknown_top_level_key_raises(tmp_path):
    path = write_yaml(tmp_path, "seed: 1\nnot_a_real_key: 5\n")
    with pytest.raises(ConfigError):
        from_yaml(path)


def test_unknown_nested_key_raises(tmp_path):
    path = write_yaml(tmp_path, "optim:\n  lr: 0.001\n  not_a_real_key: 5\n")
    with pytest.raises(ConfigError):
        from_yaml(path)


def test_known_keys_load_fine(tmp_path):
    path = write_yaml(
        tmp_path,
        """\
        seed: 3
        optim:
          lr: 0.0003
          max_steps: 100
        alloc:
          strategy: gradnorm_prop
          temperature: 0.5
        """,
    )
    cfg = from_yaml(path)
    assert cfg.seed == 3
    assert cfg.optim.lr == 0.0003
    assert cfg.optim.max_steps == 100
    assert cfg.alloc.strategy == "gradnorm_prop"
    assert cfg.alloc.temperature == 0.5
    # untouched fields keep their defaults
    assert cfg.optim.max_grad_norm == 1.0


def test_run_id_stable_under_key_reordering(tmp_path):
    # build two YAML files with identical content but different key order
    path_a = tmp_path / "a.yaml"
    path_a.write_text("seed: 7\noptim:\n  lr: 0.0002\n  max_steps: 50\n")
    path_b = tmp_path / "b.yaml"
    path_b.write_text("optim:\n  max_steps: 50\n  lr: 0.0002\nseed: 7\n")

    cfg_a = from_yaml(path_a)
    cfg_b = from_yaml(path_b)
    assert run_id(cfg_a) == run_id(cfg_b)


def test_run_id_changes_when_any_value_changes():
    base = RunConfig()
    changed_seed = apply_overrides(base, ["seed=1"])
    changed_lr = apply_overrides(base, ["optim.lr=0.0009"])
    changed_temp = apply_overrides(base, ["alloc.temperature=0.9"])

    ids = {run_id(base), run_id(changed_seed), run_id(changed_lr), run_id(changed_temp)}
    assert len(ids) == 4


def test_run_id_includes_seed():
    base = RunConfig(seed=0)
    other_seed = RunConfig(seed=1)
    assert run_id(base) != run_id(other_seed)


def test_run_id_deterministic_repeated_calls():
    cfg = RunConfig(seed=42)
    assert run_id(cfg) == run_id(cfg)


def test_to_json_round_trips():
    cfg = RunConfig(seed=5)
    cfg = apply_overrides(cfg, ["alloc.strategy=random", "optim.lr=0.00015", "alloc.probe_id=abc123"])
    restored = from_json(to_json(cfg))
    assert restored == cfg
    assert to_json(restored) == to_json(cfg)


def test_apply_overrides_coerces_types():
    cfg = RunConfig()
    cfg2 = apply_overrides(cfg, ["optim.lr=1e-4", "alloc.temperature=0.5", "eval.run_generation=false"])
    assert cfg2.optim.lr == 1e-4
    assert isinstance(cfg2.optim.lr, float)
    assert cfg2.alloc.temperature == 0.5
    assert cfg2.eval.run_generation is False


def test_apply_overrides_unknown_key_raises():
    cfg = RunConfig()
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["optim.not_a_field=1"])


def test_apply_overrides_malformed_raises():
    cfg = RunConfig()
    with pytest.raises(ConfigError):
        apply_overrides(cfg, ["optim.lr"])


def test_optional_field_override_to_none():
    cfg = RunConfig()
    cfg2 = apply_overrides(cfg, ["alloc.probe_id=none"])
    assert cfg2.alloc.probe_id is None
