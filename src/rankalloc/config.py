"""Config schema, YAML loading with strict key validation, dotted CLI overrides, and run_id hashing.

See BUILD_SPEC.md §4.1. `run_id` is the resumability primitive for run_grid.py: it hashes the fully
resolved config (including seed), so any change to any field -- including a CLI override -- produces
a different run_id, and re-running the exact same config produces the same one.
"""
import dataclasses
import hashlib
import json
import typing
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import yaml


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DataConfig:
    task: str = "gsm8k"  # "gsm8k" | "alpaca"
    max_seq_len: int = 512
    split_seed: int = 12345
    val_loss_n: int = 300
    val_gen_n: int = 200
    # Order in which training examples are consumed. None => derive from split_seed, i.e. every run
    # sees identical data in identical order. That is the compute/data-matching invariant (I3), but
    # it also means the seed-to-seed spread measures *initialisation* noise only and understates
    # true run-to-run variance. Set this to the run seed to fold data-order noise into the noise
    # floor -- see README "Limitations" L4.
    train_order_seed: Optional[int] = None


@dataclass(frozen=True)
class ProbeConfig:
    rank: int = 8
    steps: int = 100
    task: str = "gsm8k"  # "gsm8k" | "alpaca"
    # Freeze lora_A so dL/dB stays an unbiased random sketch of the frozen-weight gradient for the
    # whole probe (probe.py module docstring). Turning this off measures a different quantity.
    freeze_a: bool = True


@dataclass(frozen=True)
class AllocConfig:
    strategy: str = "uniform"  # uniform | gradnorm_prop | gradnorm_inverse | random | early_heavy | late_heavy
    budget_rank: int = 16
    signal: str = "rms"  # rms | raw_norm | fisher | relative | coherent
    temperature: float = 1.0
    r_min: int = 1
    r_max: int = 128
    lambda_decay: float = 1.0  # early_heavy / late_heavy exponent scale
    probe_id: Optional[str] = None


@dataclass(frozen=True)
class ScalingConfig:
    mode: str = "constant_ratio"  # constant_ratio | rslora | fixed_alpha
    alpha_ratio: int = 2
    fixed_alpha: int = 32


@dataclass(frozen=True)
class OptimConfig:
    lr: float = 2e-4
    max_steps: int = 400
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    # "global" clips the norm over the union of all LoRA parameters (standard practice, but it makes
    # each module's effective step depend on the *other* modules' gradients, which varies with the
    # allocation); "per_module" clips each adapter independently, removing that coupling; "none"
    # disables clipping. See README "Limitations" L3.
    clip_mode: str = "global"  # global | per_module | none
    micro_batch: int = 4  # calibrated on a real Tesla T4 in P0 preflight, see NOTES.md


@dataclass(frozen=True)
class EvalConfig:
    run_generation: bool = True
    max_new_tokens: int = 256


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    alloc: AllocConfig = field(default_factory=AllocConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _build_dataclass(cls, data: dict, path: str):
    if not isinstance(data, dict):
        raise ConfigError(f"Expected a mapping at '{path}', got {type(data).__name__}")
    field_map = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(field_map)
    if unknown:
        raise ConfigError(f"Unknown config key(s) at '{path}': {sorted(unknown)}")
    kwargs = {}
    for name, f in field_map.items():
        if name not in data:
            continue
        raw = data[name]
        if dataclasses.is_dataclass(f.type) and isinstance(raw, dict):
            kwargs[name] = _build_dataclass(f.type, raw, f"{path}.{name}")
        else:
            kwargs[name] = raw
    return cls(**kwargs)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def from_yaml(path) -> RunConfig:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return _build_dataclass(RunConfig, data, "<root>")


def to_json(cfg: RunConfig) -> str:
    return canonical_json(dataclasses.asdict(cfg))


def from_json(json_str: str) -> RunConfig:
    return _build_dataclass(RunConfig, json.loads(json_str), "<root>")


def run_id(cfg: RunConfig) -> str:
    payload = dataclasses.asdict(cfg)
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[:12]


def _field_type(cls, path_parts):
    for i, part in enumerate(path_parts):
        f = next((f for f in fields(cls) if f.name == part), None)
        if f is None:
            raise ConfigError(f"Unknown override key: {'.'.join(path_parts)}")
        if i == len(path_parts) - 1:
            return f.type
        cls = f.type
        if not dataclasses.is_dataclass(cls):
            raise ConfigError(f"Cannot descend into non-config field: {'.'.join(path_parts[: i + 1])}")
    raise ConfigError("Empty override key")


def _coerce(value_str: str, ftype):
    origin = typing.get_origin(ftype)
    if origin is typing.Union:
        args = [a for a in typing.get_args(ftype) if a is not type(None)]
        if len(args) == 1:
            if value_str.lower() in ("none", "null", "~"):
                return None
            return _coerce(value_str, args[0])
    if ftype is bool:
        return value_str.lower() in ("1", "true", "yes", "on")
    if ftype is int:
        return int(value_str)
    if ftype is float:
        return float(value_str)
    if ftype is str:
        return value_str
    # lists/tuples/anything else: parse as YAML scalar/sequence
    return yaml.safe_load(value_str)


def apply_overrides(cfg: RunConfig, overrides: List[str]) -> RunConfig:
    data = dataclasses.asdict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Malformed --set override (expected key=value): {item!r}")
        key, raw_value = item.split("=", 1)
        parts = key.split(".")
        ftype = _field_type(RunConfig, parts)
        value = _coerce(raw_value, ftype)
        node = data
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = value
    return _build_dataclass(RunConfig, data, "<root>")
