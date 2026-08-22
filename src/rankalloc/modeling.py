"""Build the LoRA-wrapped model, turn a rank allocation into a live PEFT config, and verify the
result against the live model rather than the config dict. BUILD_SPEC.md §4.5.

allocation.py stays ignorant of scaling mode on purpose (see NOTES.md, P2) -- this module is where a
rank_pattern becomes an alpha_pattern (the I2 scaling-invariance trap) and where both become an
actual peft.PeftModel whose real parameter counts we read back and assert against, never trusted from
the config alone.
"""
import math
import re
from typing import Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from rankalloc.allocation import Allocation, ModuleSpec

DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SCALING_MODES = ("constant_ratio", "rslora", "fixed_alpha")

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def load_base_model(model_name: str, dtype=torch.float16, device: Optional[str] = "cuda", gradient_checkpointing: bool = True):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    if device:
        model = model.to(device)
    model.config.use_cache = False
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return model


def discover_module_specs(model, target_modules=DEFAULT_TARGET_MODULES) -> List[ModuleSpec]:
    """Walk the live base model for every nn.Linear whose leaf name is a target module."""
    target_set = set(target_modules)
    specs = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in target_set or not hasattr(module, "in_features"):
            continue
        layer_match = _LAYER_IDX_RE.search(name)
        layer_idx = int(layer_match.group(1)) if layer_match else 0
        specs.append(ModuleSpec(name=name, d_in=module.in_features, d_out=module.out_features, layer_idx=layer_idx))
    if not specs:
        raise ValueError(f"No target modules found among {target_modules}")
    return specs


def compute_alpha_pattern(
    rank_pattern: Dict[str, int],
    scaling_mode: str,
    alpha_ratio: int,
    fixed_alpha: int,
    reference_rank: int = 16,
) -> Dict[str, float]:
    """The I2 scaling trap lives here. Each mode is a different exponent on the multiplier
    s(r) that LoRA applies to B @ A x:

        constant_ratio   s(r) = alpha_ratio                       (alpha = alpha_ratio * r)
        rslora           s(r) = alpha_ratio * sqrt(R/r)           (alpha = alpha_ratio * sqrt(R),
                                                                   PEFT then divides by sqrt(r))
        fixed_alpha      s(r) = alpha_ratio * R / r               (alpha = alpha_ratio * R)

    All three are anchored so that s(reference_rank) == alpha_ratio: at r = R every mode applies an
    identical multiplier, and they differ *only* in how they extrapolate away from R. Without that
    anchoring the modes would also differ by an overall constant, confounding "which r-exponent"
    with "how strong is the adapter overall".

    NOTE on the exponent choice. Holding alpha/r constant does NOT make the comparison
    rank-neutral under AdamW. With B initialised at zero, Adam drives |B_jk| towards ~lr*t
    independently of r, and ||A x|| grows as sqrt(r), so the adapter's contribution scales as
    s(r) * r^theta with theta in [1/2, 1] (1/2 if B's columns stay incoherent with A x, 1 if they
    align, which consistent gradients encourage). Rank-neutrality therefore needs
    s(r) ~ r^(-theta), i.e. an exponent in [-1, -1/2]: fixed_alpha sits at -1, rslora at -1/2, and
    constant_ratio at 0 -- outside the bracket on the wrong side, meaning higher-rank modules adapt
    *faster*, not equally. theta is an empirical property of the trajectory, not a derivable
    constant, so scaling.mode is deliberately left as a config choice and the ablation is what
    settles it. See README "Limitations" L2.
    """
    if scaling_mode == "constant_ratio":
        return {name: float(alpha_ratio * r) for name, r in rank_pattern.items()}
    if scaling_mode == "fixed_alpha":
        return {name: float(fixed_alpha) for name in rank_pattern}
    if scaling_mode == "rslora":
        # PEFT computes scaling = lora_alpha / sqrt(r) when use_rslora=True, so a *constant* alpha
        # is what produces the canonical alpha/sqrt(r) rule. Passing alpha_ratio * r here (as an
        # earlier revision did) yields s(r) = alpha_ratio * sqrt(r) -- scaling that *grows* with
        # rank, the opposite of what rsLoRA specifies.
        return {name: float(alpha_ratio) * math.sqrt(reference_rank) for name in rank_pattern}
    raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")


def wrap_with_lora(base_model, rank_pattern: Dict[str, int], alpha_pattern: Dict[str, float], scaling_mode: str, target_modules=DEFAULT_TARGET_MODULES):
    if scaling_mode not in SCALING_MODES:
        raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")
    fallback_r = min(rank_pattern.values())
    fallback_alpha = min(alpha_pattern.values())
    lora_cfg = LoraConfig(
        r=fallback_r,
        lora_alpha=fallback_alpha,
        target_modules=list(target_modules),
        rank_pattern=dict(rank_pattern),
        alpha_pattern=dict(alpha_pattern),
        use_rslora=(scaling_mode == "rslora"),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(base_model, lora_cfg)


def build_model(
    model_name: str,
    rank_pattern: Dict[str, int],
    scaling_mode: str,
    alpha_ratio: int,
    fixed_alpha: int,
    target_modules=DEFAULT_TARGET_MODULES,
    dtype=torch.float16,
    device: Optional[str] = "cuda",
    gradient_checkpointing: bool = True,
    reference_rank: int = 16,
):
    base = load_base_model(model_name, dtype=dtype, device=device, gradient_checkpointing=gradient_checkpointing)
    alpha_pattern = compute_alpha_pattern(rank_pattern, scaling_mode, alpha_ratio, fixed_alpha, reference_rank)
    model = wrap_with_lora(base, rank_pattern, alpha_pattern, scaling_mode, target_modules)
    return model, alpha_pattern


PEFT_NAME_PREFIX = "base_model.model."


def verify_live_model(model, rank_pattern: Dict[str, int], alpha_pattern: Dict[str, float], adapter_name: str = "default") -> dict:
    """Walk named_modules(), read back each adapter's actual r/alpha and recompute its parameter
    count from the live lora_A/lora_B tensor shapes -- never trust the config dict.

    get_peft_model() wraps the base model under `base_model.model.<original path>`; discover_module_
    specs() names modules by their original (pre-wrap) path, since that's what rank_pattern/
    alpha_pattern are keyed by (PEFT matches those during injection against the base model it's
    wrapping, not the final wrapped tree) -- strip that prefix back off before comparing.
    """
    found = {}
    total_params = 0
    for raw_name, module in model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None or adapter_name not in lora_a:
            continue
        name = raw_name[len(PEFT_NAME_PREFIX) :] if raw_name.startswith(PEFT_NAME_PREFIX) else raw_name
        r_live = module.r[adapter_name]
        alpha_live = float(module.lora_alpha[adapter_name])
        a_weight = module.lora_A[adapter_name].weight
        b_weight = module.lora_B[adapter_name].weight
        d_in = a_weight.shape[1]
        d_out = b_weight.shape[0]
        assert a_weight.shape[0] == r_live, f"{name}: lora_A rows {a_weight.shape[0]} != r {r_live}"
        assert b_weight.shape[1] == r_live, f"{name}: lora_B cols {b_weight.shape[1]} != r {r_live}"
        params = a_weight.numel() + b_weight.numel()
        assert params == r_live * (d_in + d_out), f"{name}: live params {params} != r*(d_in+d_out)"
        found[name] = {"r": r_live, "alpha": alpha_live, "params": params, "d_in": d_in, "d_out": d_out}
        total_params += params

    expected_names = set(rank_pattern)
    live_names = set(found)
    missing = expected_names - live_names
    extra = live_names - expected_names
    assert not missing, f"in allocation but not a live LoRA layer: {sorted(missing)}"
    assert not extra, f"live LoRA layer not in allocation: {sorted(extra)}"

    for name, expected_r in rank_pattern.items():
        assert found[name]["r"] == expected_r, f"{name}: live r={found[name]['r']} != expected {expected_r}"
    for name, expected_alpha in alpha_pattern.items():
        assert found[name]["alpha"] == expected_alpha, f"{name}: live alpha={found[name]['alpha']} != expected {expected_alpha}"

    return {"adapter_params_verified": True, "adapter_params_total": total_params, "per_module": found}


def alloc_json_payload(allocation: Allocation, alpha_pattern: Dict[str, float], scaling_mode: str) -> dict:
    payload = allocation.to_dict()
    payload["alpha_pattern"] = alpha_pattern
    payload["scaling_mode"] = scaling_mode
    return payload
