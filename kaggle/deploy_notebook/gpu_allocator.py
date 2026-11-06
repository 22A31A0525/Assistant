"""
Phase 3: Model-Specific Resource Admission & VRAM Headroom Check.

Calculates accurate KV cache using the specific model's n_layers, n_kv_heads, and head_dim.
Enforces strict 1-2 GB headroom per GPU. Fails startup if resource admission fails.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

try:
    from kaggle.config.model_config.profile import ModelProfile
except ImportError:
    pass

VRAM_PER_GPU_MIB = 16160.0       # ~15.78 GiB physical VRAM on Tesla T4
MAX_ALLOC_TARGET_MIB = 14336.0   # ~14.00 GiB (reserves ~1.8 GiB safety headroom per card)
SAFETY_HEADROOM_MIB = 1824.0     # ~1.8 GiB headroom


def estimate_kv_cache_mib(
    context_tokens: int,
    n_layers: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    cache_type: str = "q8_0",
) -> float:
    """
    Accurately estimate total KV cache memory footprint using model-specific architecture params.
    """
    bytes_per_elem = 1.0625 if cache_type == "q8_0" else 2.0
    # 2 buffers (K and V) * layers * heads * head_dim * context_tokens * bytes
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * context_tokens * bytes_per_elem
    return total_bytes / (1024 * 1024)


def check_resource_admission(
    model_size_gb: float,
    profile: ModelProfile,
    target_context: int = 65536,
    num_gpus: int = 2,
    has_vision_active: bool = False,
) -> Tuple[bool, int, Dict[str, float], str]:
    """
    Verify whether the model weights and target context window can be safely admitted into VRAM.
    Enforces per-GPU limits independently (assuming layer split across GPUs) to guarantee safety headroom.

    Returns:
        (admitted, effective_context, breakdown, message)
    """
    model_size_mib = model_size_gb * 1024.0
    n_layers = getattr(profile, "n_layers", 64)

    # Layer splitting distribution across GPUs
    layers_per_gpu = [math.ceil(n_layers / num_gpus) if i == 0 else math.floor(n_layers / num_gpus) for i in range(num_gpus)]

    # Model-specific KV cache estimation
    kv_cache_mib = estimate_kv_cache_mib(
        context_tokens=target_context,
        n_layers=n_layers,
        n_kv_heads=getattr(profile, "n_kv_heads", 8),
        head_dim=getattr(profile, "head_dim", 128),
        cache_type="q8_0",
    )
    vision_buffer_mib = 1024.0 if has_vision_active else 0.0
    scratch_compute_mib = 1024.0

    total_required_mib = model_size_mib + kv_cache_mib + vision_buffer_mib + scratch_compute_mib

    # Per-GPU allocation estimation (GPU 0 handles vision projector buffer & primary overhead)
    weight_fraction_gpu0 = layers_per_gpu[0] / n_layers
    weight_fraction_gpu1 = layers_per_gpu[1] / n_layers if num_gpus > 1 else 0.0

    est_gpu0_mib = (model_size_mib * weight_fraction_gpu0) + (kv_cache_mib * 0.5) + vision_buffer_mib + (scratch_compute_mib * 0.5)
    est_gpu1_mib = (model_size_mib * weight_fraction_gpu1) + (kv_cache_mib * 0.5) + (scratch_compute_mib * 0.5) if num_gpus > 1 else 0.0

    per_gpu_safe_limit = MAX_ALLOC_TARGET_MIB

    breakdown = {
        "model_weights_mib": round(model_size_mib, 1),
        "kv_cache_mib": round(kv_cache_mib, 1),
        "vision_buffer_mib": round(vision_buffer_mib, 1),
        "scratch_compute_mib": round(scratch_compute_mib, 1),
        "total_required_mib": round(total_required_mib, 1),
        "est_gpu0_mib": round(est_gpu0_mib, 1),
        "est_gpu1_mib": round(est_gpu1_mib, 1),
        "per_gpu_limit_mib": round(per_gpu_safe_limit, 1),
        "headroom_per_gpu_mib": round(SAFETY_HEADROOM_MIB, 1),
    }

    # Strict per-card admission check: neither GPU can exceed MAX_ALLOC_TARGET_MIB (~14.0 GB)
    gpu0_fits = est_gpu0_mib <= per_gpu_safe_limit
    gpu1_fits = (est_gpu1_mib <= per_gpu_safe_limit) if num_gpus > 1 else True

    if gpu0_fits and gpu1_fits:
        msg = (
            f"Admitted at full {target_context:,} context. "
            f"Estimated per-card peak: GPU 0: {est_gpu0_mib / 1024:.2f} GB, GPU 1: {est_gpu1_mib / 1024:.2f} GB "
            f"(Safe limit: {per_gpu_safe_limit / 1024:.2f} GB with {SAFETY_HEADROOM_MIB / 1024:.2f} GB safety margin each)."
        )
        return True, target_context, breakdown, msg

    # Calculate safe fallback context if requested
    max_kv_allowed = 2 * (per_gpu_safe_limit - max(est_gpu0_mib - (kv_cache_mib * 0.5), est_gpu1_mib - (kv_cache_mib * 0.5)))
    if max_kv_allowed > 512.0:
        bytes_per_token = kv_cache_mib / target_context
        safe_ctx = int(max_kv_allowed / bytes_per_token)
        safe_ctx = max(8192, (safe_ctx // 4096) * 4096)
        msg = (
            f"Target context {target_context:,} exceeds safe per-GPU VRAM limit ({SAFETY_HEADROOM_MIB:.0f} MB headroom per card). "
            f"Safe supported context ceiling: {safe_ctx:,} tokens."
        )
        return False, safe_ctx, breakdown, msg

    return False, 0, breakdown, f"Model size ({model_size_gb:.2f} GB) too large to maintain required per-GPU VRAM safety headroom."
