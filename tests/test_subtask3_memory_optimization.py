"""
Unit tests for Subtask 3: Hybrid 16/48 attention memory accounting and context thresholds.
"""
import pytest
from kaggle.deploy_notebook.gpu_allocator import (
    estimate_hybrid_memory_footprint,
    estimate_kv_cache_mib,
    check_resource_admission,
    MAX_ALLOC_TARGET_MIB,
    SAFETY_HEADROOM_MIB,
)
from kaggle.config.model_config.catalog import ModelCatalog


def test_hybrid_attention_memory_exact_calculation():
    # 65,536 tokens, 16 full-attention layers, 4 KV heads, 256 head_dim, q8_0 (1.0625 bytes/elem)
    # Expected: 2 * 16 * 4 * 256 * 65536 * 1.0625 / (1024 * 1024) = 2,176.0 MiB (~2.125 GiB)
    total_mib, token_kv_mib, state_mib, status = estimate_hybrid_memory_footprint(
        context_tokens=65536,
        n_full_attn_layers=16,
        n_deltanet_layers=48,
        n_kv_heads=4,
        head_dim=256,
        cache_type="q8_0",
        deltanet_recurrent_mib=None,  # Not empirically verified -> applies conservative allowance (24.0 MiB), NEVER 0.0
    )
    assert round(token_kv_mib, 1) == 2176.0
    assert state_mib > 0.0  # Must never be zero
    assert round(state_mib, 1) == 24.0
    assert status == "CONSERVATIVE_CALCULATED_ALLOWANCE"
    assert total_mib == token_kv_mib + state_mib


def test_hybrid_memory_with_configured_recurrent_state():
    # When empirical recurrent state is provided, it uses verified figure
    total_mib, token_kv_mib, state_mib, status = estimate_hybrid_memory_footprint(
        context_tokens=65536,
        n_full_attn_layers=16,
        n_deltanet_layers=48,
        n_kv_heads=4,
        head_dim=256,
        cache_type="q8_0",
        deltanet_recurrent_mib=512.0,
    )
    assert round(token_kv_mib, 1) == 2176.0
    assert state_mib == 512.0
    assert status == "VERIFIED_EMPIRICAL_STATE"
    assert total_mib == 2176.0 + 512.0


def test_resource_admission_per_gpu_headroom():
    model = ModelCatalog.get_by_id("qwen38-27b")
    assert model is not None

    admitted, eff_ctx, breakdown, msg = check_resource_admission(
        model_size_gb=16.24,
        profile=model,
        target_context=65536,
        num_gpus=2,
        has_vision_active=True,
    )
    assert admitted is True
    assert eff_ctx == 65536
    # Both GPUs must respect the per-card safety limit
    assert breakdown["est_gpu0_mib"] <= MAX_ALLOC_TARGET_MIB
    assert breakdown["est_gpu1_mib"] <= MAX_ALLOC_TARGET_MIB
    assert breakdown["headroom_per_gpu_mib"] == SAFETY_HEADROOM_MIB


def test_chat_context_warning_threshold():
    # Verify the 85% capacity threshold arithmetic
    max_ctx = 65536
    warn_thresh = int(max_ctx * 0.85)
    assert warn_thresh == 55705


def test_auth_gateway_hard_context_overflow_rejection():
    from kaggle.deploy_notebook.auth_gateway import AuthenticatedGatewayHandler
    import json

    handler = AuthenticatedGatewayHandler.__new__(AuthenticatedGatewayHandler)
    handler.path = "/v1/chat/completions"
    handler.max_context_tokens = 65536

    # Normal sized request (fits well within context window)
    normal_payload = json.dumps({
        "messages": [{"role": "user", "content": "Hello, write a simple function."}],
        "max_tokens": 1024,
    }).encode("utf-8")
    valid, err = handler.validate_context_limits(normal_payload)
    assert valid is True
    assert err is None

    # Oversized payload exceeding max context window (e.g. 260,000 characters ~ 68,000 tokens)
    huge_prompt = "x" * 260000
    huge_payload = json.dumps({
        "messages": [{"role": "user", "content": huge_prompt}],
        "max_tokens": 2048,
    }).encode("utf-8")
    valid, err = handler.validate_context_limits(huge_payload)
    assert valid is False
    assert err is not None
    assert "exceeds maximum context capacity" in err
