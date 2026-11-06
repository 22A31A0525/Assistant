"""Unit tests for ModelCompatibilityManifest and granular capability states."""
import pytest
from kaggle.config.model_config.manifest import (
    ModelCompatibilityManifest,
    ModelCapabilities,
    CapabilityState,
)
from kaggle.config.model_config.catalog import ModelCatalog


def test_manifest_initial_state():
    manifest = ModelCompatibilityManifest(
        model_id="qwen38-27b",
        gguf_source_repo="saitejavenkatavinay/ornith-qwen38-27b",
        gguf_filename="Qwen3.8-27B-Q4_K_M.gguf",
    )
    # Text is verified basic by default
    assert manifest.capabilities.can_process_text() is True
    # Vision simple and complex are unverified initially
    assert manifest.capabilities.can_process_vision(complex_prompt=False) is False
    assert manifest.capabilities.can_process_vision(complex_prompt=True) is False
    assert manifest.capabilities.can_use_tools() is False


def test_dynamic_gguf_metadata_recording():
    manifest = ModelCompatibilityManifest(
        model_id="qwen38-27b",
        gguf_source_repo="saitejavenkatavinay/ornith-qwen38-27b",
        gguf_filename="Qwen3.8-27B-Q4_K_M.gguf",
    )
    # Simulate recording dynamic binary GGUF header values
    manifest.record_gguf_metadata(
        architecture="qwen2",
        n_layers=65,
        context_length=65536,
        auxiliary_tensors=["blk.64.attn_q.weight", "blk.64.nextn.eh_proj.weight"],
        file_size_bytes=17438258380,
    )
    assert manifest.architecture == "qwen2"
    assert manifest.n_layers == 65
    assert manifest.auxiliary_layer_status == "DETECTED_UNVERIFIED_AUXILIARY_LAYER"
    assert len(manifest.auxiliary_tensors) == 2


def test_independent_capability_authorization():
    caps = ModelCapabilities()
    assert caps.can_process_vision(complex_prompt=False) is False

    # Authorize basic vision only
    caps.vision_simple = CapabilityState.VERIFIED_BASIC
    assert caps.can_process_vision(complex_prompt=False) is True
    # Complex vision must strictly remain unverified
    assert caps.can_process_vision(complex_prompt=True) is False
    # Tool calling must strictly remain unverified
    assert caps.can_use_tools() is False


def test_catalog_qwen38_manifest_attachment():
    model = ModelCatalog.get_by_id("qwen38-27b")
    assert model is not None
    assert model.manifest is not None
    assert model.manifest.model_id == "qwen38-27b"
    assert model.manifest.projector_filename == "mmproj-Qwen3.8-27B-f16.gguf"


def test_template_source_classification():
    from kaggle.config.model_config.manifest import TemplateSource
    manifest = ModelCompatibilityManifest(
        model_id="qwen38-27b",
        gguf_source_repo="saitejavenkatavinay/ornith-qwen38-27b",
        gguf_filename="Qwen3.8-27B-Q4_K_M.gguf",
    )
    assert manifest.chat_template_source == TemplateSource.UNVERIFIED

    # Update with detected embedded template
    manifest.record_gguf_metadata(
        architecture="qwen3",
        n_layers=65,
        context_length=65536,
        has_chat_template=True,
        template_source=TemplateSource.EMBEDDED_GGUF,
    )
    assert manifest.chat_template_present is True
    assert manifest.chat_template_source == TemplateSource.EMBEDDED_GGUF
