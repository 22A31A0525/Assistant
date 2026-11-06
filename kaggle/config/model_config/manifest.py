"""
Subtask 2: Model Compatibility Manifest & Capability Gating.

Defines granular capability states, dynamic metadata container,
and safe capability admission policies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class CapabilityState(str, Enum):
    """Granular verification state of individual model capabilities."""
    UNVERIFIED = "UNVERIFIED"
    VERIFIED_BASIC = "VERIFIED_BASIC"
    VERIFIED_FULL = "VERIFIED_FULL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class ModelCapabilities:
    """Independent capability tracker for a deployed model."""
    text: CapabilityState = CapabilityState.VERIFIED_BASIC
    vision_simple: CapabilityState = CapabilityState.UNVERIFIED
    vision_complex: CapabilityState = CapabilityState.UNVERIFIED
    tool_calling: CapabilityState = CapabilityState.UNVERIFIED

    def can_process_text(self) -> bool:
        return self.text in (CapabilityState.VERIFIED_BASIC, CapabilityState.VERIFIED_FULL)

    def can_process_vision(self, complex_prompt: bool = False) -> bool:
        if complex_prompt:
            return self.vision_complex in (CapabilityState.VERIFIED_BASIC, CapabilityState.VERIFIED_FULL)
        return self.vision_simple in (CapabilityState.VERIFIED_BASIC, CapabilityState.VERIFIED_FULL)

    def can_use_tools(self) -> bool:
        return self.tool_calling in (CapabilityState.VERIFIED_BASIC, CapabilityState.VERIFIED_FULL)


@dataclass
class ModelCompatibilityManifest:
    """
    Formal compatibility manifest for versioned GGUF weights.
    Avoids hardcoding assumptions by accepting dynamically parsed GGUF metadata.
    """
    model_id: str
    gguf_source_repo: str
    gguf_filename: str
    gguf_file_size_bytes: int = 0
    gguf_sha256: Optional[str] = None

    # Dynamically extracted from GGUF metadata
    architecture: Optional[str] = None          # e.g., 'qwen2' from GGUF header
    n_layers: Optional[int] = None              # e.g., block_count from GGUF header
    context_length: Optional[int] = None        # e.g., context_length from GGUF header
    chat_template_present: bool = True
    chat_template_flavor: str = "model_native_jinja"

    # Auxiliary tensor tracking (never blindly classified as harmless)
    auxiliary_layer_status: str = "UNVERIFIED_AUXILIARY_LAYER"
    auxiliary_tensors: List[str] = field(default_factory=list)
    mtp_policy: str = "disabled_initially"

    # Vision projector pairing
    projector_filename: Optional[str] = None
    projector_source_repo: Optional[str] = None

    # Granular capability states
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)

    def record_gguf_metadata(
        self,
        architecture: str,
        n_layers: int,
        context_length: int,
        auxiliary_tensors: Optional[List[str]] = None,
        file_size_bytes: int = 0,
    ) -> None:
        """Update manifest with live dynamically parsed GGUF metadata."""
        self.architecture = architecture
        self.n_layers = n_layers
        self.context_length = context_length
        if file_size_bytes > 0:
            self.gguf_file_size_bytes = file_size_bytes
        if auxiliary_tensors:
            self.auxiliary_tensors = list(auxiliary_tensors)
            if any("blk.64" in t for t in auxiliary_tensors):
                self.auxiliary_layer_status = "DETECTED_UNVERIFIED_AUXILIARY_LAYER"

    def authorize_capability(self, capability_name: str, state: CapabilityState) -> None:
        """Promote or demote a specific capability state."""
        if hasattr(self.capabilities, capability_name):
            setattr(self.capabilities, capability_name, state)

    def to_dict(self) -> Dict[str, Any]:
        """Convert manifest to serializable dictionary."""
        return {
            "model_id": self.model_id,
            "gguf_source_repo": self.gguf_source_repo,
            "gguf_filename": self.gguf_filename,
            "gguf_file_size_bytes": self.gguf_file_size_bytes,
            "architecture": self.architecture,
            "n_layers": self.n_layers,
            "context_length": self.context_length,
            "auxiliary_layer_status": self.auxiliary_layer_status,
            "auxiliary_tensors": self.auxiliary_tensors,
            "mtp_policy": self.mtp_policy,
            "projector_filename": self.projector_filename,
            "capabilities": {
                "text": self.capabilities.text.value,
                "vision_simple": self.capabilities.vision_simple.value,
                "vision_complex": self.capabilities.vision_complex.value,
                "tool_calling": self.capabilities.tool_calling.value,
            },
        }
