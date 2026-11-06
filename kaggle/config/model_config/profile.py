"""Data models defining LLM profiles, quantization, and vision/multimodal capabilities."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelProfile:
    """Specification of an LLM profile hosted on Kaggle GPU."""

    id: str
    display_name: str
    filename: str
    has_vision: bool = False
    context_size: int = 32768
    quantization: str = "Q4_K_M"
    gpu_target: str = "NVIDIA Tesla T4 x2"
    split_mode: str = "LAYER"
    use_mmap: bool = True
    
    # Kaggle source definitions (model/dataset slugs)
    model_source: str = ""
    dataset_source: str = ""
    fallback_sources: List[str] = field(default_factory=list)
    filename_patterns: List[str] = field(default_factory=list)
    
    # Multimodal / Vision projector config (for Qwen 3.6 / Qwen 3.8)
    mmproj_source: str = ""
    mmproj_filename: str = ""
    
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_multimodal(self) -> bool:
        """True if the model supports vision / image analysis."""
        return self.has_vision and bool(self.mmproj_source or self.mmproj_filename)
