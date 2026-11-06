"""
ModelProfile definition bundled for standalone execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ModelProfile:
    """Specification of an LLM profile hosted on Kaggle GPU."""

    id: str
    display_name: str
    filename: str
    has_vision: bool = False
    context_size: int = 65536
    quantization: str = "Q4_K_M"
    gpu_target: str = "NVIDIA Tesla T4 x2"
    split_mode: str = "LAYER"
    use_mmap: bool = True
    
    # Model architecture parameters for accurate KV cache calculation
    n_layers: int = 64
    n_kv_heads: int = 8
    head_dim: int = 128
    
    model_source: str = ""
    dataset_source: str = ""
    fallback_sources: List[str] = field(default_factory=list)
    filename_patterns: List[str] = field(default_factory=list)
    
    mmproj_source: str = ""
    mmproj_filename: str = ""
    extra_metadata: Dict[str, Any] = field(default_factory=dict)
    manifest: Optional[Any] = None
