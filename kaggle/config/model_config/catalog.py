"""
Curated 3-Model Catalog with strict owner and user dataset resolution.

Model 1: Ornith-1.0-35B (Text / Reasoning / Zero Image capability)
  - Fixed model source: rohitvijayadapa/ornith-1-0-35b/gguf/q4-k-m/1
  - User dataset fallback: <active_user>/ornith-1-0-35b

Model 2: Qwen-3.6-35B (Multimodal with mmproj vision projector)
  - Fixed dataset source: saitejavenkatavinay/qwen36-35b-a3b-q4km
  - Fixed mmproj source: saitejavenkatavinay/qwen36-35b-mmproj

Model 3: Qwen-3.8-27B (Multimodal with mmproj vision projector)
  - Fixed model source: saitejavenkatavinay/ornith-qwen38-27b/gguf/q4-k-m-vision/1
  - Fixed dataset source: saitejavenkatavinay/ornith-qwen38-27b
  - Fixed mmproj source: saitejavenkatavinay/qwen36-35b-mmproj
"""
from __future__ import annotations

from typing import List, Optional
from kaggle.config.auth_config.manager import load_accounts
from kaggle.config.model_config.common import COMMON_RUNTIME_DATASETS
from kaggle.config.model_config.profile import ModelProfile


def get_active_username() -> str:
    """Retrieve active Kaggle username from .env without hardcoding."""
    active, _ = load_accounts()
    return active.username if active else ""


def get_ornith_35b() -> ModelProfile:
    """Ornith-1.0-35B: Pure text & reasoning (No vision)."""
    user = get_active_username()
    user_fallback = [f"{user}/ornith-1-0-35b"] if user else []

    return ModelProfile(
        id="ornith-35b",
        display_name="Ornith-1.0-35B (Text / Reasoning)",
        filename="Ornith-1.0-35B-Q4_K_M.gguf",
        has_vision=False,
        context_size=32768,
        quantization="Q4_K_M",
        gpu_target="NVIDIA Tesla T4 x2",
        split_mode="LAYER",
        # Fixed primary source on Kaggle
        model_source="rohitvijayadapa/ornith-1-0-35b/gguf/q4-k-m/1",
        dataset_source="rohitvijayadapa/ornith-1-0-35b",
        fallback_sources=user_fallback + [
            "rohitvijayadapa/ornith-1-0-35b",
            "rohitvijayadapa/ornith-1-0-35b-q4-k-m",
        ],
        filename_patterns=[
            "Ornith-1.0-35B-Q4_K_M.gguf",
            "ornith-1.0-35b-Q4_K_M.gguf",
            "*ornith*35b*.gguf",
        ],
    )


def get_qwen36_35b() -> ModelProfile:
    """Qwen-3.6-35B: Multimodal with vision projector."""
    user = get_active_username()
    user_fallback = [f"{user}/qwen36-35b-a3b-q4km"] if user else []

    return ModelProfile(
        id="qwen36-35b",
        display_name="Qwen-3.6-35B (Vision / Multimodal)",
        filename="qwen36-35b-a3b-q4km.gguf",
        has_vision=True,
        context_size=32768,
        quantization="Q4_K_M",
        gpu_target="NVIDIA Tesla T4 x2",
        split_mode="LAYER",
        # Fixed shared source
        dataset_source="saitejavenkatavinay/qwen36-35b-a3b-q4km",
        fallback_sources=user_fallback + [
            "saitejavenkatavinay/qwen36-35b-a3b-q4km",
        ],
        filename_patterns=[
            "*qwen*36*35b*q4_k_m*.gguf",
            "*Qwen3.6*35B*Q4_K_M*.gguf",
            "qwen36-35b-a3b-q4km.gguf",
        ],
        # Fixed shared vision projector
        mmproj_source="saitejavenkatavinay/qwen36-35b-mmproj",
        mmproj_filename="mmproj-qwen36-35b-f16.gguf",
    )


def get_qwen38_27b() -> ModelProfile:
    """Qwen-3.8-27B: Multimodal with vision projector."""
    user = get_active_username()
    user_fallback = [f"{user}/ornith-qwen38-27b"] if user else []

    return ModelProfile(
        id="qwen38-27b",
        display_name="Qwen-3.8-27B (Vision / Multimodal)",
        filename="Qwen3.8-27B-Q4_K_M.gguf",
        has_vision=True,
        context_size=32768,
        quantization="Q4_K_M",
        gpu_target="NVIDIA Tesla T4 x2",
        split_mode="LAYER",
        # Fixed shared model source & dataset
        model_source="saitejavenkatavinay/ornith-qwen38-27b/gguf/q4-k-m-vision/1",
        dataset_source="saitejavenkatavinay/ornith-qwen38-27b",
        fallback_sources=user_fallback + [
            "saitejavenkatavinay/ornith-qwen38-27b",
        ],
        filename_patterns=[
            "Qwen3.8-27B-Q4_K_M.gguf",
            "*qwen*3*8*27b*q4_k_m*.gguf",
            "*ornith*qwen*38*.gguf",
        ],
        # Fixed shared vision projector
        mmproj_source="saitejavenkatavinay/qwen36-35b-mmproj",
        mmproj_filename="mmproj-qwen38-27b-f16.gguf",
    )


class ModelCatalog:
    """Registry maintaining exactly our 3 target models with fixed/shared sources."""

    @classmethod
    def list_models(cls) -> List[ModelProfile]:
        return [
            get_ornith_35b(),
            get_qwen36_35b(),
            get_qwen38_27b(),
        ]

    @classmethod
    def get_by_id(cls, model_id: str) -> Optional[ModelProfile]:
        norm = model_id.strip().lower().replace("_", "-").replace(".", "")
        for m in cls.list_models():
            m_norm = m.id.lower().replace("_", "-").replace(".", "")
            if norm == m_norm or norm in m_norm:
                return m
        return None

    @classmethod
    def get_kernel_sources(cls, model: ModelProfile) -> tuple[list[str], list[str]]:
        """
        Generate exact (dataset_sources, model_sources) list for kernel-metadata.json.
        Merges common runtime wheels with model-specific fixed paths.
        """
        datasets = list(COMMON_RUNTIME_DATASETS)
        models = []

        if model.dataset_source and model.dataset_source not in datasets:
            datasets.append(model.dataset_source)
        if model.mmproj_source and model.mmproj_source not in datasets:
            datasets.append(model.mmproj_source)
        if model.model_source:
            models.append(model.model_source)

        return datasets, models
