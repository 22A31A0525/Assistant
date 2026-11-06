"""
Kaggle Kernel Metadata Builder.

Generates kernel-metadata.json with exact model sources, shared CUDA 12 wheels,
and dual Tesla T4 GPU configuration for the chosen model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from kaggle.config.auth_config.manager import load_accounts
from kaggle.config.model_config.catalog import ModelCatalog
from kaggle.config.model_config.profile import ModelProfile


def generate_kernel_metadata(
    model: ModelProfile,
    code_file: str = "worker.py",
    output_dir: Path | None = None,
) -> Path:
    """
    Generate kernel-metadata.json specifically tuned for the chosen model.
    Attaches shared runtime wheels + model weights + vision mmproj projector.
    """
    active, _ = load_accounts()
    username = active.username if active else "alluriakshay"
    slug = f"assistant-worker-{username.lower().replace('_', '-')}"
    kernel_id = f"{username}/{slug}"

    dataset_sources, model_sources = ModelCatalog.get_kernel_sources(model)

    metadata: Dict[str, Any] = {
        "id": kernel_id,
        "title": f"Assistant Worker ({username})",
        "code_file": code_file,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": "true",
        "docker_image_pinning_type": "latest",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": model_sources,
    }

    target_dir = output_dir or (Path(__file__).parent)
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "kernel-metadata.json"
    out_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_file
