"""
Llama Server Configuration and CLI Flag Generator.

Produces optimized command-line flags and JSON settings for running `llama-server`
on Kaggle Dual Tesla T4 GPUs with FlashAttention, Layer Split, and mmproj vision projectors.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from kaggle.config.model_config.profile import ModelProfile


def build_llama_server_args(
    model_path: str | Path,
    profile: ModelProfile,
    port: int = 8080,
    host: str = "127.0.0.1",
    mmproj_path: Optional[str | Path] = None,
) -> List[str]:
    """
    Construct the optimized command-line invocation for llama-server.
    """
    ctx = int(os.environ.get("N_CTX", str(profile.context_size)))
    batch = int(os.environ.get("N_BATCH", "2048"))
    ubatch = int(os.environ.get("N_UBATCH", "512"))
    threads = int(os.environ.get("N_THREADS", "4"))

    args = [
        "llama-server",
        "--model", str(model_path),
        "--alias", profile.id,
        "--host", host,
        "--port", str(port),
        "-ngl", "99",               # Offload all layers to GPUs
        "--ctx-size", str(ctx),
        "--batch-size", str(batch),
        "--ubatch-size", str(ubatch),
        "--threads", str(threads),
        "--flash-attn",             # Enable FlashAttention for memory efficiency
    ]

    # Dual GPU layer split
    if profile.split_mode == "LAYER":
        args.extend(["--split-mode", "layer"])
        if os.environ.get("USE_EQUAL_TENSOR_SPLIT", "1") == "1":
            args.extend(["--tensor-split", "0.5,0.5"])

    # Vision projector if multimodal model
    if profile.has_vision and mmproj_path:
        args.extend(["--mmproj", str(mmproj_path)])

    return args


def generate_server_config_dict(
    model_path: str | Path,
    profile: ModelProfile,
    port: int = 8080,
    host: str = "127.0.0.1",
    mmproj_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Generate structured configuration dictionary for llama-server runner.
    """
    ctx = int(os.environ.get("N_CTX", str(profile.context_size)))

    cfg: Dict[str, Any] = {
        "model": str(model_path),
        "model_alias": profile.id,
        "host": host,
        "port": port,
        "n_gpu_layers": -1,
        "split_mode": 1 if profile.split_mode == "LAYER" else 2,
        "n_ctx": ctx,
        "n_batch": int(os.environ.get("N_BATCH", "2048")),
        "n_ubatch": int(os.environ.get("N_UBATCH", "512")),
        "flash_attn": True,
        "use_mmap": profile.use_mmap,
        "tensor_split": [0.5, 0.5] if os.environ.get("USE_EQUAL_TENSOR_SPLIT", "1") == "1" else None,
    }

    if profile.has_vision and mmproj_path:
        cfg["mmproj"] = str(mmproj_path)

    return cfg
