"""
Phase 2: Deterministic Model Asset Resolution & Strict Projector Verification.

Resolves:
- Deterministic model selection: searches specifically for target profile filename & patterns (no random first-found matching).
- Strict vision projector compatibility: model-specific architecture check.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from kaggle.config.model_config.profile import ModelProfile
except ImportError:
    pass


def find_model_weights(profile: ModelProfile, search_root: Path | str = "/kaggle/input") -> Tuple[bool, Path | None, str]:
    """
    Deterministically search Kaggle input mounts for the SPECIFIC model profile filename.
    Never grabs an arbitrary first-found GGUF.
    """
    root = Path(search_root)
    if not root.exists():
        root = Path(".")

    # 1. Exact filename match for target profile
    for p in root.rglob(profile.filename):
        if p.is_file() and p.stat().st_size > 1 * 1024 * 1024 * 1024:
            size_gb = p.stat().st_size / (1024 ** 3)
            return True, p.resolve(), f"Resolved target model '{profile.id}' ({size_gb:.2f} GB) at {p.resolve()}"

    # 2. Strict pattern matching for target profile
    for pattern in profile.filename_patterns:
        for p in root.rglob(pattern):
            if p.is_file() and p.stat().st_size > 1 * 1024 * 1024 * 1024:
                size_gb = p.stat().st_size / (1024 ** 3)
                return True, p.resolve(), f"Matched target pattern '{pattern}' ({size_gb:.2f} GB) at {p.resolve()}"

    return False, None, f"Deterministic search failed: target model '{profile.id}' ({profile.filename}) not found in {root}."


def resolve_vision_projector(
    profile: ModelProfile,
    model_path: Path,
    search_root: Path | str = "/kaggle/input",
) -> Tuple[bool, Path | None, str]:
    """
    Strict vision projector admission check.
    Only enables vision if mmproj is explicitly verified for the target architecture.
    """
    if not profile.has_vision:
        return False, None, f"Model '{profile.id}' is configured for pure text/reasoning (has_vision=False)."

    root = Path(search_root)
    if not root.exists():
        root = Path(".")

    candidates: List[Path] = []
    
    # Check next to model file
    for p in model_path.parent.glob("*.gguf"):
        if p.is_file() and "mmproj" in p.name.lower():
            candidates.append(p.resolve())

    # Check search root
    for p in root.rglob("*.gguf"):
        if p.is_file() and "mmproj" in p.name.lower() and p.resolve() not in candidates:
            candidates.append(p.resolve())

    if not candidates:
        return False, None, "No mmproj vision projector file found. Dropping to text-only mode."

    m_stem = model_path.stem.lower()
    for cand in candidates:
        c_name = cand.name.lower()
        if "27b" in profile.id or "27b" in m_stem:
            if "27b" in c_name:
                return True, cand, f"Verified compatible vision projector: {cand.name}"
        elif "35b" in profile.id or "35b" in m_stem:
            if "35b" in c_name:
                return True, cand, f"Verified compatible vision projector: {cand.name}"

    return False, None, f"mmproj candidates found ({[c.name for c in candidates]}), but none match architecture for {profile.id}. Preserving text-only mode."
