"""
Common fixed runtime dependencies shared across all Kaggle workers.
Regardless of who runs the kernel, these runtime wheels and CUDA bindings
are hosted by the fixed owner account.
"""
from __future__ import annotations

from typing import List

# Fixed runtime dependencies (wheels, CUDA 12 SM75 binaries for Dual Tesla T4)
COMMON_RUNTIME_DATASETS: List[str] = [
    "rohitvijayadapa/ornith-runtime-wheels",
    "rohitvijayadapa/ornith-runtime-cuda12-sm75",
]
