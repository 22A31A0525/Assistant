"""
Deploy Notebook Subsystem Package.

Implements Subtask 1 Kaggle Dual-T4 Infrastructure:
- Phase 1: Environment bootstrap & CUDA verification (environment.py)
- Phase 2: Asset resolution & strict vision gating (model_resolver.py)
- Phase 3: Resource admission & VRAM headroom check (gpu_allocator.py)
- Phase 4: Native server execution with layer splitting (server_runner.py)
- Phase 5: Kernel metadata generation for Kaggle CLI push (metadata.py)
"""
from kaggle.deploy_notebook.environment import (
    check_gpu_topology,
    stage_runtime_and_libraries,
)
from kaggle.deploy_notebook.gpu_allocator import (
    check_resource_admission,
    estimate_kv_cache_mib,
)
from kaggle.deploy_notebook.metadata import generate_kernel_metadata
from kaggle.deploy_notebook.model_resolver import (
    find_model_weights,
    resolve_vision_projector,
)
from kaggle.deploy_notebook.server_runner import (
    build_native_server_command,
    verify_server_readiness,
)

__all__ = [
    "check_gpu_topology",
    "stage_runtime_and_libraries",
    "find_model_weights",
    "resolve_vision_projector",
    "estimate_kv_cache_mib",
    "check_resource_admission",
    "build_native_server_command",
    "verify_server_readiness",
    "generate_kernel_metadata",
]
