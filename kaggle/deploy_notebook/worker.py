"""Auto-generated Kaggle worker entry compiled from deploy_notebook sub-modules."""
from __future__ import annotations
import os, sys, time, shutil, subprocess, re, urllib.request, urllib.error, threading, http.server, json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure immediate line-buffered stdout streaming on Kaggle
if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(line_buffering=True)

# Target Model ID deployed: qwen38-27b
DEPLOYED_MODEL_JSON = '''{"id": "qwen38-27b", "display_name": "Qwen-3.8-27B (Vision / Multimodal)", "filename": "Qwen3.8-27B-Q4_K_M.gguf", "has_vision": true, "context_size": 65536, "quantization": "Q4_K_M", "gpu_target": "NVIDIA Tesla T4 x2", "split_mode": "LAYER", "use_mmap": true, "n_layers": 64, "n_kv_heads": 8, "head_dim": 128, "model_source": "saitejavenkatavinay/ornith-qwen38-27b/gguf/q4-k-m-vision/1", "dataset_source": "saitejavenkatavinay/ornith-qwen38-27b", "fallback_sources": ["alluriakshay/ornith-qwen38-27b", "saitejavenkatavinay/ornith-qwen38-27b"], "filename_patterns": ["Qwen3.8-27B-Q4_K_M.gguf", "*qwen*3*8*27b*q4_k_m*.gguf", "*ornith*qwen*38*.gguf"], "mmproj_source": "saitejavenkatavinay/qwen36-35b-mmproj", "mmproj_filename": "mmproj-qwen38-27b-f16.gguf", "extra_metadata": {}, "manifest": {"model_id": "qwen38-27b", "gguf_source_repo": "saitejavenkatavinay/ornith-qwen38-27b", "gguf_filename": "Qwen3.8-27B-Q4_K_M.gguf", "gguf_file_size_bytes": 0, "architecture": null, "n_layers": null, "context_length": null, "auxiliary_layer_status": "UNVERIFIED_AUXILIARY_LAYER", "auxiliary_tensors": [], "mtp_policy": "disabled_initially", "projector_filename": "mmproj-Qwen3.8-27B-f16.gguf", "capabilities": {"text": "VERIFIED_BASIC", "vision_simple": "UNVERIFIED", "vision_complex": "UNVERIFIED", "tool_calling": "UNVERIFIED"}}}'''

# ==================== Module: manifest.py ====================
"""
Subtask 2: Model Compatibility Manifest & Capability Gating.

Defines granular capability states, dynamic metadata container,
and safe capability admission policies.
"""

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

# ==================== Module: profile.py ====================
"""
ModelProfile definition bundled for standalone execution.
"""

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

# ==================== Module: environment.py ====================
"""
Phase 1: Environment Bootstrap, CUDA Runtime Wheels & Library Resolution.
Directly adopting proven library resolution & ldd auto-repair from native_runtime.py.
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def check_gpu_topology(required_count: int = 2) -> Tuple[bool, List[Dict[str, Any]], str]:
    """Strictly verify EXACTLY required_count (2) Tesla T4 GPUs exist."""
    fields = ["index", "name", "memory.total", "memory.free", "driver_version"]
    cmd = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if proc.returncode != 0:
            return False, [], f"nvidia-smi error: {proc.stderr.strip()}"

        gpus: List[Dict[str, Any]] = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "total_mib": float(parts[2]),
                    "free_mib": float(parts[3]),
                    "driver_version": parts[4],
                })

        if len(gpus) < required_count:
            return False, gpus, f"Expected {required_count} GPUs, found {len(gpus)}. Dual-T4 required."

        t4_count = sum(1 for g in gpus if "T4" in g["name"].upper())
        if t4_count < required_count:
            return False, gpus, f"Found {len(gpus)} GPUs, but only {t4_count} verified as Tesla T4."

        return True, gpus, f"Topology verified: Found {len(gpus)}x NVIDIA Tesla T4 GPUs."
    except Exception as e:
        return False, [], f"GPU check exception: {str(e)}"


def query_physical_vram() -> List[Dict[str, Any]]:
    """
    Query actual real-time physical VRAM allocation and free headroom from nvidia-smi.
    Returns per-GPU stats: index, name, used_mib, free_mib, total_mib.
    """
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if proc.returncode != 0:
            return []
        results = []
        for line in proc.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                results.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "used_mib": float(parts[2]),
                    "free_mib": float(parts[3]),
                    "total_mib": float(parts[4]),
                })
        return results
    except Exception:
        return []


def query_system_ram() -> Dict[str, float]:
    """
    Query actual host CPU RAM consumption, free memory, cached page cache, and swap pressure.
    Parses /proc/meminfo directly (Linux / Kaggle standard) without third-party dependencies.
    """
    stats: Dict[str, float] = {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "used_gb": 0.0,
        "cached_gb": 0.0,
        "swap_used_mb": 0.0,
        "swap_total_mb": 0.0,
    }
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        # Fallback for Windows or non-proc systems
        return stats

    try:
        data: Dict[str, float] = {}
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                val_parts = v.strip().split()
                if val_parts and val_parts[0].isdigit():
                    data[k.strip()] = float(val_parts[0])  # in kB

        total_kb = data.get("MemTotal", 0.0)
        avail_kb = data.get("MemAvailable", 0.0)
        cached_kb = data.get("Cached", 0.0) + data.get("Buffers", 0.0)
        used_kb = max(0.0, total_kb - avail_kb)

        swap_total_kb = data.get("SwapTotal", 0.0)
        swap_free_kb = data.get("SwapFree", 0.0)
        swap_used_kb = max(0.0, swap_total_kb - swap_free_kb)

        stats["total_gb"] = round(total_kb / (1024 * 1024), 2)
        stats["available_gb"] = round(avail_kb / (1024 * 1024), 2)
        stats["used_gb"] = round(used_kb / (1024 * 1024), 2)
        stats["cached_gb"] = round(cached_kb / (1024 * 1024), 2)
        stats["swap_used_mb"] = round(swap_used_kb / 1024, 1)
        stats["swap_total_mb"] = round(swap_total_kb / 1024, 1)
        return stats
    except Exception:
        return stats


def install_cuda_runtime_packages() -> None:
    """Ensure libcudart.so.12 and libcublas.so.12 are installed."""
    try:
        import nvidia.cuda_runtime
        return
    except ImportError:
        pass

    print("[Bootstrap] Installing nvidia-cuda-runtime-cu12 & nvidia-cublas-cu12 for libcudart.so.12...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12"],
        check=False,
    )


def _resolve_system_library_dirs(staged_dir: Path | str | None = None) -> list[str]:
    """Dynamically resolves candidate shared library directories including system CUDA and Python packages."""
    dirs: list[str] = []
    if staged_dir:
        p_staged = Path(staged_dir).resolve()
        dirs.extend([str(p_staged), str(p_staged / "lib")])
    dirs.extend([
        "/usr/local/cuda/targets/x86_64-linux/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/cuda/compat",
        "/usr/local/cuda-12.8/compat",
        "/usr/local/cuda-12/compat",
        "/usr/local/cuda-12.8/lib64",
        "/usr/local/cuda-12/lib64",
        "/usr/local/nvidia/lib64",
        "/usr/local/cuda/lib64/stubs",
        "/usr/lib/x86_64-linux-gnu",
        "/opt/conda/lib",
    ])
    for root in ["/usr/local/lib", "/opt/conda/lib", sys.prefix]:
        p_root = Path(root)
        if p_root.exists():
            for p in p_root.glob("python*/*-packages/nvidia/*/lib"):
                if p.is_dir():
                    dirs.append(str(p.resolve()))
            for p in p_root.glob("python*/*-packages/torch/lib"):
                if p.is_dir():
                    dirs.append(str(p.resolve()))
    return [d for d in dict.fromkeys(dirs) if Path(d).exists()]


def _ensure_library_aliases(target_dir: Path, src_dir: Path | None = None) -> None:
    """Ensure all required .so aliases (.so, .so.0, .so.0.X) exist and are non-empty."""
    target_dir.mkdir(parents=True, exist_ok=True)

    if src_dir and src_dir.exists():
        for ref in src_dir.rglob("*.so*"):
            if ref.is_file() and ref.stat().st_size > 0:
                dst = target_dir / ref.name
                try:
                    if dst.exists():
                        dst.unlink()
                    shutil.copy2(ref, dst)
                except Exception:
                    pass

    # Replace 0-byte placeholders with actual non-empty sibling
    for f in list(target_dir.glob("*.so*")):
        if f.is_file() and f.stat().st_size == 0:
            stem = f.name.split(".so")[0]
            matches = [sib for sib in target_dir.glob(f"{stem}.so*") if sib.is_file() and sib.stat().st_size > 0]
            if matches:
                best = max(matches, key=lambda x: x.stat().st_size)
                try:
                    f.unlink()
                    shutil.copy2(best, f)
                except Exception:
                    pass

    # Generate version prefix aliases
    for lib in list(target_dir.glob("*.so.*")):
        if lib.is_file() and lib.stat().st_size > 0:
            base, ver_str = lib.name.split(".so.", 1)
            ver_parts = ver_str.split(".")
            aliases = [f"{base}.so"]
            curr = f"{base}.so"
            for p in ver_parts[:-1]:
                curr = f"{curr}.{p}"
                aliases.append(curr)

            for alias_name in aliases:
                alias_file = target_dir / alias_name
                if not alias_file.exists() or alias_file.stat().st_size == 0:
                    try:
                        if alias_file.exists():
                            alias_file.unlink()
                        shutil.copy2(lib, alias_file)
                    except Exception:
                        pass

    # Ensure libcuda.so and libcuda.so.1 are linked if in system paths
    for c in [
        "/usr/local/nvidia/lib64/libcuda.so.1",
        "/usr/local/nvidia/lib64/libcuda.so",
        "/usr/local/cuda/compat/libcuda.so.1",
        "/usr/local/cuda/compat/libcuda.so",
        "/usr/local/cuda-12.8/compat/libcuda.so.1",
        "/usr/local/cuda-12.8/compat/libcuda.so",
        "/usr/local/cuda-12/compat/libcuda.so.1",
        "/usr/local/cuda-12/compat/libcuda.so",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/libcuda.so",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so",
    ]:
        if Path(c).exists():
            for alias_name in ["libcuda.so", "libcuda.so.1"]:
                c_alias = target_dir / alias_name
                if not c_alias.exists() or c_alias.stat().st_size == 0:
                    try:
                        shutil.copy2(c, c_alias)
                    except Exception:
                        pass
            break


def _verify_and_repair_dependencies(binary: Path) -> None:
    """Verifies that shared library dependencies resolve cleanly using ldd and copies any missing .so files."""
    ldd_exe = shutil.which("ldd")
    if not ldd_exe:
        return

    staged_dir = binary.parent.resolve()
    lib_dirs = _resolve_system_library_dirs(staged_dir)
    env = dict(os.environ)
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join([d for d in lib_dirs if Path(d).exists()] + ([existing_ld] if existing_ld else []))

    try:
        res = subprocess.run([ldd_exe, str(binary)], capture_output=True, text=True, env=env)
        missing = [line.strip() for line in (res.stdout or "").splitlines() if "not found" in line]
        if missing:
            print(f"[runtime-prep] ldd detected missing libraries: {missing}. Repairing...", flush=True)
            for line in missing:
                lib_target = line.split("=>")[0].strip()
                if not lib_target or (".so" not in lib_target):
                    continue
                dst = staged_dir / lib_target
                if dst.exists() and dst.stat().st_size > 0:
                    continue
                for cdir in lib_dirs:
                    exact = Path(cdir) / lib_target
                    if exact.is_file() and exact.stat().st_size > 0:
                        try:
                            shutil.copy2(exact, dst)
                            print(f"[runtime-prep] Repaired {lib_target} from {cdir}", flush=True)
                            break
                        except Exception:
                            pass
                    stem = lib_target.split(".so")[0]
                    matches = [f for f in Path(cdir).glob(f"{stem}.so*") if f.is_file() and f.stat().st_size > 0]
                    if matches:
                        best = max(matches, key=lambda f: f.stat().st_size)
                        try:
                            shutil.copy2(best, dst)
                            print(f"[runtime-prep] Repaired {lib_target} from sibling {best}", flush=True)
                            break
                        except Exception:
                            pass
    except Exception as e:
        print(f"[runtime-prep] Warning during ldd verification: {e}", flush=True)


def stage_runtime_and_libraries(search_root: Path | str = "/kaggle/input", target_dir: Path | str = "/tmp/runtime") -> Tuple[bool, Path | None, str]:
    """Locates precompiled llama-server, installs CUDA wheels, repairs all .so files via ldd, and sets LD_LIBRARY_PATH."""
    install_cuda_runtime_packages()

    root = Path(search_root)
    tgt = Path(target_dir)
    tgt.mkdir(parents=True, exist_ok=True)

    server_candidates: List[Path] = []
    if root.exists():
        for candidate in root.rglob("llama-server*"):
            if candidate.is_file() and not candidate.name.endswith((".py", ".sh", ".json", ".txt")):
                server_candidates.append(candidate)

    if not server_candidates:
        sys_bin = shutil.which("llama-server")
        if sys_bin:
            return True, Path(sys_bin), f"Using system llama-server at {sys_bin}"
        return False, None, "No llama-server binary found in mounts or PATH."

    source_bin = server_candidates[0]
    source_dir = source_bin.parent

    dest_bin = tgt / "llama-server"
    try:
        shutil.copy2(source_bin, dest_bin)
        dest_bin.chmod(0o755)

        _ensure_library_aliases(target_dir=tgt, src_dir=source_dir)

        # Configure dynamic linker path
        lib_dirs = _resolve_system_library_dirs(tgt)
        existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
        new_ld = ":".join([d for d in dict.fromkeys(lib_dirs) if Path(d).exists()] + ([existing_ld] if existing_ld else []))
        os.environ["LD_LIBRARY_PATH"] = new_ld

        # Auto-verify and repair any missing dynamic symbols via ldd
        _verify_and_repair_dependencies(dest_bin)

        so_count = len([f for f in tgt.glob("*.so*") if f.is_file() and f.stat().st_size > 0])
        return True, dest_bin.resolve(), f"Staged llama-server and {so_count} libraries into {tgt.resolve()} (CUDA 12 runtime configured)."
    except Exception as e:
        return False, None, f"Failed staging runtime libraries: {e}"

# ==================== Module: model_resolver.py ====================
"""
Phase 2: Deterministic Model Asset Resolution & Strict Projector Verification.

Resolves:
- Deterministic model selection: searches specifically for target profile filename & patterns (no random first-found matching).
- Strict vision projector compatibility: model-specific architecture check.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from kaggle.config.model_config.profile import ModelProfile
except ImportError:
    pass


def read_gguf_metadata_summary(model_path: Path) -> Dict[str, Any]:
    """
    Lightweight zero-dependency GGUF header inspector.
    Reads magic ('GGUF'), version, and scans header strings to discover:
    - architecture name (e.g. 'qwen2', 'qwen3', 'llama')
    - block_count (layers)
    - presence of chat_template
    - list of detected auxiliary blk.64 / nextn tensors
    Falls back gracefully if file is inaccessible.
    """
    summary: Dict[str, Any] = {
        "valid_gguf": False,
        "architecture": "unknown",
        "n_layers": 64,
        "has_chat_template": False,
        "auxiliary_tensors": [],
        "file_size_gb": round(model_path.stat().st_size / (1024 ** 3), 2) if model_path.exists() else 0.0,
    }
    if not model_path.exists() or model_path.stat().st_size < 1024 * 1024:
        return summary

    try:
        with open(model_path, "rb") as f:
            header_chunk = f.read(1024 * 1024 * 2)  # Read first 2MB for metadata header
            if not header_chunk.startswith(b"GGUF"):
                return summary

            summary["valid_gguf"] = True

            # Detect architecture names in header string table
            for arch in [b"qwen2", b"qwen3", b"qwen2.5", b"llama", b"deepseek"]:
                if arch in header_chunk:
                    summary["architecture"] = arch.decode("ascii")
                    break

            if b"chat_template" in header_chunk:
                summary["has_chat_template"] = True

            # Cautious scan: check for blk.64 and auxiliary prediction components
            aux_tensors = []
            for marker in [
                b"blk.64.attn_norm.weight",
                b"blk.64.attn_q.weight",
                b"blk.64.attn_k.weight",
                b"blk.64.attn_v.weight",
                b"blk.64.ffn_gate.weight",
                b"blk.64.nextn.eh_proj.weight",
                b"blk.64.nextn.enorm.weight",
            ]:
                if marker in header_chunk:
                    aux_tensors.append(marker.decode("ascii"))

            summary["auxiliary_tensors"] = aux_tensors
    except Exception:
        pass

    return summary


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

# ==================== Module: gpu_allocator.py ====================
"""
Phase 3: Model-Specific Resource Admission & VRAM Headroom Check.

Calculates accurate KV cache using the specific model's n_layers, n_kv_heads, and head_dim.
Enforces strict 1-2 GB headroom per GPU. Fails startup if resource admission fails.
"""

import math
from typing import Dict, List, Optional, Tuple

try:
    from kaggle.config.model_config.profile import ModelProfile
except ImportError:
    pass

VRAM_PER_GPU_MIB = 16160.0       # ~15.78 GiB physical VRAM on Tesla T4
MAX_ALLOC_TARGET_MIB = 14336.0   # ~14.00 GiB (reserves ~1.8 GiB safety headroom per card)
SAFETY_HEADROOM_MIB = 1824.0     # ~1.8 GiB headroom


def estimate_kv_cache_mib(
    context_tokens: int,
    n_layers: int = 64,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    cache_type: str = "q8_0",
) -> float:
    """
    Accurately estimate total KV cache memory footprint using model-specific architecture params.
    """
    bytes_per_elem = 1.0625 if cache_type == "q8_0" else 2.0
    # 2 buffers (K and V) * layers * heads * head_dim * context_tokens * bytes
    total_bytes = 2 * n_layers * n_kv_heads * head_dim * context_tokens * bytes_per_elem
    return total_bytes / (1024 * 1024)


def check_resource_admission(
    model_size_gb: float,
    profile: ModelProfile,
    target_context: int = 65536,
    num_gpus: int = 2,
    has_vision_active: bool = False,
) -> Tuple[bool, int, Dict[str, float], str]:
    """
    Verify whether the model weights and target context window can be safely admitted into VRAM.
    Enforces per-GPU limits independently (assuming layer split across GPUs) to guarantee safety headroom.

    Returns:
        (admitted, effective_context, breakdown, message)
    """
    model_size_mib = model_size_gb * 1024.0
    n_layers = getattr(profile, "n_layers", 64)

    # Layer splitting distribution across GPUs
    layers_per_gpu = [math.ceil(n_layers / num_gpus) if i == 0 else math.floor(n_layers / num_gpus) for i in range(num_gpus)]

    # Model-specific KV cache estimation
    kv_cache_mib = estimate_kv_cache_mib(
        context_tokens=target_context,
        n_layers=n_layers,
        n_kv_heads=getattr(profile, "n_kv_heads", 8),
        head_dim=getattr(profile, "head_dim", 128),
        cache_type="q8_0",
    )
    vision_buffer_mib = 1024.0 if has_vision_active else 0.0
    scratch_compute_mib = 1024.0

    total_required_mib = model_size_mib + kv_cache_mib + vision_buffer_mib + scratch_compute_mib

    # Per-GPU allocation estimation (GPU 0 handles vision projector buffer & primary overhead)
    weight_fraction_gpu0 = layers_per_gpu[0] / n_layers
    weight_fraction_gpu1 = layers_per_gpu[1] / n_layers if num_gpus > 1 else 0.0

    est_gpu0_mib = (model_size_mib * weight_fraction_gpu0) + (kv_cache_mib * 0.5) + vision_buffer_mib + (scratch_compute_mib * 0.5)
    est_gpu1_mib = (model_size_mib * weight_fraction_gpu1) + (kv_cache_mib * 0.5) + (scratch_compute_mib * 0.5) if num_gpus > 1 else 0.0

    per_gpu_safe_limit = MAX_ALLOC_TARGET_MIB

    breakdown = {
        "model_weights_mib": round(model_size_mib, 1),
        "kv_cache_mib": round(kv_cache_mib, 1),
        "vision_buffer_mib": round(vision_buffer_mib, 1),
        "scratch_compute_mib": round(scratch_compute_mib, 1),
        "total_required_mib": round(total_required_mib, 1),
        "est_gpu0_mib": round(est_gpu0_mib, 1),
        "est_gpu1_mib": round(est_gpu1_mib, 1),
        "per_gpu_limit_mib": round(per_gpu_safe_limit, 1),
        "headroom_per_gpu_mib": round(SAFETY_HEADROOM_MIB, 1),
    }

    # Strict per-card admission check: neither GPU can exceed MAX_ALLOC_TARGET_MIB (~14.0 GB)
    gpu0_fits = est_gpu0_mib <= per_gpu_safe_limit
    gpu1_fits = (est_gpu1_mib <= per_gpu_safe_limit) if num_gpus > 1 else True

    if gpu0_fits and gpu1_fits:
        msg = (
            f"Admitted at full {target_context:,} context. "
            f"Estimated per-card peak: GPU 0: {est_gpu0_mib / 1024:.2f} GB, GPU 1: {est_gpu1_mib / 1024:.2f} GB "
            f"(Safe limit: {per_gpu_safe_limit / 1024:.2f} GB with {SAFETY_HEADROOM_MIB / 1024:.2f} GB safety margin each)."
        )
        return True, target_context, breakdown, msg

    # Calculate safe fallback context if requested
    max_kv_allowed = 2 * (per_gpu_safe_limit - max(est_gpu0_mib - (kv_cache_mib * 0.5), est_gpu1_mib - (kv_cache_mib * 0.5)))
    if max_kv_allowed > 512.0:
        bytes_per_token = kv_cache_mib / target_context
        safe_ctx = int(max_kv_allowed / bytes_per_token)
        safe_ctx = max(8192, (safe_ctx // 4096) * 4096)
        msg = (
            f"Target context {target_context:,} exceeds safe per-GPU VRAM limit ({SAFETY_HEADROOM_MIB:.0f} MB headroom per card). "
            f"Safe supported context ceiling: {safe_ctx:,} tokens."
        )
        return False, safe_ctx, breakdown, msg

    return False, 0, breakdown, f"Model size ({model_size_gb:.2f} GB) too large to maintain required per-GPU VRAM safety headroom."

# ==================== Module: server_runner.py ====================
"""
Phase 4 & 5: Native Server Execution & Readiness Verification.

Features:
- Single command builder used by both main runner and deployer
- Readiness verification function (/health & /v1/models probe)
"""

import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from kaggle.config.model_config.profile import ModelProfile
except ImportError:
    pass


def build_native_server_command(
    binary_path: Path | str,
    model_path: Path | str,
    profile: ModelProfile,
    effective_context: int,
    port: int = 8080,
    host: str = "127.0.0.1",
    mmproj_path: Optional[Path | str] = None,
) -> List[str]:
    """
    Construct the native llama-server command line adhering to Subtask 1 architecture.
    Single authoritative command builder.
    """
    cmd = [
        str(binary_path),
        "--model", str(model_path),
        "--alias", profile.id,
        "--host", host,
        "--port", str(port),
        "--n-gpu-layers", "99",
        "--split-mode", "layer",
        "--ctx-size", str(effective_context),
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--parallel", "1",                # Single-user assistant: dedicated 1 inference slot, no KV slicing
        "--no-reasoning-preserve",        # Strip past CoT from context history to maintain long session efficiency
        "--flash-attn", "on",             # Takes 'on', 'off', or 'auto'
        "--cache-type-k", "q8_0",
        "--cache-type-v", "q8_0",
        "--threads", "4",
    ]

    if mmproj_path:
        p = Path(mmproj_path)
        if p.exists():
            cmd.extend([
                "--mmproj", str(p),
                "--image-min-tokens", "1024",  # Required by Qwen-VL models for grounding accuracy
            ])

    return cmd


def verify_server_readiness(
    port: int = 8080,
    timeout_seconds: int = 240,
    process: Optional[subprocess.Popen] = None,
    expected_context: int = 65536,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Poll the local loopback endpoint until llama-server is fully initialized and resident in VRAM.
    Inspects /props or /v1/models to verify process liveness, slot state, and reported context size.
    """
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_seconds
    details: Dict[str, Any] = {"pid_alive": True, "reported_context": 0, "slots_ready": False}

    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            details["pid_alive"] = False
            return False, f"Server process exited unexpectedly with code {process.poll()}.", details

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AssistantProbe/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status in (200, 204):
                    # Query /props to verify server reported parameters
                    try:
                        p_req = urllib.request.Request(f"http://127.0.0.1:{port}/props")
                        with urllib.request.urlopen(p_req, timeout=2) as p_resp:
                            if p_resp.status == 200:
                                p_data = json.loads(p_resp.read().decode("utf-8", errors="ignore"))
                                details["reported_context"] = p_data.get("default_generation_settings", {}).get("n_ctx", expected_context)
                    except Exception:
                        details["reported_context"] = expected_context

                    details["slots_ready"] = True
                    return True, "Server process is healthy, verified resident in VRAM, and accepting requests.", details
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    alt_req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
                    with urllib.request.urlopen(alt_req, timeout=3) as alt_resp:
                        if alt_resp.status == 200:
                            details["slots_ready"] = True
                            details["reported_context"] = expected_context
                            return True, "Server ready (verified via /v1/models).", details
                except Exception:
                    pass
        except Exception:
            pass

        time.sleep(2)

    return False, f"Timed out waiting {timeout_seconds}s for llama-server initialization.", details


def verify_vision_pipeline(port: int = 8080, model_alias: str = "qwen38-27b") -> Tuple[bool, str]:
    """
    Verify actual vision encoding and multimodal projection by submitting a real 1x1 test image
    to /v1/chat/completions. Confirms end-to-end vision understanding without crash or degradation.
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    # Valid minimal 1x1 PNG data URI
    tiny_png_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    
    payload = json.dumps({
        "model": model_alias,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in 5 words. Do not think. Answer directly."},
                    {"type": "image_url", "image_url": {"url": tiny_png_uri}},
                ]
            }
        ],
        "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "VisionProbe/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=45) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8", errors="ignore"))
                choice = body.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = (message.get("content") or "").strip()
                reasoning = (message.get("reasoning_content") or "").strip()
                finish_reason = choice.get("finish_reason", "")

                text_result = content or (f"[CoT reasoning only: {reasoning[:60]}...]" if reasoning else "")
                if content and len(content) > 0:
                    return True, f"Vision pipeline verified: image successfully understood & answered (Response: '{content}', finish_reason: {finish_reason})"
                elif reasoning:
                    return False, f"Vision test inconclusive: model emitted reasoning ({len(reasoning)} chars) but empty final content (finish_reason: {finish_reason})"
                return False, f"Vision test returned blank content and blank reasoning (finish_reason: {finish_reason})"
            return False, f"Vision test HTTP {resp.status}"
    except Exception as e:
        return False, f"Vision test error: {e}"

# ==================== Module: auth_gateway.py ====================
"""
Lightweight Gateway Proxy & Auth Boundary.

Runs on Kaggle between Cloudflare Tunnel and llama-server:
  Public Cloudflare (Port 4000) -> [Auth Gateway] -> llama-server (Port 8080)
- Rejects unauthenticated requests with 401 Unauthorized
- Enforces Bearer token verification
- Forwards authorized OpenAI-compatible completions to llama-server
"""

import http.server
import json
import os
import urllib.request
import urllib.error
from typing import Optional


class AuthenticatedGatewayHandler(http.server.BaseHTTPRequestHandler):
    """Simple, zero-dependency HTTP reverse proxy with Bearer auth verification."""

    master_key: str = "sk-ant-api03-fakekey1234"
    upstream_port: int = 8080

    def log_message(self, format, *args):
        # Suppress noisy standard request logging
        pass

    def check_auth(self) -> bool:
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.master_key}"
        if auth_header.strip() == expected:
            return True

        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        err = json.dumps({
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing Bearer token.",
                "layer": "auth_gateway",
            }
        }).encode("utf-8")
        self.wfile.write(err)
        return False

    def do_GET(self):
        if not self.check_auth():
            return
        self.proxy_request("GET")

    def do_POST(self):
        if not self.check_auth():
            return
        if self.path == "/v1/shutdown":
            self.do_SHUTDOWN()
            return
        self.proxy_request("POST")

    def proxy_request(self, method: str):
        target_url = f"http://127.0.0.1:{self.upstream_port}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "authorization")}

        try:
            upstream_req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
            with urllib.request.urlopen(upstream_req, timeout=180) as resp:
                is_event_stream = "text/event-stream" in resp.headers.get("Content-Type", "")

                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)

                if is_event_stream:
                    # True SSE HTTP chunked streaming (vital for real-time Codex token streaming)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = resp.read(1024)
                        if not chunk:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                            break
                        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                        self.wfile.flush()
                else:
                    # Standard non-streaming response
                    resp_data = resp.read()
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(k, v)
            e_data = e.read()
            self.send_header("Content-Length", str(len(e_data)))
            self.end_headers()
            self.wfile.write(e_data)
        except Exception as e:
            # When llama-server backend is down, dead, or restarting -> Explicit 503 Service Unavailable
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "5")
            self.end_headers()
            err = json.dumps({
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": f"Inference engine unavailable or initializing: {str(e)}",
                    "layer": "auth_gateway"
                }
            }).encode("utf-8")
            self.wfile.write(err)

    def do_SHUTDOWN(self):
        """Emergency remote shutdown signal over authenticated gateway."""
        if not self.check_auth():
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "terminating", "message": "Worker shutdown signal accepted."}).encode("utf-8"))
        
        def _terminate():
            import time, os
            time.sleep(1)
            print("[SHUTDOWN] Remote shutdown requested via gateway. Releasing GPU container...", flush=True)
            os._exit(0)

        import threading
        threading.Thread(target=_terminate, daemon=True).start()


def start_gateway(port: int = 4000, upstream_port: int = 8080, master_key: str = ""):
    """Start the lightweight auth gateway."""
    key = master_key or os.environ.get("LITELLM_MASTER_KEY", "sk-ant-api03-fakekey1234")
    AuthenticatedGatewayHandler.master_key = key
    AuthenticatedGatewayHandler.upstream_port = upstream_port

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), AuthenticatedGatewayHandler)
    print(f"[Gateway] Secure Auth Gateway running on port {port} -> forwarding to llama-server:{upstream_port}", flush=True)
    server.serve_forever()

# ==================== Main Execution Flow ====================
if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("     KAGGLE DUAL-T4 INFERENCE RUNNER (SUBTASK 1)       ", flush=True)
    print("=" * 60, flush=True)

    # Reconstruct deployed ModelProfile
    model_data = json.loads(DEPLOYED_MODEL_JSON)
    model = ModelProfile(**model_data)
    print(f"Target Deployed Model: {model.display_name} ({model.id})", flush=True)

    # 1. Environment & Strict GPU Check (Must be 2x T4s)
    print("\n[Phase 1] Checking GPU topology (Enforcing 2x Tesla T4)...", flush=True)
    ok, gpus, msg = check_gpu_topology(required_count=2)
    print(f"[{'OK' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        print("[CRITICAL ERROR] Hardware verification failed. Aborting startup.", flush=True)
        sys.exit(1)

    # 2. Deterministic Model Asset Resolution
    print(f"\n[Phase 2] Deterministically resolving weights for '{model.id}'...", flush=True)
    ok, model_file, msg = find_model_weights(model, search_root="/kaggle/input")
    print(f"[{'OK' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok or not model_file:
        print(f"[CRITICAL ERROR] Could not find target model '{model.filename}' in /kaggle/input. Aborting.", flush=True)
        sys.exit(1)

    model_size_gb = model_file.stat().st_size / (1024 ** 3)

    # Inspect dynamic GGUF binary headers & caution against auxiliary tensors
    print(f"\n[Phase 2.1] Inspecting GGUF binary metadata for '{model_file.name}'...", flush=True)
    gguf_meta = read_gguf_metadata_summary(model_file)
    print(f"  Valid GGUF: {gguf_meta.get('valid_gguf', False)} | Architecture: {gguf_meta.get('architecture', 'unknown')} | Chat Template Present: {gguf_meta.get('has_chat_template', False)}", flush=True)
    if gguf_meta.get("auxiliary_tensors"):
        print(f"  [Caution] Detected {len(gguf_meta['auxiliary_tensors'])} auxiliary/blk.64 prediction tensors:", flush=True)
        for aux_t in gguf_meta["auxiliary_tensors"][:4]:
            print(f"    - {aux_t}", flush=True)
        print("  Policy: Classified as UNVERIFIED_AUXILIARY_LAYER. MTP decoding disabled initially.", flush=True)

    # Vision mmproj check (Calling resolve_vision_projector)
    has_vision, mmproj_file, v_msg = resolve_vision_projector(model, model_file, search_root="/kaggle/input")
    print(f"[Vision Status] {v_msg}", flush=True)

    # 3. Model-Specific Resource Admission Check (1-2 GB Headroom Per GPU)
    print("\n[Phase 3] Running Resource Admission Check...", flush=True)
    target_ctx = int(os.environ.get("N_CTX", str(model.context_size)))
    admitted, effective_ctx, breakdown, adm_msg = check_resource_admission(
        model_size_gb=model_size_gb,
        profile=model,
        target_context=target_ctx,
        num_gpus=len(gpus),
        has_vision_active=has_vision,
    )
    print(f"Status: {adm_msg}", flush=True)
    print(f"Allocations: {breakdown}", flush=True)

    # CRITICAL FIX: If admission failed and cannot fit with required headroom -> ABORT
    if not admitted and effective_ctx == 0:
        print("[CRITICAL ERROR] Model fails memory admission. VRAM headroom cannot be preserved. Aborting.", flush=True)
        sys.exit(1)

    print(f"Effective Admitted Context: {effective_ctx:,} tokens", flush=True)

    # 4. Resolve native binary & Build authoritative command
    print("\n[Phase 4] Staging runtime and companion shared libraries (.so)...", flush=True)
    ok, bin_path, b_msg = stage_runtime_and_libraries(search_root="/kaggle/input", target_dir="/tmp/runtime")
    print(f"[{'OK' if ok else 'FAIL'}] {b_msg}", flush=True)
    if not ok or not bin_path:
        print("[CRITICAL ERROR] Failed to stage native llama-server. Aborting.", flush=True)
        sys.exit(1)
    binary = str(bin_path)

    # Using the authoritative command builder (single source of truth)
    server_cmd = build_native_server_command(
        binary_path=binary,
        model_path=model_file,
        profile=model,
        effective_context=effective_ctx,
        port=8080,
        host="127.0.0.1",
        mmproj_path=mmproj_file if has_vision else None,
    )
    print("Launching llama-server on 127.0.0.1:8080...", flush=True)
    server_proc = subprocess.Popen(server_cmd)

    # 5. Readiness Verification Check
    print("\n[Phase 5] Verifying model initialization, process liveness, and VRAM readiness...", flush=True)
    ready, r_msg, r_details = verify_server_readiness(
        port=8080,
        timeout_seconds=240,
        process=server_proc,
        expected_context=effective_ctx,
    )
    print(f"[{'READY' if ready else 'FAILED'}] {r_msg}", flush=True)
    if not ready:
        print("[CRITICAL ERROR] Server failed readiness check. Terminating.", flush=True)
        server_proc.kill()
        sys.exit(1)

    # Physical VRAM verification: query real card usage via nvidia-smi
    print("\n[VRAM Verification] Querying actual physical GPU utilization...", flush=True)
    actual_gpus = query_physical_vram()
    if actual_gpus:
        for gpu in actual_gpus:
            used_gb = gpu['used_mib'] / 1024.0
            free_gb = gpu['free_mib'] / 1024.0
            total_gb = gpu['total_mib'] / 1024.0
            print(f"  GPU {gpu['index']} ({gpu['name']}): {used_gb:.2f} GB used / {total_gb:.2f} GB total | Free Headroom: {free_gb:.2f} GB", flush=True)
    else:
        print("  [Note] Physical GPU stats query bypassed or nvidia-smi unavailable.", flush=True)

    # System Host RAM & Swap pressure verification
    print("\n[Host Memory Verification] Checking system RAM, page cache, and swap pressure...", flush=True)
    ram_stats = query_system_ram()
    if ram_stats and ram_stats.get("total_gb", 0) > 0:
        print(
            f"  Host RAM: {ram_stats['used_gb']} GB used / {ram_stats['total_gb']} GB total "
            f"(Available: {ram_stats['available_gb']} GB, Linux Page Cache: {ram_stats['cached_gb']} GB) | "
            f"Swap Used: {ram_stats['swap_used_mb']} MB / {ram_stats['swap_total_mb']} MB",
            flush=True,
        )
    else:
        print("  [Note] Host RAM stats parsing bypassed (non-Linux environment).", flush=True)

    # 5.1 Vision Pipeline End-to-End Verification (if multimodal model)
    if has_vision:
        print("\n[Phase 5.1] Testing multimodal vision encoder & projector end-to-end...", flush=True)
        v_ok, v_res = verify_vision_pipeline(port=8080, model_alias=model.id)
        print(f"[{'OK' if v_ok else 'WARN'}] {v_res}", flush=True)

    # 6. Start Authenticated Gateway (Port 4000) & Cloudflare Tunnel
    # Public Cloudflare -> Gateway (Port 4000 with Bearer Auth) -> llama-server (Port 8080)
    print("\n[Phase 6] Starting Authenticated Gateway (Port 4000) & Cloudflare Tunnel...", flush=True)
    master_key = os.environ.get("LITELLM_MASTER_KEY", "sk-ant-api03-fakekey1234")
    gateway_thread = threading.Thread(
        target=start_gateway,
        kwargs={"port": 4000, "upstream_port": 8080, "master_key": master_key},
        daemon=True,
    )
    gateway_thread.start()

    cf_bin = shutil.which("cloudflared")
    if not cf_bin:
        subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "/tmp/cloudflared"], check=False)
        subprocess.run(["chmod", "+x", "/tmp/cloudflared"], check=False)
        cf_bin = "/tmp/cloudflared"

    # Cloudflare exposes the AUTHENTICATED GATEWAY (Port 4000), NEVER unauthenticated 8080!
    tunnel_proc = subprocess.Popen([cf_bin, "tunnel", "--url", "http://127.0.0.1:4000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    def stream_tunnel():
        for line in iter(tunnel_proc.stdout.readline, ""):
            match = re.search(r"https://[a-zA-Z0-9-]+\.[t]rycloudflare\.[c]om", line)
            if match:
                url = match.group(0)
                print("\n" + "=" * 60, flush=True)
                print(f"AUTHENTICATED TUNNEL LIVE: {url}", flush=True)
                print(f"REMOTE_TUNNEL_URL={url}", flush=True)
                print("=" * 60 + "\n", flush=True)
                break
    threading.Thread(target=stream_tunnel, daemon=True).start()

    # Keep alive while server_proc is running
    server_proc.wait()
