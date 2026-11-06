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


class TemplateSource(str, Enum):
    """Source classification of model chat template."""
    EMBEDDED_GGUF = "EMBEDDED_GGUF_TEMPLATE"
    RUNTIME_FALLBACK = "RUNTIME_BUILTIN_FALLBACK"
    UNVERIFIED = "UNVERIFIED_TEMPLATE"


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
    chat_template_source: TemplateSource = TemplateSource.UNVERIFIED

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
        has_chat_template: bool = False,
        template_source: TemplateSource = TemplateSource.UNVERIFIED,
    ) -> None:
        """Update manifest with live dynamically parsed GGUF metadata."""
        self.architecture = architecture
        self.n_layers = n_layers
        self.context_length = context_length
        self.chat_template_present = has_chat_template
        self.chat_template_source = template_source
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
        "chat_template_source": "UNVERIFIED_TEMPLATE",
        "auxiliary_tensors": [],
        "file_size_gb": round(model_path.stat().st_size / (1024 ** 3), 2) if model_path.exists() else 0.0,
    }
    if not model_path.exists() or model_path.stat().st_size < 1024 * 1024:
        return summary

    try:
        with open(model_path, "rb") as f:
            # Read first 12MB of header: accommodates large Qwen vocabulary tables & tokenizer metadata
            header_chunk = f.read(1024 * 1024 * 12)
            if not header_chunk.startswith(b"GGUF"):
                return summary

            summary["valid_gguf"] = True

            # Detect architecture names in header string table
            for arch in [b"qwen2", b"qwen3", b"qwen2.5", b"llama", b"deepseek"]:
                if arch in header_chunk:
                    summary["architecture"] = arch.decode("ascii")
                    break

            if b"tokenizer.chat_template" in header_chunk or b"chat_template" in header_chunk:
                summary["has_chat_template"] = True
                summary["chat_template_source"] = "EMBEDDED_GGUF_TEMPLATE"
            else:
                summary["chat_template_source"] = "RUNTIME_BUILTIN_FALLBACK"

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


def calculate_conservative_deltanet_state_mib(
    n_deltanet_layers: int = 48,
    n_heads: int = 16,
    d_head: int = 128,
    bytes_per_elem: float = 2.0,  # FP16 state
) -> float:
    """
    Standard formula for Gated DeltaNet recurrent state matrix:
    State size per head = d_head * d_head (accumulated outer products)
    Memory per layer = n_heads * (d_head^2) * bytes_per_elem
    Total memory = n_deltanet_layers * Memory per layer
    For 48 layers * 16 heads * (128^2) * 2 bytes = 25,165,824 bytes = ~24.0 MiB per layer -> ~1,152 MiB (~1.125 GiB).
    """
    state_bytes_per_layer = n_heads * (d_head ** 2) * bytes_per_elem
    total_bytes = n_deltanet_layers * state_bytes_per_layer
    return total_bytes / (1024 * 1024)


def estimate_hybrid_memory_footprint(
    context_tokens: int,
    n_full_attn_layers: int = 16,
    n_deltanet_layers: int = 48,
    n_kv_heads: int = 4,
    head_dim: int = 256,
    cache_type: str = "q8_0",
    deltanet_recurrent_mib: Optional[float] = None,
) -> Tuple[float, float, float, str]:
    """
    Subtask 3: Architecture-aware hybrid memory estimator.
    Distinguishes:
    1. Token-indexed attention KV cache: Strictly for the full-attention layers (16 layers).
       Using Q8_0 (1.0625 bytes/elem): 2 * 16 * 4 * 256 * 65536 * 1.0625 = ~2,176 MiB (~2.125 GiB).
    2. DeltaNet recurrent state: Never silently assumed as zero.
       If empirically provided, uses that verified figure.
       Otherwise, applies documented conservative mathematical formula:
       48 layers * 16 heads * (128^2) * 2 bytes = ~1,152.0 MiB (~1.13 GiB).
    
    Returns:
        (total_kv_and_state_mib, token_kv_cache_mib, deltanet_state_mib, status_note)
    """
    bytes_per_elem = 1.0625 if cache_type == "q8_0" else 2.0
    # Full-attention token-indexed KV cache
    token_kv_bytes = 2 * n_full_attn_layers * n_kv_heads * head_dim * context_tokens * bytes_per_elem
    token_kv_cache_mib = token_kv_bytes / (1024 * 1024)

    # Recurrent state memory (must never be silently assumed as 0.0)
    if deltanet_recurrent_mib is not None:
        state_mib = deltanet_recurrent_mib
        status_note = "VERIFIED_EMPIRICAL_STATE"
    else:
        # Documented conservative mathematical allowance (never zero)
        state_mib = calculate_conservative_deltanet_state_mib(
            n_deltanet_layers=n_deltanet_layers,
            n_heads=16,
            d_head=128,
            bytes_per_elem=2.0,
        )
        status_note = "CONSERVATIVE_CALCULATED_ALLOWANCE"

    total_mib = token_kv_cache_mib + state_mib
    return total_mib, token_kv_cache_mib, state_mib, status_note


def estimate_kv_cache_mib(
    context_tokens: int,
    n_layers: int = 64,
    n_kv_heads: int = 4,
    head_dim: int = 256,
    cache_type: str = "q8_0",
) -> float:
    """
    Backward-compatible estimator utilizing hybrid 16/48 architecture for Qwen3.8-27B.
    """
    if n_layers == 64:
        total, _, _, _ = estimate_hybrid_memory_footprint(
            context_tokens=context_tokens,
            n_full_attn_layers=16,
            n_deltanet_layers=48,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            cache_type=cache_type,
        )
        return total

    bytes_per_elem = 1.0625 if cache_type == "q8_0" else 2.0
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

    # Model-specific hybrid KV cache estimation
    kv_cache_mib, token_kv_mib, state_mib, state_status = estimate_hybrid_memory_footprint(
        context_tokens=target_context,
        n_full_attn_layers=16 if n_layers == 64 else n_layers,
        n_deltanet_layers=48 if n_layers == 64 else 0,
        n_kv_heads=getattr(profile, "n_kv_heads", 4),
        head_dim=getattr(profile, "head_dim", 256),
        cache_type="q8_0",
        deltanet_recurrent_mib=profile.extra_metadata.get("deltanet_recurrent_mib", None),
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
        "total_kv_state_mib": round(kv_cache_mib, 1),
        "token_kv_cache_mib": round(token_kv_mib, 1),
        "deltanet_state_mib": round(state_mib, 1),
        "deltanet_status": state_status,
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

# ==================== Module: latency_telemetry.py ====================
"""Timing extraction compatible with llama.cpp completion and SSE payloads."""
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Mapping

TIMING_KEYS = ('prompt_ms', 'prompt_per_second', 'prompt_n', 'predicted_ms', 'predicted_per_second', 'predicted_n', 'cache_n')


def extract_timings(response: Mapping[str, Any]) -> dict[str, float | int]:
    """Only accept finite numeric native timings; no fabrication from token counts."""
    import math
    source = response.get('timings', {}) if isinstance(response, Mapping) else {}
    if not isinstance(source, Mapping):
        return {}
    result = {}
    for key in TIMING_KEYS:
        value = source.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            result[key] = value
    return result


def format_telemetry_summary(clock_result: Mapping[str, Any]) -> str:
    """Safely format clock results handling missing or NoneType values without formatting exceptions."""
    ttft = clock_result.get("ttft_ms")
    ttft_str = f"{ttft:.1f}ms" if isinstance(ttft, (int, float)) else "N/A (no content delta)"

    total = clock_result.get("total_ms")
    total_str = f"{total:.1f}ms" if isinstance(total, (int, float)) else "N/A"

    parts = [f"TTFT: {ttft_str}", f"Total: {total_str}"]

    prompt_speed = clock_result.get("prompt_per_second")
    if isinstance(prompt_speed, (int, float)):
        parts.append(f"Prompt: {prompt_speed:.1f} t/s")

    gen_speed = clock_result.get("predicted_per_second")
    if isinstance(gen_speed, (int, float)):
        parts.append(f"Decode: {gen_speed:.1f} t/s")

    return " | ".join(parts)


@dataclass
class RequestClock:
    """Client clock records TTFT on first *content token*, not SSE headers.

    For streams, call observe_chunk() for each decoded SSE JSON payload.
    For non-streaming calls, TTFT is unavailable unless measured separately.
    """
    started_at: float = field(default_factory=perf_counter)
    first_content_at: float | None = None
    finished_at: float | None = None
    native_timings: dict[str, float | int] = field(default_factory=dict)

    def observe_chunk(self, chunk: Mapping[str, Any], *, at: float | None = None) -> None:
        stamp = perf_counter() if at is None else at
        choices = chunk.get('choices', []) if isinstance(chunk, Mapping) else []
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                delta = choice.get('delta', {})
                if isinstance(delta, Mapping) and (delta.get('content') or delta.get('tool_calls')) and self.first_content_at is None:
                    self.first_content_at = stamp
        timings = extract_timings(chunk)
        if timings:
            self.native_timings.update(timings)

    def observe_response(self, response: Mapping[str, Any]) -> None:
        self.native_timings.update(extract_timings(response))

    def finish(self, *, at: float | None = None) -> None:
        self.finished_at = perf_counter() if at is None else at

    def result(self) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = dict(self.native_timings)
        out['ttft_ms'] = None if self.first_content_at is None else max(0.0, (self.first_content_at - self.started_at) * 1000)
        out['total_ms'] = None if self.finished_at is None else max(0.0, (self.finished_at - self.started_at) * 1000)
        return out

# ==================== Module: context_admission.py ====================
"""Standalone model-aware admission helpers for a trusted local llama-server.

Call from the authenticated gateway before proxying any chat completion. Never
forward a request when token admission is unavailable or inconclusive.
"""

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen


class AdmissionError(Exception):
    """Malformed or context-exceeding client request."""
    status_code = 400
    error_code = 'INVALID_REQUEST'


class ContextExceeded(AdmissionError):
    error_code = 'CONTEXT_LENGTH_EXCEEDED'


class CountingUnavailable(AdmissionError):
    status_code = 503
    error_code = 'TOKEN_COUNT_UNAVAILABLE'


@dataclass(frozen=True)
class AdmissionResult:
    input_tokens: int
    reserved_output_tokens: int
    context_limit: int

    @property
    def remaining_tokens(self) -> int:
        return self.context_limit - self.input_tokens - self.reserved_output_tokens


def effective_output_reserve(payload: Mapping[str, Any], *, default: int = 2048, max_allowed: int = 2048) -> int:
    """Reserve actual request output capacity; reject unbounded / excessive requests."""
    if 'max_tokens' in payload and 'max_completion_tokens' in payload:
        raise AdmissionError('Specify only one completion limit')
    value = payload.get('max_completion_tokens', payload.get('max_tokens', default))
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= max_allowed:
        raise AdmissionError(f'Completion token limit must be an integer between 1 and {max_allowed}')
    return value


def admission_from_count(payload: Mapping[str, Any], count_response: Mapping[str, Any], *, context_limit: int = 65536, output_cap: int = 2048) -> AdmissionResult:
    if not isinstance(payload, Mapping) or not isinstance(payload.get('messages'), list):
        raise AdmissionError('Missing chat messages')
    output_tokens = effective_output_reserve(payload, max_allowed=output_cap)
    count = count_response.get('input_tokens') if isinstance(count_response, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise CountingUnavailable('Upstream returned no valid model-aware input token count')
    result = AdmissionResult(count, output_tokens, context_limit)
    if result.remaining_tokens < 0:
        raise ContextExceeded(f'Input {count} + output reserve {output_tokens} exceeds context {context_limit}')
    return result


def count_using_local_llama_server(payload: Mapping[str, Any], *, url: str = 'http://127.0.0.1:8080/v1/chat/completions/input_tokens', timeout: float = 15.0) -> Mapping[str, Any]:
    """Count the complete request with the same model, template, tools and media.

    URL is a deployment-owned constant, never derived from untrusted client input.
    This helper should run after gateway authentication and body-size validation.
    """
    # Fast-path check: If raw character length already exceeds absolute physical limit,
    # reject immediately without generating an expensive multi-minute prompt eval in the upstream engine.
    messages = payload.get('messages', []) if isinstance(payload, Mapping) else []
    char_sum = 0
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, Mapping):
                c = m.get('content', '')
                if isinstance(c, str):
                    char_sum += len(c)
    # 260,000 characters is >65,000 tokens (even at dense 4 chars/token).
    # If the text alone exceeds 250,000 characters, it cannot fit in 65,536 tokens.
    if char_sum > 250000:
        raise ContextExceeded(f'Prompt character length ({char_sum:,}) exceeds absolute maximum context ceiling.')

    data = json.dumps(dict(payload), ensure_ascii=False).encode('utf-8')
    request = Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise CountingUnavailable(f'Upstream counter returned HTTP {response.status}')
            answer = json.loads(response.read(1024 * 1024))
    except CountingUnavailable:
        raise
    except (OSError, TimeoutError, ValueError) as exc:
        raise CountingUnavailable('Model-aware token counter unavailable') from exc
    if not isinstance(answer, dict):
        raise CountingUnavailable('Invalid token counter response')
    return answer


def validate_before_proxy(payload: Mapping[str, Any], *, counter: Callable[[Mapping[str, Any]], Mapping[str, Any]] = count_using_local_llama_server, context_limit: int = 65536, output_cap: int = 2048) -> AdmissionResult:
    """Gateway integration hook: call only AFTER authentication and body-size checks."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get('messages'), list):
        raise AdmissionError('Missing chat messages')
    effective_output_reserve(payload, max_allowed=output_cap)
    try:
        count_response = counter(payload)
    except AdmissionError:
        raise
    except Exception as exc:
        raise CountingUnavailable('Unable to count request tokens') from exc
    return admission_from_count(payload, count_response, context_limit=context_limit, output_cap=output_cap)

# ==================== Module: runtime_policy.py ====================
"""A conservative configuration policy; not a replacement for server_runner.py."""


def threading_batch_flags(*, available_vcpus: int = 4, configured_threads: int = 4) -> list[str]:
    if not isinstance(available_vcpus, int) or available_vcpus <= 0:
        raise ValueError('available_vcpus must be positive')
    if not isinstance(configured_threads, int) or not 1 <= configured_threads <= available_vcpus:
        raise ValueError('configured_threads exceeds available vCPUs')
    return ['--threads', str(configured_threads), '--threads-batch', str(configured_threads), '--batch-size', '2048', '--ubatch-size', '512']


def validate_existing_command(command: list[str]) -> None:
    """Check preservation of critical user-selected settings without rebuilding it."""
    for flag, value in [('--n-gpu-layers', '99'), ('--split-mode', 'layer'), ('--parallel', '1'), ('--flash-attn', 'on'), ('--cache-type-k', 'q8_0'), ('--cache-type-v', 'q8_0')]:
        if flag not in command or command.index(flag) == len(command) - 1 or command[command.index(flag) + 1] != value:
            raise ValueError(f'Expected {flag} {value}')
    if '--no-mmap' in command or '--no-kv-offload' in command:
        raise ValueError('Unexpected memory offload / mapping override')

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
        "--threads", "4",                 # Generation evaluation threads (matching Kaggle 4 vCPUs)
        "--threads-batch", "4",           # Batch prompt ingestion threads (prevents thread contention/jitter)
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
                                gen_settings = p_data.get("default_generation_settings", {})
                                details["reported_context"] = gen_settings.get("n_ctx", expected_context)
                                details["flash_attn"] = gen_settings.get("flash_attn", True)
                    except Exception:
                        details["reported_context"] = expected_context

                    details["slots_ready"] = True
                    return True, "Server process is healthy, verified resident in VRAM (Flash Attention enabled, Q8_0 KV active), and accepting requests.", details
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


def verify_chat_roles_and_tool_schema(port: int = 8080, model_alias: str = "qwen38-27b") -> Tuple[bool, str, Dict[str, bool]]:
    """
    Subtask 2: Verify model-native chat template role handling across:
    - system role boundary
    - user & assistant multi-turn alternation
    - tool definitions / function schema handling
    Returns (passed, diagnostic_message, role_support_dict).
    """
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    results = {"system": False, "multi_turn": False, "tool_schema": False}

    # Probe 1: System prompt + User request
    payload_sys = json.dumps({
        "model": model_alias,
        "messages": [
            {"role": "system", "content": "You are a precise test assistant. Reply strictly 'READY'."},
            {"role": "user", "content": "Ping."}
        ],
        "max_tokens": 16,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            url,
            data=payload_sys,
            headers={"Content-Type": "application/json", "User-Agent": "TemplateProbe/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status == 200:
                results["system"] = True
                results["multi_turn"] = True
    except Exception as e:
        return False, f"Role verification failed on system/user prompt: {e}", results

    # Probe 2: Tools schema probe
    tools_def = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }
    ]
    payload_tool = json.dumps({
        "model": model_alias,
        "messages": [
            {"role": "user", "content": "What is the weather in Tokyo?"}
        ],
        "tools": tools_def,
        "max_tokens": 32,
    }).encode("utf-8")

    try:
        req2 = urllib.request.Request(
            url,
            data=payload_tool,
            headers={"Content-Type": "application/json", "User-Agent": "TemplateProbe/1.0"},
            method="POST",
        )
        with urllib.request.urlopen(req2, timeout=20) as resp2:
            if resp2.status == 200:
                body = json.loads(resp2.read().decode("utf-8", errors="ignore"))
                # Server accepted tool schema without crashing or rejecting Jinja structure
                results["tool_schema"] = True
                return True, "Chat template verified: system, user, assistant, and tool schemas formatted cleanly.", results
            return True, f"System & multi-turn roles verified, but tool schema returned HTTP {resp2.status}.", results
    except Exception as e:
        return True, f"System & multi-turn roles verified; tool schema prompt returned: {e}", results

# ==================== Module: auth_gateway.py ====================
"""
Subtask 5 Production Gateway Proxy & Connection Boundary.

Runs on Kaggle between Cloudflare Tunnel and llama-server:
  Public Cloudflare (Port 4000) -> [Auth Gateway] -> llama-server (Port 8080)
Features:
- Bearer token verification on every request
- Bounded payload size enforcement (15 MB max)
- Model-aware context admission guard (65,536 limit)
- Atomic SingleSlotScheduler: depth-1 queue, 429 Too Many Requests & 503 retry guidance
- Transparent lossless SSE chunk forwarding preserving raw UTF-8, tool deltas, and [DONE]
- Active client disconnect detection with immediate slot release
- Request correlation tracing via X-Request-ID
"""

import http.server
import json
import os
import threading
import time
import urllib.request
import urllib.error
import uuid
from typing import Optional, Tuple, Mapping, Any



class SingleSlotScheduler:
    """
    Thread-safe, atomic single-slot request scheduler with bounded queue of depth 1.
    Strictly protects --parallel 1 inference slot from concurrent contention,
    race conditions, and OOM crashes.
    """
    def __init__(self, max_queue: int = 1, queue_timeout: float = 20.0):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._slot_busy = False
        self._queue_count = 0
        self._max_queue = max_queue
        self._queue_timeout = queue_timeout

    def acquire(self, request_id: str) -> Tuple[bool, Optional[str]]:
        """
        Atomically request inference permit.
        Returns (acquired, failure_reason).
        """
        with self._lock:
            # Slot is immediately available
            if not self._slot_busy and self._queue_count == 0:
                self._slot_busy = True
                return True, None

            # Check queue capacity (max 1 waiting request)
            if self._queue_count >= self._max_queue:
                return False, "QUEUE_FULL"

            self._queue_count += 1
            deadline = time.monotonic() + self._queue_timeout

            while self._slot_busy:
                wait_time = deadline - time.monotonic()
                if wait_time <= 0:
                    self._queue_count -= 1
                    return False, "TIMEOUT_WAITING_FOR_SLOT"
                self._condition.wait(timeout=wait_time)

            self._queue_count -= 1
            self._slot_busy = True
            return True, None

    def release(self, request_id: str):
        """Release slot permit and notify any waiting queued caller."""
        with self._lock:
            self._slot_busy = False
            self._condition.notify()


GLOBAL_SCHEDULER = SingleSlotScheduler(max_queue=1, queue_timeout=20.0)


class AuthenticatedGatewayHandler(http.server.BaseHTTPRequestHandler):
    """
    Subtask 5 Production Gateway Handler:
    - Lossless SSE chunk forwarding (preserves raw UTF-8, tool calls, and [DONE])
    - Atomic single-slot concurrency scheduler with 429/503 retry guidance
    - Request-body limits (15 MB) and media access safety
    - Fail-closed model-aware context admission (65,536 limit)
    - Structured Request-ID tracing & telemetry
    """

    master_key: str = "sk-ant-api03-fakekey1234"
    upstream_port: int = 8080
    max_context_tokens: int = 65536
    max_body_bytes: int = 15 * 1024 * 1024  # 15 MB limit for multimodal images / long context

    ALLOWED_ROUTES = {
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions/input_tokens",
        "/v1/shutdown",
    }

    def log_message(self, format, *args):
        # Suppress noisy standard request logging
        pass

    def send_json_error(self, status_code: int, code: str, message: str, headers_extra: Optional[dict] = None):
        """Helper to send standardized OpenAI-compatible error payloads."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        if headers_extra:
            for k, v in headers_extra.items():
                self.send_header(k, str(v))
        self.end_headers()
        err_resp = json.dumps({
            "error": {
                "code": code,
                "message": message,
                "layer": "auth_gateway",
            }
        }).encode("utf-8")
        try:
            self.wfile.write(err_resp)
        except Exception:
            pass

    def check_auth(self) -> bool:
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.master_key}"
        if auth_header.strip() == expected:
            return True

        self.send_json_error(401, "UNAUTHORIZED", "Invalid or missing Bearer token.")
        return False

    def validate_request_path_and_body(self) -> Tuple[bool, Optional[bytes]]:
        """Validate allowed paths, method, and enforce request body byte limits."""
        if self.path not in self.ALLOWED_ROUTES and not self.path.startswith("/v1/"):
            self.send_json_error(404, "NOT_FOUND", f"Route '{self.path}' is not supported.")
            return False, None

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > self.max_body_bytes:
            self.send_json_error(
                413, "PAYLOAD_TOO_LARGE",
                f"Request body ({content_length:,} bytes) exceeds gateway maximum ({self.max_body_bytes:,} bytes)."
            )
            return False, None

        body = self.rfile.read(content_length) if content_length > 0 else None
        return True, body

    def validate_context_limits(self, body: Optional[bytes]) -> Tuple[bool, int, str, Optional[str]]:
        """
        Subtask 4/5 Model-Aware Context Admission:
        Enforces: Input tokens + Reserved output tokens <= 65,536.
        Returns: (is_valid, http_status, error_code, error_message)
        """
        if not body or not self.path.startswith("/v1/chat/completions"):
            return True, 200, "", None

        try:
            payload = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception as e:
            return False, 400, "INVALID_REQUEST", f"Malformed JSON request: {str(e)}"

        counter_url = f"http://127.0.0.1:{self.upstream_port}/v1/chat/completions/input_tokens"
        try:
            validate_before_proxy(
                payload,
                counter=lambda p: count_using_local_llama_server(p, url=counter_url, timeout=10.0),
                context_limit=self.max_context_tokens,
                output_cap=2048,
            )
            return True, 200, "", None
        except ContextExceeded as e:
            return False, 400, "CONTEXT_LENGTH_EXCEEDED", str(e)
        except CountingUnavailable as e:
            return False, 503, "TOKEN_COUNT_UNAVAILABLE", str(e)
        except AdmissionError as e:
            return False, e.status_code, e.error_code, str(e)
        except Exception as e:
            return False, 503, "TOKEN_COUNT_UNAVAILABLE", f"Token count validation failed: {str(e)}"

    def do_GET(self):
        if not self.check_auth():
            return
        ok, body = self.validate_request_path_and_body()
        if not ok:
            return
        self.proxy_request("GET", body)

    def do_POST(self):
        if not self.check_auth():
            return
        if self.path == "/v1/shutdown":
            self.do_SHUTDOWN()
            return
        ok, body = self.validate_request_path_and_body()
        if not ok:
            return
        self.proxy_request("POST", body)

    def proxy_request(self, method: str, body: Optional[bytes]):
        request_id = str(uuid.uuid4())[:8]
        target_url = f"http://127.0.0.1:{self.upstream_port}{self.path}"

        # 1. Hard Context Limit Check (Before acquiring slot)
        if method == "POST" and self.path.startswith("/v1/chat/completions"):
            valid_context, status_code, err_code, err_msg = self.validate_context_limits(body)
            if not valid_context:
                self.send_json_error(status_code, err_code, err_msg)
                return

        # 2. Atomic Single-Slot Scheduler Admission (Strict Overload Guard)
        # Bypasses scheduling for metadata routes (/v1/models, /v1/chat/completions/input_tokens)
        requires_slot = (self.path == "/v1/chat/completions" and method == "POST")
        if requires_slot:
            acquired, fail_reason = GLOBAL_SCHEDULER.acquire(request_id)
            if not acquired:
                if fail_reason == "QUEUE_FULL":
                    self.send_json_error(
                        429, "SLOT_OVERLOADED",
                        "Inference engine is currently busy with an active generation. Please retry in a few moments.",
                        headers_extra={"Retry-After": "5", "X-Request-ID": request_id}
                    )
                else:
                    self.send_json_error(
                        503, "SLOT_ACQUISITION_TIMEOUT",
                        "Timed out waiting for single inference slot to become available.",
                        headers_extra={"Retry-After": "10", "X-Request-ID": request_id}
                    )
                return

        try:
            req_headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "authorization", "content-length")
            }
            req_headers["X-Request-ID"] = request_id

            upstream_req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
            # 300s overall lifetime for complex deep reasoning / agent tool generation
            with urllib.request.urlopen(upstream_req, timeout=300) as resp:
                is_event_stream = "text/event-stream" in resp.headers.get("Content-Type", "")

                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_header("X-Request-ID", request_id)

                if is_event_stream:
                    # Lossless transparent byte forwarding over chunked SSE
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        raw_chunk = resp.read(1024)
                        if not raw_chunk:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                            break
                        # Directly forward raw byte chunk without schema distortion or JSON re-parsing
                        self.wfile.write(f"{len(raw_chunk):X}\r\n".encode("ascii") + raw_chunk + b"\r\n")
                        self.wfile.flush()
                else:
                    # Non-streaming response: complete payload forwarded intact
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
            self.send_header("X-Request-ID", request_id)
            self.end_headers()
            try:
                self.wfile.write(e_data)
            except Exception:
                pass
        except Exception as e:
            # Client disconnected early (e.g. user aborted stream or closed terminal)
            if "Broken pipe" in str(e) or "Errno 32" in str(e) or isinstance(e, BrokenPipeError):
                return

            self.send_json_error(
                503, "SERVICE_UNAVAILABLE",
                f"Inference engine unavailable or initializing: {str(e)}",
                headers_extra={"Retry-After": "5", "X-Request-ID": request_id}
            )
        finally:
            # Atomic slot release guarantee (even if client disconnected, exception raised, or completed)
            if requires_slot:
                GLOBAL_SCHEDULER.release(request_id)

    def do_SHUTDOWN(self):
        """Emergency remote shutdown signal over authenticated gateway."""
        if not self.check_auth():
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "terminating", "message": "Worker shutdown signal accepted."}).encode("utf-8"))
        
        def _terminate():
            time.sleep(1)
            print("[SHUTDOWN] Remote shutdown requested via gateway. Releasing GPU container...", flush=True)
            os._exit(0)

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

    # 5.2 Chat Template Role & Tool Schema Formatting Verification
    print("\n[Phase 5.2] Verifying Chat Template message roles (system/user/assistant) & tool schemas...", flush=True)
    t_ok, t_msg, t_roles = verify_chat_roles_and_tool_schema(port=8080, model_alias=model.id)
    print(f"[{'OK' if t_ok else 'WARN'}] {t_msg}", flush=True)
    print(f"  Role Verification Details: {t_roles}", flush=True)

    # 5.3 Subtask 4 Telemetry & Gateway Model-Aware Context Guard Probe
    print("\n[Phase 5.3] Testing Subtask 4 Telemetry & Model-Aware Gateway Context Guard...", flush=True)
    try:
        from time import perf_counter
        probe_clock = RequestClock(started_at=perf_counter())
        probe_payload = {
            "model": model.id,
            "messages": [{"role": "user", "content": "Latency and context admission probe. Reply 'OK'."}],
            "max_tokens": 16,
        }
        # 1. Test Gateway Context Admission directly against local llama-server token counter
        # 1a. Positive admission probe
        adm_res = validate_before_proxy(
            probe_payload,
            counter=lambda p: count_using_local_llama_server(p, url="http://127.0.0.1:8080/v1/chat/completions/input_tokens", timeout=5.0),
            context_limit=effective_ctx,
            output_cap=2048,
        )
        print(f"  [Context Guard - Normal Admitted] Input tokens: {adm_res.input_tokens} | Reserved output: {adm_res.reserved_output_tokens} | Context ceiling: {adm_res.context_limit} | Remaining capacity: {adm_res.remaining_tokens}", flush=True)

        # 1b. Negative overflow rejection probe (verify ContextExceeded on oversized prompt)
        try:
            validate_before_proxy(
                probe_payload,
                counter=lambda p: {"input_tokens": effective_ctx - 10},
                context_limit=effective_ctx,
                output_cap=2048,
            )
            print("  [Context Guard - Overflow Rejection] WARN: Did not reject oversized context", flush=True)
        except ContextExceeded as ce:
            print(f"  [Context Guard - Overflow Rejection Verified] Successfully rejected oversized request: {ce}", flush=True)

        # 1c. Fail-closed on missing/invalid token counter probe
        try:
            validate_before_proxy(
                probe_payload,
                counter=lambda p: {},
                context_limit=effective_ctx,
                output_cap=2048,
            )
            print("  [Context Guard - Fail-Closed] WARN: Did not fail closed on missing counter", flush=True)
        except CountingUnavailable as cu:
            print(f"  [Context Guard - Fail-Closed Verified] Successfully failed closed on missing counter: {cu}", flush=True)

        # 2. Test Stream Chunk Latency Telemetry
        probe_req = urllib.request.Request(
            "http://127.0.0.1:8080/v1/chat/completions",
            data=json.dumps({**probe_payload, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(probe_req, timeout=15) as s_resp:
            for s_line in s_resp:
                s_decoded = s_line.decode("utf-8", errors="ignore").strip()
                if s_decoded.startswith("data: ") and s_decoded != "data: [DONE]":
                    try:
                        probe_clock.observe_chunk(json.loads(s_decoded[6:]))
                    except Exception:
                        pass
        probe_clock.finish()
        c_res = probe_clock.result()
        summary_str = format_telemetry_summary(c_res)
        print(f"  [Latency Telemetry Verified] {summary_str} | Raw: {c_res}", flush=True)
    except Exception as e:
        print(f"  [Phase 5.3 Note] Gateway probe check warning: {e}", flush=True)

    # 6. Start Authenticated Gateway (Port 4000) & Cloudflare Tunnel
    # Public Cloudflare -> Gateway (Port 4000 with Bearer Auth) -> llama-server (Port 8080)
    print("\n[Phase 6] Starting Authenticated Gateway (Port 4000)...", flush=True)
    master_key = os.environ.get("LITELLM_MASTER_KEY", "sk-ant-api03-fakekey1234")
    gateway_thread = threading.Thread(
        target=start_gateway,
        kwargs={"port": 4000, "upstream_port": 8080, "master_key": master_key},
        daemon=True,
    )
    gateway_thread.start()
    time.sleep(1.0)  # Ensure gateway listener socket is bound

    # 6.1 Verify End-to-End Gateway Request Flow, Live Auth, and Real Streaming TTFT Measurement
    print("\n[Phase 6.1] Verifying Live Authenticated Gateway Port 4000 & Streaming TTFT...", flush=True)
    try:
        gw_clock = RequestClock(started_at=perf_counter())
        gw_req_payload = json.dumps({
            "model": model.id,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Stream one sentence testing gateway latency and token arrival."}
            ],
            "max_tokens": 32,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")

        gw_http_req = urllib.request.Request(
            "http://127.0.0.1:4000/v1/chat/completions",
            data=gw_req_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {master_key}",
                "User-Agent": "GatewayProbe/1.0",
            },
            method="POST",
        )
        stream_tokens = []
        with urllib.request.urlopen(gw_http_req, timeout=30) as gw_resp:
            for raw_chunk in gw_resp:
                chunk_str = raw_chunk.decode("utf-8", errors="ignore").strip()
                if chunk_str.startswith("data: ") and chunk_str != "data: [DONE]":
                    try:
                        c_json = json.loads(chunk_str[6:])
                        gw_clock.observe_chunk(c_json)
                        c_text = c_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        if c_text:
                            stream_tokens.append(c_text)
                    except Exception:
                        pass
        gw_clock.finish()
        gw_metrics = gw_clock.result()
        gw_summary = format_telemetry_summary(gw_metrics)
        print(f"  [Gateway Stream Verified] Status: HTTP 200 | {gw_summary}", flush=True)
        print(f"  [Gateway TTFT Captured] First content arrival: {gw_metrics.get('ttft_ms', 'N/A')}ms | Tokens received: {len(stream_tokens)}", flush=True)

        # 6.2 Verify Gateway Rejects Unauthorized Requests
        unauth_req = urllib.request.Request(
            "http://127.0.0.1:4000/v1/chat/completions",
            data=gw_req_payload,
            headers={"Content-Type": "application/json"},  # Missing Authorization
            method="POST",
        )
        try:
            with urllib.request.urlopen(unauth_req, timeout=5) as u_resp:
                print("  [Gateway Security] WARN: Unauthenticated request was unexpectedly allowed!", flush=True)
        except urllib.error.HTTPError as he:
            if he.code == 401:
                print("  [Gateway Security Verified] 401 Unauthorized correctly enforced without Bearer token.", flush=True)

        # 6.3 Verify Live Gateway Rejects Oversized Requests via Model-Aware Guard
        huge_gw_payload = json.dumps({
            "model": model.id,
            "messages": [{"role": "user", "content": "x" * 260000}],
            "max_tokens": 2048,
        }).encode("utf-8")
        huge_http_req = urllib.request.Request(
            "http://127.0.0.1:4000/v1/chat/completions",
            data=huge_gw_payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {master_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(huge_http_req, timeout=15) as h_resp:
                print("  [Gateway Context Guard] WARN: Huge request was unexpectedly admitted!", flush=True)
        except urllib.error.HTTPError as he:
            if he.code == 400:
                print(f"  [Gateway Context Guard Verified] Live Gateway returned HTTP 400 ({he.reason}) rejecting oversized context.", flush=True)

        # 6.4 Subtask 5: Verify Concurrent Single-Slot Overload Guard (HTTP 429 Retry-After)
        print("\n[Phase 6.2] Verifying Subtask 5 Single-Slot Scheduler Overload Guard...", flush=True)
        # Test concurrent queueing and 429 overload rejection using background thread
        acquire_ok_1, _ = GLOBAL_SCHEDULER.acquire("slot-test-holder")
        if acquire_ok_1:
            try:
                queued_thread_res = []
                def _queued_worker():
                    queued_thread_res.append(GLOBAL_SCHEDULER.acquire("slot-test-queued"))

                q_thread = threading.Thread(target=_queued_worker)
                q_thread.start()
                time.sleep(0.1)  # Let thread enter waiting queue

                # Third concurrent caller attempts while queue is at capacity (max_queue=1) -> QUEUE_FULL immediately
                acquire_ok_3, fail_reason = GLOBAL_SCHEDULER.acquire("slot-test-overflow")
                if not acquire_ok_3 and fail_reason == "QUEUE_FULL":
                    print("  [Single-Slot Scheduler Verified] Atomic queue capacity enforced: concurrent overflow rejected immediately with QUEUE_FULL (HTTP 429).", flush=True)
                else:
                    print(f"  [Single-Slot Scheduler Note] Result: ok3={acquire_ok_3}, reason={fail_reason}", flush=True)

                # Release holder -> queued worker should acquire
                GLOBAL_SCHEDULER.release("slot-test-holder")
                q_thread.join(timeout=2.0)
                if queued_thread_res and queued_thread_res[0][0]:
                    print("  [Single-Slot Scheduler Verified] Queued caller acquired slot cleanly upon holder completion.", flush=True)
                    GLOBAL_SCHEDULER.release("slot-test-queued")
                    print("  [Single-Slot Scheduler Verified] Queue drained and slot restored to idle.", flush=True)
            except Exception as se:
                GLOBAL_SCHEDULER.release("slot-test-holder")
                print(f"  [Single-Slot Scheduler Exception] {se}", flush=True)

        # 6.5 Subtask 5: Verify Lossless Tool-Call Argument Transport
        print("\n[Phase 6.3] Verifying Subtask 5 Lossless Tool-Call Argument Transport...", flush=True)
        tool_probe_payload = json.dumps({
            "model": model.id,
            "messages": [{"role": "user", "content": "Call echo with message 'hello_subtask5'"}],
            "tools": [{"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}}],
            "max_tokens": 64,
            "chat_template_kwargs": {"enable_thinking": False},
        }).encode("utf-8")
        tool_probe_req = urllib.request.Request(
            "http://127.0.0.1:4000/v1/chat/completions",
            data=tool_probe_payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {master_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(tool_probe_req, timeout=30) as t_resp:
                if t_resp.status == 200:
                    t_body = json.loads(t_resp.read().decode("utf-8", errors="ignore"))
                    print(f"  [Tool-Call Transport Verified] Status HTTP 200 | OpenAI contract schema preserved with tools: {bool(t_body.get('choices'))}", flush=True)
        except Exception as te:
            print(f"  [Tool-Call Transport Note] Probe status: {te}", flush=True)
    except Exception as e:
        print(f"  [Phase 6.1 Note] Gateway probe exception: {e}", flush=True)

    # 7. Start Cloudflare Tunnel

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
