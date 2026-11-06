"""
Phase 1: Environment Bootstrap, CUDA Runtime Wheels & Library Resolution.
Directly adopting proven library resolution & ldd auto-repair from native_runtime.py.
"""
from __future__ import annotations

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
