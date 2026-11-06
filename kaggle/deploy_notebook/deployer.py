"""
Kaggle Kernel Bundler & Deployer.

Compiles our modular architecture:
  1. profile.py          (ModelProfile class definition)
  2. environment.py      (Phase 1: Strict 2x T4 GPU topology & CUDA check)
  3. model_resolver.py   (Phase 2: Deterministic model weights search & vision gating)
  4. gpu_allocator.py    (Phase 3: Accurate KV cache & VRAM admission check with 1-2GB headroom)
  5. server_runner.py    (Phase 4: Single authoritative llama-server command & readiness verification)
  6. auth_gateway.py     (Phase 5: Secure Bearer Auth Gateway between Cloudflare and llama-server)
into the self-contained worker.py pushed to Kaggle.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from kaggle.config.auth_config.manager import load_accounts
from kaggle.config.model_config.catalog import ModelCatalog
from kaggle.config.model_config.profile import ModelProfile
from kaggle.deploy_notebook.metadata import generate_kernel_metadata

MODULE_FILES = [
    (Path(__file__).resolve().parents[1] / "config" / "model_config" / "manifest.py", "manifest.py"),
    (Path(__file__).resolve().parents[1] / "config" / "model_config" / "profile.py", "profile.py"),
    (Path(__file__).resolve().parent / "environment.py", "environment.py"),
    (Path(__file__).resolve().parent / "model_resolver.py", "model_resolver.py"),
    (Path(__file__).resolve().parent / "gpu_allocator.py", "gpu_allocator.py"),
    (Path(__file__).resolve().parent / "latency_telemetry.py", "latency_telemetry.py"),
    (Path(__file__).resolve().parent / "context_admission.py", "context_admission.py"),
    (Path(__file__).resolve().parent / "runtime_policy.py", "runtime_policy.py"),
    (Path(__file__).resolve().parent / "server_runner.py", "server_runner.py"),
    (Path(__file__).resolve().parent / "auth_gateway.py", "auth_gateway.py"),
]


def bundle_worker_script(target_model: ModelProfile, deploy_dir: Path) -> Path:
    """
    Compile our sub-modules directly into a clean, standalone worker.py
    with active model configuration injected.
    """
    worker_path = deploy_dir / "worker.py"
    
    model_dict = dict(target_model.__dict__)
    if model_dict.get("manifest") and hasattr(model_dict["manifest"], "to_dict"):
        model_dict["manifest"] = model_dict["manifest"].to_dict()

    sections = [
        '"""Auto-generated Kaggle worker entry compiled from deploy_notebook sub-modules."""',
        "from __future__ import annotations",
        "import os, sys, time, shutil, subprocess, re, urllib.request, urllib.error, threading, http.server, json",
        "from dataclasses import dataclass, field",
        "from enum import Enum",
        "from pathlib import Path",
        "from typing import Any, Dict, List, Optional, Tuple",
        "",
        "# Ensure immediate line-buffered stdout streaming on Kaggle",
        "if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(line_buffering=True)",
        "if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(line_buffering=True)",
        "",
        f"# Target Model ID deployed: {target_model.id}",
        f"DEPLOYED_MODEL_JSON = '''{json.dumps(model_dict)}'''",
    ]

    for fpath, label in MODULE_FILES:
        if fpath.exists():
            code = fpath.read_text(encoding="utf-8")
            clean_lines = []
            skip_multiline_import = False
            for line in code.splitlines():
                if skip_multiline_import:
                    if ")" in line:
                        skip_multiline_import = False
                    continue
                if line.startswith("from kaggle.") or line.startswith("import kaggle.") or "from __future__ import annotations" in line:
                    if "(" in line and ")" not in line:
                        skip_multiline_import = True
                    continue
                clean_lines.append(line)
            sections.append(f"\n# ==================== Module: {label} ====================")
            sections.append("\n".join(clean_lines))

    # Add the robust main execution flow
    sections.append("""
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
    print("\\n[Phase 1] Checking GPU topology (Enforcing 2x Tesla T4)...", flush=True)
    ok, gpus, msg = check_gpu_topology(required_count=2)
    print(f"[{'OK' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok:
        print("[CRITICAL ERROR] Hardware verification failed. Aborting startup.", flush=True)
        sys.exit(1)

    # 2. Deterministic Model Asset Resolution
    print(f"\\n[Phase 2] Deterministically resolving weights for '{model.id}'...", flush=True)
    ok, model_file, msg = find_model_weights(model, search_root="/kaggle/input")
    print(f"[{'OK' if ok else 'FAIL'}] {msg}", flush=True)
    if not ok or not model_file:
        print(f"[CRITICAL ERROR] Could not find target model '{model.filename}' in /kaggle/input. Aborting.", flush=True)
        sys.exit(1)

    model_size_gb = model_file.stat().st_size / (1024 ** 3)

    # Inspect dynamic GGUF binary headers & caution against auxiliary tensors
    print(f"\\n[Phase 2.1] Inspecting GGUF binary metadata for '{model_file.name}'...", flush=True)
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
    print("\\n[Phase 3] Running Resource Admission Check...", flush=True)
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
    print("\\n[Phase 4] Staging runtime and companion shared libraries (.so)...", flush=True)
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
    print("\\n[Phase 5] Verifying model initialization, process liveness, and VRAM readiness...", flush=True)
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
    print("\\n[VRAM Verification] Querying actual physical GPU utilization...", flush=True)
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
    print("\\n[Host Memory Verification] Checking system RAM, page cache, and swap pressure...", flush=True)
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
        print("\\n[Phase 5.1] Testing multimodal vision encoder & projector end-to-end...", flush=True)
        v_ok, v_res = verify_vision_pipeline(port=8080, model_alias=model.id)
        print(f"[{'OK' if v_ok else 'WARN'}] {v_res}", flush=True)

    # 5.2 Chat Template Role & Tool Schema Formatting Verification
    print("\\n[Phase 5.2] Verifying Chat Template message roles (system/user/assistant) & tool schemas...", flush=True)
    t_ok, t_msg, t_roles = verify_chat_roles_and_tool_schema(port=8080, model_alias=model.id)
    print(f"[{'OK' if t_ok else 'WARN'}] {t_msg}", flush=True)
    print(f"  Role Verification Details: {t_roles}", flush=True)

    # 6. Start Authenticated Gateway (Port 4000) & Cloudflare Tunnel
    # Public Cloudflare -> Gateway (Port 4000 with Bearer Auth) -> llama-server (Port 8080)
    print("\\n[Phase 6] Starting Authenticated Gateway (Port 4000) & Cloudflare Tunnel...", flush=True)
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
            match = re.search(r"https://[a-zA-Z0-9-]+\\.[t]rycloudflare\\.[c]om", line)
            if match:
                url = match.group(0)
                print("\\n" + "=" * 60, flush=True)
                print(f"AUTHENTICATED TUNNEL LIVE: {url}", flush=True)
                print(f"REMOTE_TUNNEL_URL={url}", flush=True)
                print("=" * 60 + "\\n", flush=True)
                break
    threading.Thread(target=stream_tunnel, daemon=True).start()

    # Keep alive while server_proc is running
    server_proc.wait()
""")

    worker_path.write_text("\n".join(sections), encoding="utf-8")
    return worker_path


def push_worker_to_kaggle(model_id: str = "qwen38-27b") -> Tuple[bool, str]:
    """Compile sub-modules into worker.py, create metadata, and push to Kaggle."""
    model = ModelCatalog.get_by_id(model_id) or ModelCatalog.get_default()
    deploy_dir = Path(__file__).resolve().parent

    print(f"\n[Deployer] Compiling sub-modules into standalone worker.py for {model.id}...")
    bundle_worker_script(model, deploy_dir)

    generate_kernel_metadata(model, code_file="worker.py", output_dir=deploy_dir)
    print(f"[Deployer] kernel-metadata.json configured for: {model.display_name}")

    active, _ = load_accounts()
    active_user = active.username if active else "alluriakshay"
    print(f"[Deployer] Pushing kernel to Kaggle as {active_user}...")

    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(deploy_dir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if proc.returncode == 0:
            msg = proc.stdout.strip()
            print(f"[SUCCESS] {msg}")
            return True, msg
        err = proc.stderr.strip() or proc.stdout.strip()
        print(f"[FAILED] {err}")
        return False, err
    except Exception as e:
        return False, f"Deployment failed: {str(e)}"


def run_deployment_wizard() -> None:
    models = ModelCatalog.list_models()

    print("\n" + "=" * 55)
    print("         Deploy Model Worker to Kaggle GPU       ")
    print("=" * 55)
    for idx, m in enumerate(models, start=1):
        print(f"  [{idx}] {m.display_name}")
    print("  [q] Cancel\n")

    choice = input("Select model to deploy [1-3]: ").strip()
    if choice.lower() in ("q", ""):
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            selected = models[idx]
            ok, msg = push_worker_to_kaggle(selected.id)
            if ok:
                print("\nKernel successfully queued on Kaggle GPU!")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


if __name__ == "__main__":
    run_deployment_wizard()
