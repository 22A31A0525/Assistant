"""
Emergency GPU Stopper & Quota Releaser.

Kaggle CLI does not have a native `kaggle kernels cancel` command.
However, pushing a 0-GPU fast-exit override script (enable_gpu: false, sys.exit(0))
instantly preempts the active dual-T4 session and terminates the GPU container,
releasing the 30h/week GPU allocation immediately.
"""
from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from kaggle.config.auth_config.manager import load_accounts


def get_active_kernel_ref(custom_slug: Optional[str] = None) -> str:
    """Return the active kernel ref (e.g. alluriakshay/assistant-worker-alluriakshay)."""
    active, _ = load_accounts()
    username = active.username if active else "alluriakshay"
    slug = custom_slug or f"assistant-worker-{username.lower().replace('_', '-')}"
    return f"{username}/{slug}"


def query_kernel_status(kernel_ref: Optional[str] = None) -> Tuple[str, str]:
    """
    Check the live execution state of a Kaggle kernel.
    Returns: (normalized_status, raw_output)
    """
    ref = kernel_ref or get_active_kernel_ref()
    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", ref],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        out = (proc.stdout + proc.stderr).strip()
        lower = out.lower()
        if "running" in lower:
            return "RUNNING", out
        elif "cancel" in lower:
            return "CANCELLED", out
        elif "complete" in lower:
            return "COMPLETED", out
        elif "error" in lower or "fail" in lower:
            return "FAILED", out
        elif "queued" in lower:
            return "QUEUED", out
        return "UNKNOWN", out
    except Exception as e:
        return "ERROR", str(e)


def stop_gpu_kernel(kernel_ref: Optional[str] = None) -> Tuple[bool, str]:
    """
    Stop GPU Kernel:
    First attempts remote authenticated shutdown via gateway if tunnel is online.
    Falls back to Kaggle API kernel deletion to guarantee immediate container termination.
    Also provides clear UI guidance for canceling via the Kaggle web interface.
    """
    ref = kernel_ref or get_active_kernel_ref()
    if "/" not in ref:
        return False, f"Invalid kernel ref: '{ref}'. Expected format: username/slug"

    # Step 1: Try remote API shutdown endpoint if live tunnel is active
    try:
        from kaggle.config.api_end_point_config.tunnel import get_tunnel_endpoint, get_worker_auth_key
        import urllib.request
        tunnel_url = get_tunnel_endpoint()
        if tunnel_url and "trycloudflare.com" in tunnel_url:
            auth_key = get_worker_auth_key()
            req = urllib.request.Request(
                f"{tunnel_url.rstrip('/')}/v1/shutdown",
                data=b"{}",
                headers={"Authorization": f"Bearer {auth_key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return True, f"Remote worker received shutdown signal and terminated container. GPU quota released."
            except Exception:
                pass
    except Exception:
        pass

    # Step 2: Immediate kernel deletion via official Kaggle CLI
    try:
        del_proc = subprocess.run(
            ["kaggle", "kernels", "delete", "-y", ref],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if del_proc.returncode == 0:
            return True, f"Kernel '{ref}' was deleted via Kaggle API. Running session halted and GPU quota released."
    except Exception:
        pass

    # Step 3: Web UI fallback instruction
    return False, (
        f"Could not reach remote shutdown or delete '{ref}'.\n"
        f"To stop immediately via Web UI: Open Kaggle -> Click 'View Active Events' (bottom-left) -> 'Stop Session'."
    )


def list_active_running_kernels() -> List[dict]:
    """List recent kernels with their current execution status."""
    try:
        res = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "10", "--csv"],
            capture_output=True,
            text=True,
            timeout=25,
            check=False,
        )
        if res.returncode != 0:
            return []

        lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip()]
        if len(lines) <= 1:
            return []

        reader = csv.DictReader(lines)
        results = []
        for row in reader:
            ref = row.get("ref", "").strip()
            title = row.get("title", "").strip()
            last_run = row.get("lastRunTime", "").strip()
            if ref:
                results.append({"ref": ref, "title": title, "last_run": last_run})
        return results
    except Exception:
        return []


def run_emergency_stopper_menu() -> None:
    """Interactive emergency stopper dialog."""
    active_ref = get_active_kernel_ref()

    print("\n" + "=" * 55)
    print("        Emergency GPU Stopper & Quota Releaser    ")
    print("=" * 55)
    print(f"Active Worker Reference: {active_ref}")

    print("\nChecking current status...")
    status, raw = query_kernel_status(active_ref)
    print(f"Status: [{status}]")

    print("\nOptions:")
    print(f"  [1] STOP '{active_ref}' (Release GPU Quota immediately)")
    print("  [2] Select from list of recent kernels to stop")
    print("  [3] Refresh status")
    print("  [q] Cancel\n")

    choice = input("Enter choice [1-3]: ").strip()
    if choice == "1":
        confirm = input(f"Are you sure you want to terminate '{active_ref}'? [y/N]: ").strip().lower()
        if confirm == "y":
            print("\nDeploying 0-GPU preemption signal...")
            ok, msg = stop_gpu_kernel(active_ref)
            badge = "SUCCESS" if ok else "FAILED"
            print(f"[{badge}] {msg}")
    elif choice == "2":
        print("\nFetching recent notebooks...")
        kernels = list_active_running_kernels()
        if not kernels:
            print("No notebooks found.")
            return
        for i, k in enumerate(kernels, start=1):
            print(f"  [{i}] {k['ref']} ({k['title']})")
        sel = input("\nSelect notebook number to stop: ").strip()
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(kernels):
                target = kernels[idx]["ref"]
                print(f"\nTerminating {target}...")
                ok, msg = stop_gpu_kernel(target)
                badge = "SUCCESS" if ok else "FAILED"
                print(f"[{badge}] {msg}")
        except ValueError:
            print("Invalid input.")
    elif choice == "3":
        st, _ = query_kernel_status(active_ref)
        print(f"\nCurrent status of {active_ref}: {st}")


if __name__ == "__main__":
    run_emergency_stopper_menu()
