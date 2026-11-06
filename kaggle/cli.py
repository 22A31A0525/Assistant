"""
Unified Kaggle Control Center CLI.

Run from workspace root:
    python -m kaggle.cli
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from kaggle.config.auth_config.manager import (
    apply_active_credentials,
    load_accounts,
    verify_account,
)
from kaggle.config.model_config.catalog import ModelCatalog


def print_banner() -> None:
    active, _ = load_accounts()
    active_user = active.username if active else "None"

    print("\n" + "=" * 60)
    print("           KAGGLE REMOTE GPU ASSISTANT HUB            ")
    print("=" * 60)
    print(f"  • Active Kaggle Account : {active_user}")
    print(f"  • Active Workspace      : {WORKSPACE_ROOT.name}")
    print("=" * 60)


def menu_account_switcher() -> None:
    """Multi-account switcher dialog."""
    active, accounts = load_accounts()

    print("\n" + "-" * 50)
    print("        Kaggle Multi-Account Switcher")
    print("-" * 50)
    active_str = active.display_name if active else "[None Set]"
    print(f"Current Active: {active_str}\n")

    if not accounts:
        print("No multi-accounts registered in .env.")
        return

    for i, acc in enumerate(accounts, start=1):
        is_active = " (Active)" if (active and active.username == acc.username) else ""
        print(f"  [{i}] {acc.display_name}{is_active}")
    print("  [v] Verify current connection")
    print("  [q] Back")

    choice = input("\nSelect account number or action: ").strip()
    if choice.lower() in ("q", ""):
        return
    if choice.lower() == "v":
        print("\nVerifying credentials...")
        ok, msg = verify_account(active.username if active else None)
        status = "SUCCESS" if ok else "FAILED"
        print(f"[{status}] {msg}")
        return

    try:
        idx = int(choice) - 1
        if 0 <= idx < len(accounts):
            selected = accounts[idx]
            print(f"\nSwitching active account to: {selected.username}...")
            apply_active_credentials(selected)
            print("Credentials synchronized to ~/.kaggle/kaggle.json and .env.")
            ok, msg = verify_account(selected.username)
            status = "SUCCESS" if ok else "FAILED"
            print(f"[{status}] {msg}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


def menu_model_catalog() -> None:
    """Model catalog viewer."""
    models = ModelCatalog.list_models()

    print("\n" + "-" * 55)
    print("           Remote GPU LLM Model Catalog")
    print("-" * 55)

    for idx, m in enumerate(models, start=1):
        vision_badge = " [IMAGE/VISION READY]" if m.has_vision else " [PURE TEXT/REASONING]"
        print(f"\n[{idx}] {m.display_name}")
        print(f"    • Capability  : {vision_badge}")
        print(f"    • Quantization: {m.quantization} | Context: {m.context_size:,} tokens")
        print(f"    • Weights GGUF: {m.filename}")
        if m.has_vision:
            print(f"    • Vision Proj : {m.mmproj_source} ({m.mmproj_filename})")
        print(f"    • Hardware    : {m.gpu_target} (Split: {m.split_mode})")


def main_menu() -> None:
    while True:
        print_banner()
        print("Control Center Operations:")
        print("  [1] Authentication & Multi-Account Switcher")
        print("  [2] Model Catalog (Ornith 35B / Qwen 3.6 / Qwen 3.8)")
        print("  [3] Test Kaggle API Connection")
        print("  [4] Kernel Deployment (Push to Kaggle GPU)")
        print("  [5] Kernel Execution Logs & Tunnel Status")
        print("  [6] Emergency GPU Stopper (Release Quota)")
        print("  [0] Exit\n")

        choice = input("Select operation [0-6]: ").strip()

        if choice == "0":
            print("\nExiting Kaggle Hub. Goodbye!")
            break
        elif choice == "1":
            menu_account_switcher()
        elif choice == "2":
            menu_model_catalog()
        elif choice == "3":
            print("\nPinging Kaggle API...")
            ok, msg = verify_account()
            status = "SUCCESS" if ok else "FAILED"
            print(f"[{status}] {msg}")
        elif choice in ("4", "5", "6"):
            print(f"\n[Info] Operation [{choice}] will be wired up as we build the deploy & tracker modules next!")
        else:
            print("\nInvalid choice. Please select from the menu.")


if __name__ == "__main__":
    main_menu()
