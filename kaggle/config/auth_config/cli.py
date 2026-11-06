"""CLI interface for managing and switching Kaggle accounts."""
from __future__ import annotations

import sys
from typing import Optional

from kaggle.config.auth_config.manager import (
    apply_active_credentials,
    load_accounts,
    verify_account,
)


def run_account_switcher() -> None:
    """Interactive account switcher."""
    active, accounts = load_accounts()

    print("\n" + "=" * 50)
    print("        Kaggle Multi-Account Switcher            ")
    print("=" * 50)
    active_str = active.display_name if active else "[None Set]"
    print(f"Current Active Account: {active_str}\n")

    if not accounts:
        print("No multi-accounts registered in .env.")
        return

    print("Configured Accounts:")
    for i, acc in enumerate(accounts, start=1):
        is_active = " (Active)" if (active and active.username == acc.username) else ""
        print(f"  [{i}] {acc.display_name}{is_active}")
    print(f"  [v] Verify current connection")
    print(f"  [q] Back / Exit")

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

            print("Verifying new active credentials with Kaggle API...")
            ok, msg = verify_account(selected.username)
            status = "SUCCESS" if ok else "FAILED"
            print(f"[{status}] {msg}")
        else:
            print("Invalid selection.")
    except ValueError:
        print("Invalid input.")


if __name__ == "__main__":
    run_account_switcher()
