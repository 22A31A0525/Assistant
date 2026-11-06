"""
Optimized Kaggle Authentication Manager.

Handles:
- Loading multi-account configurations from .env
- Synchronizing active credentials with ~/.kaggle/kaggle.json and access_token
- Switching active accounts safely
- Verifying credentials via Kaggle API
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from kaggle.config.auth_config.models import KaggleAccount


def find_env_file(start_path: Optional[Path] = None) -> Optional[Path]:
    """Search for the root .env file climbing up from start_path or current directory."""
    current = (start_path or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def get_kaggle_home_dir() -> Path:
    """Return the platform-specific ~/.kaggle directory path."""
    return Path.home() / ".kaggle"


def clear_oauth_cache(kaggle_dir: Optional[Path] = None) -> None:
    """Clear cached OAuth tokens to ensure the Kaggle CLI switches accounts cleanly."""
    target_dir = kaggle_dir or get_kaggle_home_dir()
    for filename in ("access_token", "credentials.json"):
        token_file = target_dir / filename
        if token_file.exists():
            try:
                token_file.unlink()
            except OSError:
                pass


def load_accounts(env_path: Optional[Path] = None) -> Tuple[Optional[KaggleAccount], List[KaggleAccount]]:
    """
    Parse .env to load active and registered multi-accounts.

    Returns:
        (active_account, list_of_saved_accounts)
    """
    resolved_path = env_path or find_env_file()
    if not resolved_path or not resolved_path.exists():
        return None, []

    content = resolved_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    active_user = ""
    active_key = ""
    active_token = ""
    accounts_by_index: Dict[int, Dict[str, str]] = {}

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("KAGGLE_USERNAME="):
            active_user = line.split("=", 1)[1].strip().strip("'\"")
        elif line.startswith("KAGGLE_KEY="):
            active_key = line.split("=", 1)[1].strip().strip("'\"")
        elif line.startswith("KAGGLE_API_TOKEN="):
            active_token = line.split("=", 1)[1].strip().strip("'\"")

        user_match = re.match(r"^KAGGLE_ACCOUNT_(\d+)_USERNAME=(.*)", line)
        if user_match:
            idx = int(user_match.group(1))
            accounts_by_index.setdefault(idx, {})["username"] = user_match.group(2).strip().strip("'\"")

        key_match = re.match(r"^KAGGLE_ACCOUNT_(\d+)_KEY=(.*)", line)
        if key_match:
            idx = int(key_match.group(1))
            accounts_by_index.setdefault(idx, {})["key"] = key_match.group(2).strip().strip("'\"")

        token_match = re.match(r"^KAGGLE_ACCOUNT_(\d+)_API_TOKEN=(.*)", line)
        if token_match:
            idx = int(token_match.group(1))
            accounts_by_index.setdefault(idx, {})["token"] = token_match.group(2).strip().strip("'\"")

    saved_accounts: List[KaggleAccount] = []
    for idx in sorted(accounts_by_index.keys()):
        data = accounts_by_index[idx]
        user = data.get("username", "")
        key = data.get("key", "")
        if user and key:
            saved_accounts.append(
                KaggleAccount(
                    username=user,
                    key=key,
                    token=data.get("token", ""),
                    index=idx,
                    name=f"Account #{idx} ({user})",
                )
            )

    active_account: Optional[KaggleAccount] = None
    if active_user and active_key:
        active_account = KaggleAccount(
            username=active_user,
            key=active_key,
            token=active_token,
            index=0,
            name=f"Active ({active_user})",
        )

    return active_account, saved_accounts


def apply_active_credentials(
    account: KaggleAccount,
    env_path: Optional[Path] = None,
    kaggle_dir: Optional[Path] = None,
) -> Path:
    """
    Synchronize the chosen account across ~/.kaggle/kaggle.json and the root .env file.
    """
    target_kaggle_dir = kaggle_dir or get_kaggle_home_dir()
    target_kaggle_dir.mkdir(parents=True, exist_ok=True)
    clear_oauth_cache(target_kaggle_dir)

    # 1. Update ~/.kaggle/kaggle.json
    json_path = target_kaggle_dir / "kaggle.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"username": account.username, "key": account.key}, f, indent=2)

    try:
        os.chmod(json_path, 0o600)
    except OSError:
        pass

    # 2. Write OAuth token if available or key is a KGAT token
    token = account.token or (account.key if account.key.startswith("KGAT_") else "")
    if token:
        token_path = target_kaggle_dir / "access_token"
        token_path.write_text(token.strip(), encoding="utf-8")

    # 3. Update root .env active keys
    resolved_env = env_path or find_env_file()
    if resolved_env and resolved_env.exists():
        content = resolved_env.read_text(encoding="utf-8")
        updated_lines: List[str] = []
        user_updated = False
        key_updated = False
        token_updated = False

        for line in content.splitlines():
            if line.startswith("KAGGLE_USERNAME="):
                updated_lines.append(f"KAGGLE_USERNAME={account.username}")
                user_updated = True
            elif line.startswith("KAGGLE_KEY="):
                updated_lines.append(f"KAGGLE_KEY={account.key}")
                key_updated = True
            elif line.startswith("KAGGLE_API_TOKEN="):
                updated_lines.append(f"KAGGLE_API_TOKEN={account.token}")
                token_updated = True
            else:
                updated_lines.append(line)

        if not user_updated:
            updated_lines.insert(0, f"KAGGLE_USERNAME={account.username}")
        if not key_updated:
            updated_lines.insert(1, f"KAGGLE_KEY={account.key}")
        if not token_updated:
            updated_lines.insert(2, f"KAGGLE_API_TOKEN={account.token}")

        resolved_env.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")

    return json_path


def verify_account(username: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate that the current active Kaggle credentials work against the Kaggle CLI API.
    """
    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if proc.returncode == 0:
            target = f"for '{username}'" if username else "credentials"
            return True, f"Kaggle API successfully authenticated {target}."
        err_msg = proc.stderr.strip() or proc.stdout.strip()
        return False, f"Kaggle API returned an error: {err_msg}"
    except FileNotFoundError:
        return False, "Kaggle CLI is not installed or not found in system PATH."
    except subprocess.TimeoutExpired:
        return False, "Kaggle API verification timed out after 15 seconds."
