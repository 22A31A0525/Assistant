"""
Cloudflare Tunnel Endpoint Manager.

Connects to the remote Kaggle GPU worker over Cloudflare Tunnel:
- Uses configured LITELLM_MASTER_KEY for authorized requests
- Targets /v1/models for accurate health qualification and model introspection
- Smart 2-step resolution: checks .env first, auto-pulls Kaggle logs if offline
"""
from __future__ import annotations

import os
import re
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

from kaggle.config.auth_config.manager import find_env_file, load_accounts


def get_worker_auth_key(env_path: Optional[Path] = None) -> str:
    """Retrieve worker auth master key from environment or .env."""
    key = os.environ.get("LITELLM_MASTER_KEY", "").strip()
    if key:
        return key

    env = env_path or find_env_file()
    if env and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("LITELLM_MASTER_KEY="):
                return line.split("=", 1)[1].strip().strip("'\"")

    return "sk-ant-api03-fakekey1234"


def fetch_kernel_logs(kernel_id: Optional[str] = None) -> str:
    """Fetch live stdout logs from the active Kaggle kernel."""
    target_id = kernel_id
    if not target_id:
        active, _ = load_accounts()
        if not active or not active.username:
            return ""
        # Match current unified naming scheme
        slug = f"assistant-worker-{active.username.lower().replace('_', '-')}"
        target_id = f"{active.username}/{slug}"

    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "logs", target_id],
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        return proc.stdout or proc.stderr or ""
    except Exception:
        return ""


def extract_tunnel_url_from_logs(logs_text: str) -> Optional[str]:
    """Find live .trycloudflare.com tunnel in Kaggle logs."""
    if not logs_text:
        return None

    anthropic_match = re.search(
        r"ANTHROPIC_BASE_URL=(https://[a-zA-Z0-9\-]+\.trycloudflare\.com)",
        logs_text,
    )
    if anthropic_match:
        return anthropic_match.group(1).rstrip("/")

    cf_matches = re.findall(
        r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com",
        logs_text,
    )
    if cf_matches:
        return cf_matches[-1].rstrip("/")

    return None


def sync_tunnel_to_env(discovered_url: str, env_path: Optional[Path] = None) -> None:
    """Save the newly discovered tunnel URL back to .env."""
    if not discovered_url:
        return

    env = env_path or find_env_file()
    if not env or not env.exists():
        return

    lines = env.read_text(encoding="utf-8").splitlines()
    updated = False
    new_lines = []

    for line in lines:
        if line.startswith("REMOTE_TUNNEL_URL="):
            new_lines.append(f"REMOTE_TUNNEL_URL={discovered_url}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"REMOTE_TUNNEL_URL={discovered_url}")

    env.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def ping_endpoint(endpoint: str, timeout: int = 6) -> Tuple[bool, str, Optional[str]]:
    """
    Ping endpoint using /v1/models with authorization.
    
    Returns:
        (is_online, message, active_model_name)
    """
    url = endpoint.rstrip("/")
    auth_key = get_worker_auth_key()
    headers = {
        "User-Agent": "Assistant/1.0",
        "Authorization": f"Bearer {auth_key}",
    }

    try:
        req = urllib.request.Request(f"{url}/v1/models", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                body = resp.read().decode("utf-8", errors="ignore")
                model_match = re.search(r'"id":\s*"([^"]+)"', body) or re.search(r'"name":\s*"([^"]+)"', body)
                model_name = model_match.group(1) if model_match else "Online Model"
                return True, f"Online ({resp.status}) at {url}", model_name
            return False, f"Unexpected response status {resp.status} from {url}", None
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code} from {url}", None
    except Exception as e:
        return False, f"Offline ({e})", None


def resolve_live_endpoint() -> Tuple[bool, str, str, Optional[str]]:
    """
    Smart 2-step resolution:
    1. Try existing URL from .env. If ping succeeds -> return immediately.
    2. If missing or ping fails -> pull Kaggle logs once.
       - If logs contain a new URL -> sync to .env and ping.
       - If logs have no live URL or kernel isn't running -> return error to user.

    Returns:
        (is_online, active_url, message, model_name)
    """
    env = find_env_file()
    saved_url = ""
    if env and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("REMOTE_TUNNEL_URL="):
                saved_url = line.split("=", 1)[1].strip().strip("'\"").rstrip("/")
                break

    # Step 1: Test saved URL from .env
    if saved_url:
        ok, msg, model_name = ping_endpoint(saved_url)
        if ok:
            return True, saved_url, f"Endpoint online: {saved_url}", model_name

    # Step 2: Saved URL missing or offline -> attempt single log pull from Kaggle
    logs = fetch_kernel_logs()
    discovered_url = extract_tunnel_url_from_logs(logs)

    if discovered_url and discovered_url != saved_url:
        sync_tunnel_to_env(discovered_url, env)
        ok, msg, model_name = ping_endpoint(discovered_url)
        if ok:
            return True, discovered_url, f"Discovered live tunnel from Kaggle logs and updated .env: {discovered_url}", model_name
        return False, discovered_url, f"Extracted tunnel {discovered_url} from logs, but ping failed: {msg}", None

    # Step 3: Kaggle logs did not provide a working URL
    if not saved_url and not discovered_url:
        return False, "", "No tunnel URL in .env and no active tunnel found in Kaggle kernel logs.", None

    return False, saved_url, f"Saved endpoint ({saved_url}) is offline and no newer tunnel was found in Kaggle logs.", None


def get_tunnel_endpoint() -> str:
    """Convenience getter returning currently configured URL or localhost fallback."""
    env = find_env_file()
    if env and env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("REMOTE_TUNNEL_URL="):
                url = line.split("=", 1)[1].strip().strip("'\"").rstrip("/")
                if url:
                    return url
    return "http://127.0.0.1:8080"
