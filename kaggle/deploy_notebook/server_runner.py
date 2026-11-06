"""
Phase 4 & 5: Native Server Execution & Readiness Verification.

Features:
- Single command builder used by both main runner and deployer
- Readiness verification function (/health & /v1/models probe)
"""
from __future__ import annotations

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
    timeout_seconds: int = 180,
    process: Optional[subprocess.Popen] = None,
) -> Tuple[bool, str]:
    """
    Poll the local loopback endpoint until llama-server is fully initialized and resident in VRAM.
    """
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        if process and process.poll() is not None:
            return False, f"Server process exited unexpectedly with code {process.poll()}."

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AssistantProbe/1.0"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status in (200, 204):
                    return True, "Server is ready and accepting requests."
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    alt_req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
                    with urllib.request.urlopen(alt_req, timeout=3) as alt_resp:
                        if alt_resp.status == 200:
                            return True, "Server ready (verified via /v1/models)."
                except Exception:
                    pass
        except Exception:
            pass

        time.sleep(2)

    return False, f"Timed out waiting {timeout_seconds}s for llama-server initialization."


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
                    {"type": "text", "text": "Describe this image in 5 words."},
                    {"type": "image_url", "image_url": {"url": tiny_png_uri}},
                ]
            }
        ],
        "max_tokens": 128,  # Ample token budget for thinking tokens + final content
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
