"""
Interactive Live Chat Session connected directly to the Kaggle GPU Host.

Streams responses token-by-token from the live llama-server endpoint.
"""
from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
from typing import List, Dict

from kaggle.config.api_end_point_config.tunnel import get_tunnel_endpoint, get_worker_auth_key, resolve_live_endpoint


def run_live_chat() -> None:
    print("\nConnecting to Kaggle GPU worker...")
    online, url, msg, active_model = resolve_live_endpoint()

    if not online:
        print(f"\n[ERROR] Cannot start chat: {msg}")
        return

    model_name = active_model or "Ornith Qwen3.8-27B Q4_K_M (Vision)"
    auth_key = get_worker_auth_key()
    api_url = f"{url}/v1/chat/completions"

    print("\n" + "=" * 60)
    print(f"       LIVE CHAT: {model_name}")
    print("=" * 60)
    print(f"  • Endpoint : {url}")
    print(f"  • Host     : Kaggle Dual Tesla T4")
    print("  Type 'exit' or 'quit' to return to menu.")
    print("=" * 60 + "\n")

    history: List[Dict[str, str]] = [
        {"role": "system", "content": "You are a helpful, fast, and intelligent AI coding assistant powered by Kaggle GPU."}
    ]

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat...")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("Ending chat session.")
            break

        history.append({"role": "user", "content": user_input})

        payload = json.dumps({
            "model": model_name,
            "messages": history,
            "temperature": 0.7,
            "max_tokens": 2048,
            "stream": True,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_key}",
            "User-Agent": "AssistantClient/1.0",
        }

        req = urllib.request.Request(api_url, data=payload, headers=headers)

        print("\nAssistant > ", end="", flush=True)
        assistant_reply = []

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str.startswith("data: "):
                        data_part = line_str[6:]
                        if data_part == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_part)
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                print(content, end="", flush=True)
                                assistant_reply.append(content)
                        except json.JSONDecodeError:
                            pass
            print()
            full_reply = "".join(assistant_reply)
            history.append({"role": "assistant", "content": full_reply})
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"\n[HTTP Error {e.code}]: {err_msg}")
        except Exception as e:
            print(f"\n[Error communicating with Kaggle GPU]: {e}")


if __name__ == "__main__":
    run_live_chat()
