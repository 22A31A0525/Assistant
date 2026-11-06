"""
Interactive Live Chat Session connected directly to the Kaggle GPU Host.

Streams responses token-by-token from the live llama-server endpoint.
"""
from __future__ import annotations

import json
import sys
import time
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

    # Subtask 3: Context monitoring & 85% capacity threshold warning
    MAX_CONTEXT_TOKENS = 65536
    WARN_THRESHOLD_TOKENS = int(MAX_CONTEXT_TOKENS * 0.85)  # 55,705 tokens

    def estimate_message_tokens(msgs: List[Dict[str, str]]) -> int:
        # Approximate 1 token ~ 3.8 characters for code/text mix
        total_chars = sum(len(m.get("content", "")) for m in msgs)
        return int(total_chars / 3.8)

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

        current_tokens_est = estimate_message_tokens(history)
        if current_tokens_est >= WARN_THRESHOLD_TOKENS:
            print(f"\n[Context Warning] Active conversation length (~{current_tokens_est:,} tokens) has reached 85% of 64K capacity ({MAX_CONTEXT_TOKENS:,}). Consider summarizing state.", flush=True)

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
        ttft_recorded = False
        start_time = time.monotonic()
        ttft_time = 0.0
        server_timings = None

        try:
            with urllib.request.urlopen(req, timeout=90) as response:
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
                            # Track server-reported timings if returned in the final chunk
                            if "timings" in chunk:
                                server_timings = chunk["timings"]

                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if not ttft_recorded:
                                    ttft_time = time.monotonic() - start_time
                                    ttft_recorded = True
                                print(content, end="", flush=True)
                                assistant_reply.append(content)
                        except json.JSONDecodeError:
                            pass
            total_time = time.monotonic() - start_time
            print()
            full_reply = "".join(assistant_reply)
            history.append({"role": "assistant", "content": full_reply})

            # Subtask 4: Latency & Telemetry reporting
            if assistant_reply:
                token_count = len(full_reply.split()) * 1.3  # heuristic or use server count if available
                gen_speed = (len(assistant_reply) / (total_time - ttft_time)) if (total_time - ttft_time) > 0.05 else 0.0
                print(f"\n[Telemetry] TTFT: {ttft_time:.2f}s | Total: {total_time:.2f}s", end="")
                if server_timings:
                    p_eval = server_timings.get("prompt_per_second", 0)
                    g_eval = server_timings.get("predicted_per_second", 0)
                    print(f" | Prompt: {p_eval:.1f} t/s | Decode: {g_eval:.1f} t/s", end="")
                elif gen_speed > 0:
                    print(f" | Stream decode rate: ~{gen_speed:.1f} chunks/s", end="")
                print()
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="ignore")
            print(f"\n[HTTP Error {e.code}]: {err_msg}")
        except Exception as e:
            print(f"\n[Error communicating with Kaggle GPU]: {e}")


if __name__ == "__main__":
    run_live_chat()
