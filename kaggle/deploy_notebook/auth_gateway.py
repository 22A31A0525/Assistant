"""
Lightweight Gateway Proxy & Auth Boundary.

Runs on Kaggle between Cloudflare Tunnel and llama-server:
  Public Cloudflare (Port 4000) -> [Auth Gateway] -> llama-server (Port 8080)
- Rejects unauthenticated requests with 401 Unauthorized
- Enforces Bearer token verification
- Forwards authorized OpenAI-compatible completions to llama-server
"""
from __future__ import annotations

import http.server
import json
import os
import urllib.request
import urllib.error
from typing import Optional


class AuthenticatedGatewayHandler(http.server.BaseHTTPRequestHandler):
    """Simple, zero-dependency HTTP reverse proxy with Bearer auth and context admission guard."""

    master_key: str = "sk-ant-api03-fakekey1234"
    upstream_port: int = 8080
    max_context_tokens: int = 65536
    # Reserve tokens for output generation (2048) + safety buffer -> max prompt tokens ~62,000
    max_prompt_tokens: int = 62000

    def log_message(self, format, *args):
        # Suppress noisy standard request logging
        pass

    def check_auth(self) -> bool:
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.master_key}"
        if auth_header.strip() == expected:
            return True

        self.send_response(401)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        err = json.dumps({
            "error": {
                "code": "UNAUTHORIZED",
                "message": "Invalid or missing Bearer token.",
                "layer": "auth_gateway",
            }
        }).encode("utf-8")
        self.wfile.write(err)
        return False

    def validate_context_limits(self, body: Optional[bytes]) -> Tuple[bool, Optional[str]]:
        """
        Subtask 3 Hard Safeguard: Prevent requests exceeding maximum context window (65,536).
        Calculates approximate prompt tokens and verifies against max_prompt_tokens ceiling.
        """
        if not body or not self.path.startswith("/v1/chat/completions"):
            return True, None

        try:
            payload = json.loads(body.decode("utf-8", errors="ignore"))
            messages = payload.get("messages", [])
            max_tokens = payload.get("max_tokens", 2048) or 2048
            
            # Approximate token length: text characters / 3.8 + image base64 patch allowance
            total_chars = 0
            image_token_allowance = 0
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                total_chars += len(part.get("text", ""))
                            elif part.get("type") == "image_url":
                                # Qwen-VL allocates standard patch grid (~1024 tokens minimum)
                                image_token_allowance += 1024

            estimated_prompt_tokens = int(total_chars / 3.8) + image_token_allowance
            ceiling = self.max_context_tokens - max_tokens

            if estimated_prompt_tokens > ceiling:
                return False, (
                    f"Request prompt size (~{estimated_prompt_tokens:,} tokens) plus requested generation "
                    f"({max_tokens:,} tokens) exceeds maximum context capacity ({self.max_context_tokens:,}). "
                    f"Safe prompt limit is {ceiling:,} tokens."
                )
            return True, None
        except Exception:
            # If body is not standard JSON, allow upstream to parse or reject
            return True, None

    def do_GET(self):
        if not self.check_auth():
            return
        self.proxy_request("GET")

    def do_POST(self):
        if not self.check_auth():
            return
        if self.path == "/v1/shutdown":
            self.do_SHUTDOWN()
            return
        self.proxy_request("POST")

    def proxy_request(self, method: str):
        target_url = f"http://127.0.0.1:{self.upstream_port}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Hard Context Limit Check (Subtask 3 Safeguard)
        valid_context, err_msg = self.validate_context_limits(body)
        if not valid_context:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_resp = json.dumps({
                "error": {
                    "code": "CONTEXT_LENGTH_EXCEEDED",
                    "message": err_msg,
                    "layer": "auth_gateway",
                }
            }).encode("utf-8")
            self.wfile.write(err_resp)
            return

        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "authorization")}

        try:
            upstream_req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
            with urllib.request.urlopen(upstream_req, timeout=180) as resp:
                is_event_stream = "text/event-stream" in resp.headers.get("Content-Type", "")

                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)

                if is_event_stream:
                    # True SSE HTTP chunked streaming (vital for real-time Codex token streaming)
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        chunk = resp.read(1024)
                        if not chunk:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                            break
                        self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii") + chunk + b"\r\n")
                        self.wfile.flush()
                else:
                    # Standard non-streaming response
                    resp_data = resp.read()
                    self.send_header("Content-Length", str(len(resp_data)))
                    self.end_headers()
                    self.wfile.write(resp_data)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length"):
                    self.send_header(k, v)
            e_data = e.read()
            self.send_header("Content-Length", str(len(e_data)))
            self.end_headers()
            self.wfile.write(e_data)
        except Exception as e:
            # When llama-server backend is down, dead, or restarting -> Explicit 503 Service Unavailable
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "5")
            self.end_headers()
            err = json.dumps({
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": f"Inference engine unavailable or initializing: {str(e)}",
                    "layer": "auth_gateway"
                }
            }).encode("utf-8")
            self.wfile.write(err)

    def do_SHUTDOWN(self):
        """Emergency remote shutdown signal over authenticated gateway."""
        if not self.check_auth():
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "terminating", "message": "Worker shutdown signal accepted."}).encode("utf-8"))
        
        def _terminate():
            import time, os
            time.sleep(1)
            print("[SHUTDOWN] Remote shutdown requested via gateway. Releasing GPU container...", flush=True)
            os._exit(0)

        import threading
        threading.Thread(target=_terminate, daemon=True).start()


def start_gateway(port: int = 4000, upstream_port: int = 8080, master_key: str = ""):
    """Start the lightweight auth gateway."""
    key = master_key or os.environ.get("LITELLM_MASTER_KEY", "sk-ant-api03-fakekey1234")
    AuthenticatedGatewayHandler.master_key = key
    AuthenticatedGatewayHandler.upstream_port = upstream_port

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), AuthenticatedGatewayHandler)
    print(f"[Gateway] Secure Auth Gateway running on port {port} -> forwarding to llama-server:{upstream_port}", flush=True)
    server.serve_forever()
