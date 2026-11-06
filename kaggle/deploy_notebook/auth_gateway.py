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
from typing import Optional, Tuple, Mapping, Any

from kaggle.deploy_notebook.context_admission import (
    validate_before_proxy,
    AdmissionError,
    ContextExceeded,
    CountingUnavailable,
)


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

    def query_actual_input_tokens(self, body: bytes) -> Optional[int]:
        """
        Query llama-server's native /v1/chat/completions/input_tokens (or /tokenize) endpoint.
        Returns exact model-tokenized integer count of input messages and tool schemas,
        or None if upstream counting is unreachable.
        """
        tokenize_url = f"http://127.0.0.1:{self.upstream_port}/v1/chat/completions/input_tokens"
        try:
            req = urllib.request.Request(
                tokenize_url,
                data=body,
                headers={"Content-Type": "application/json", "User-Agent": "AuthGateway/1.0"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8", errors="ignore"))
                    # llama-server returns {"input_tokens": N} or {"tokens": [...]}
                    if "input_tokens" in data and isinstance(data["input_tokens"], int):
                        return data["input_tokens"]
                    if "tokens" in data and isinstance(data["tokens"], list):
                        return len(data["tokens"])
        except Exception:
            pass
        return None

    def validate_context_limits(self, body: Optional[bytes]) -> Tuple[bool, int, str, Optional[str]]:
        """
        Subtask 4 Model-Aware Context Admission:
        Enforces: Input tokens + Reserved output tokens <= 65,536.
        Returns: (is_valid, http_status, error_code, error_message)
        """
        if not body or not self.path.startswith("/v1/chat/completions"):
            return True, 200, "", None

        try:
            payload = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception as e:
            return False, 400, "INVALID_REQUEST", f"Malformed JSON request: {str(e)}"

        counter_url = f"http://127.0.0.1:{self.upstream_port}/v1/chat/completions/input_tokens"
        try:
            validate_before_proxy(
                payload,
                counter=lambda p: count_using_local_llama_server(p, url=counter_url, timeout=5.0),
                context_limit=self.max_context_tokens,
                output_cap=2048,
            )
            return True, 200, "", None
        except ContextExceeded as e:
            return False, 400, "CONTEXT_LENGTH_EXCEEDED", str(e)
        except CountingUnavailable as e:
            return False, 503, "TOKEN_COUNT_UNAVAILABLE", str(e)
        except AdmissionError as e:
            return False, e.status_code, e.error_code, str(e)
        except Exception as e:
            return False, 503, "TOKEN_COUNT_UNAVAILABLE", f"Token count validation failed: {str(e)}"

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

        # Hard Context Limit Check (Subtask 4 Model-Aware Safeguard)
        valid_context, status_code, err_code, err_msg = self.validate_context_limits(body)
        if not valid_context:
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_resp = json.dumps({
                "error": {
                    "code": err_code,
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
            # Client disconnected early (e.g. BrokenPipeError / ConnectionResetError)
            if "Broken pipe" in str(e) or "Errno 32" in str(e) or isinstance(e, BrokenPipeError):
                # Suppress noisy traceback when client closes connection early
                return

            # When llama-server backend is down, dead, or restarting -> Explicit 503 Service Unavailable
            try:
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
            except Exception:
                pass

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
