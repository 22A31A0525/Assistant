"""
Subtask 5 Production Gateway Proxy & Connection Boundary.

Runs on Kaggle between Cloudflare Tunnel and llama-server:
  Public Cloudflare (Port 4000) -> [Auth Gateway] -> llama-server (Port 8080)
Features:
- Bearer token verification on every request
- Bounded payload size enforcement (15 MB max)
- Model-aware context admission guard (65,536 limit)
- Atomic SingleSlotScheduler: depth-1 queue, 429 Too Many Requests & 503 retry guidance
- Transparent lossless SSE chunk forwarding preserving raw UTF-8, tool deltas, and [DONE]
- Active client disconnect detection with immediate slot release
- Request correlation tracing via X-Request-ID
"""
from __future__ import annotations

import http.server
import json
import os
import threading
import time
import urllib.request
import urllib.error
import uuid
from typing import Optional, Tuple, Mapping, Any

from kaggle.deploy_notebook.context_admission import (
    validate_before_proxy,
    AdmissionError,
    ContextExceeded,
    CountingUnavailable,
)


class SingleSlotScheduler:
    """
    Thread-safe, atomic single-slot request scheduler with bounded queue of depth 1.
    Strictly protects --parallel 1 inference slot from concurrent contention,
    race conditions, and OOM crashes.
    """
    def __init__(self, max_queue: int = 1, queue_timeout: float = 20.0):
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._slot_busy = False
        self._queue_count = 0
        self._max_queue = max_queue
        self._queue_timeout = queue_timeout

    def acquire(self, request_id: str) -> Tuple[bool, Optional[str]]:
        """
        Atomically request inference permit.
        Returns (acquired, failure_reason).
        """
        with self._lock:
            # Slot is immediately available
            if not self._slot_busy and self._queue_count == 0:
                self._slot_busy = True
                return True, None

            # Check queue capacity (max 1 waiting request)
            if self._queue_count >= self._max_queue:
                return False, "QUEUE_FULL"

            self._queue_count += 1
            deadline = time.monotonic() + self._queue_timeout

            while self._slot_busy:
                wait_time = deadline - time.monotonic()
                if wait_time <= 0:
                    self._queue_count -= 1
                    return False, "TIMEOUT_WAITING_FOR_SLOT"
                self._condition.wait(timeout=wait_time)

            self._queue_count -= 1
            self._slot_busy = True
            return True, None

    def release(self, request_id: str):
        """Release slot permit and notify any waiting queued caller."""
        with self._lock:
            self._slot_busy = False
            self._condition.notify()


GLOBAL_SCHEDULER = SingleSlotScheduler(max_queue=1, queue_timeout=20.0)


class AuthenticatedGatewayHandler(http.server.BaseHTTPRequestHandler):
    """
    Subtask 5 Production Gateway Handler:
    - Lossless SSE chunk forwarding (preserves raw UTF-8, tool calls, and [DONE])
    - Atomic single-slot concurrency scheduler with 429/503 retry guidance
    - Request-body limits (15 MB) and media access safety
    - Fail-closed model-aware context admission (65,536 limit)
    - Structured Request-ID tracing & telemetry
    """

    master_key: str = "sk-ant-api03-fakekey1234"
    upstream_port: int = 8080
    max_context_tokens: int = 65536
    max_body_bytes: int = 15 * 1024 * 1024  # 15 MB limit for multimodal images / long context

    ALLOWED_ROUTES = {
        "/health",
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions/input_tokens",
        "/v1/shutdown",
    }

    def log_message(self, format, *args):
        # Suppress noisy standard request logging
        pass

    def send_json_error(self, status_code: int, code: str, message: str, headers_extra: Optional[dict] = None):
        """Helper to send standardized OpenAI-compatible error payloads."""
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        if headers_extra:
            for k, v in headers_extra.items():
                self.send_header(k, str(v))
        self.end_headers()
        err_resp = json.dumps({
            "error": {
                "code": code,
                "message": message,
                "layer": "auth_gateway",
            }
        }).encode("utf-8")
        try:
            self.wfile.write(err_resp)
        except Exception:
            pass

    def check_auth(self) -> bool:
        auth_header = self.headers.get("Authorization", "")
        expected = f"Bearer {self.master_key}"
        if auth_header.strip() == expected:
            return True

        self.send_json_error(401, "UNAUTHORIZED", "Invalid or missing Bearer token.")
        return False

    def validate_request_path_and_body(self) -> Tuple[bool, Optional[bytes]]:
        """Validate allowed paths, method, and enforce request body byte limits."""
        if self.path not in self.ALLOWED_ROUTES and not self.path.startswith("/v1/"):
            self.send_json_error(404, "NOT_FOUND", f"Route '{self.path}' is not supported.")
            return False, None

        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > self.max_body_bytes:
            self.send_json_error(
                413, "PAYLOAD_TOO_LARGE",
                f"Request body ({content_length:,} bytes) exceeds gateway maximum ({self.max_body_bytes:,} bytes)."
            )
            return False, None

        body = self.rfile.read(content_length) if content_length > 0 else None
        return True, body

    def validate_context_limits(self, body: Optional[bytes]) -> Tuple[bool, int, str, Optional[str]]:
        """
        Subtask 4/5 Model-Aware Context Admission:
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
                counter=lambda p: count_using_local_llama_server(p, url=counter_url, timeout=10.0),
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
        ok, body = self.validate_request_path_and_body()
        if not ok:
            return
        self.proxy_request("GET", body)

    def do_POST(self):
        if not self.check_auth():
            return
        if self.path == "/v1/shutdown":
            self.do_SHUTDOWN()
            return
        ok, body = self.validate_request_path_and_body()
        if not ok:
            return
        self.proxy_request("POST", body)

    def proxy_request(self, method: str, body: Optional[bytes]):
        request_id = str(uuid.uuid4())[:8]
        target_url = f"http://127.0.0.1:{self.upstream_port}{self.path}"

        # 1. Hard Context Limit Check (Before acquiring slot)
        if method == "POST" and self.path.startswith("/v1/chat/completions"):
            valid_context, status_code, err_code, err_msg = self.validate_context_limits(body)
            if not valid_context:
                self.send_json_error(status_code, err_code, err_msg)
                return

        # 2. Atomic Single-Slot Scheduler Admission (Strict Overload Guard)
        # Bypasses scheduling for metadata routes (/v1/models, /v1/chat/completions/input_tokens)
        requires_slot = (self.path == "/v1/chat/completions" and method == "POST")
        if requires_slot:
            acquired, fail_reason = GLOBAL_SCHEDULER.acquire(request_id)
            if not acquired:
                if fail_reason == "QUEUE_FULL":
                    self.send_json_error(
                        429, "SLOT_OVERLOADED",
                        "Inference engine is currently busy with an active generation. Please retry in a few moments.",
                        headers_extra={"Retry-After": "5", "X-Request-ID": request_id}
                    )
                else:
                    self.send_json_error(
                        503, "SLOT_ACQUISITION_TIMEOUT",
                        "Timed out waiting for single inference slot to become available.",
                        headers_extra={"Retry-After": "10", "X-Request-ID": request_id}
                    )
                return

        try:
            req_headers = {
                k: v for k, v in self.headers.items()
                if k.lower() not in ("host", "authorization", "content-length")
            }
            req_headers["X-Request-ID"] = request_id

            upstream_req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
            # 300s overall lifetime for complex deep reasoning / agent tool generation
            with urllib.request.urlopen(upstream_req, timeout=300) as resp:
                is_event_stream = "text/event-stream" in resp.headers.get("Content-Type", "")

                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_header("X-Request-ID", request_id)

                if is_event_stream:
                    # Lossless transparent byte forwarding over chunked SSE
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    while True:
                        raw_chunk = resp.read(1024)
                        if not raw_chunk:
                            self.wfile.write(b"0\r\n\r\n")
                            self.wfile.flush()
                            break
                        # Directly forward raw byte chunk without schema distortion or JSON re-parsing
                        self.wfile.write(f"{len(raw_chunk):X}\r\n".encode("ascii") + raw_chunk + b"\r\n")
                        self.wfile.flush()
                else:
                    # Non-streaming response: complete payload forwarded intact
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
            self.send_header("X-Request-ID", request_id)
            self.end_headers()
            try:
                self.wfile.write(e_data)
            except Exception:
                pass
        except Exception as e:
            # Client disconnected early (e.g. user aborted stream or closed terminal)
            if "Broken pipe" in str(e) or "Errno 32" in str(e) or isinstance(e, BrokenPipeError):
                return

            self.send_json_error(
                503, "SERVICE_UNAVAILABLE",
                f"Inference engine unavailable or initializing: {str(e)}",
                headers_extra={"Retry-After": "5", "X-Request-ID": request_id}
            )
        finally:
            # Atomic slot release guarantee (even if client disconnected, exception raised, or completed)
            if requires_slot:
                GLOBAL_SCHEDULER.release(request_id)

    def do_SHUTDOWN(self):
        """Emergency remote shutdown signal over authenticated gateway."""
        if not self.check_auth():
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "terminating", "message": "Worker shutdown signal accepted."}).encode("utf-8"))
        
        def _terminate():
            time.sleep(1)
            print("[SHUTDOWN] Remote shutdown requested via gateway. Releasing GPU container...", flush=True)
            os._exit(0)

        threading.Thread(target=_terminate, daemon=True).start()


def start_gateway(port: int = 4000, upstream_port: int = 8080, master_key: str = ""):
    """Start the lightweight auth gateway."""
    key = master_key or os.environ.get("LITELLM_MASTER_KEY", "sk-ant-api03-fakekey1234")
    AuthenticatedGatewayHandler.master_key = key
    AuthenticatedGatewayHandler.upstream_port = upstream_port

    server = http.server.ThreadingHTTPServer(("0.0.0.0", port), AuthenticatedGatewayHandler)
    print(f"[Gateway] Secure Auth Gateway running on port {port} -> forwarding to llama-server:{upstream_port}", flush=True)
    server.serve_forever()
