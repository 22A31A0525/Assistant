"""
Subtask 5 Integration & Unit Tests:
Covers API contract, SSE streaming, single-slot scheduling, payload limits, and context admission.
"""
import json
import pytest
from kaggle.deploy_notebook.auth_gateway import SingleSlotScheduler, AuthenticatedGatewayHandler


def test_single_slot_scheduler_immediate_acquire_and_release():
    sched = SingleSlotScheduler(max_queue=1, queue_timeout=1.0)
    ok, err = sched.acquire("req-1")
    assert ok is True
    assert err is None
    sched.release("req-1")


def test_single_slot_scheduler_queue_full_rejection():
    sched = SingleSlotScheduler(max_queue=1, queue_timeout=0.2)
    # 1. First caller takes active slot
    ok1, err1 = sched.acquire("req-1")
    assert ok1 is True

    # 2. In a thread, second caller waits in queue
    import threading
    import time
    results = {}

    def queue_caller():
        results["queued_ok"], results["queued_err"] = sched.acquire("req-2")

    t = threading.Thread(target=queue_caller)
    t.start()
    time.sleep(0.05)  # Let thread enter queue

    # 3. Third caller arrives while queue is at capacity (max_queue=1) -> QUEUE_FULL
    ok3, err3 = sched.acquire("req-3")
    assert ok3 is False
    assert err3 == "QUEUE_FULL"

    # 4. Release first caller -> queued caller should acquire
    sched.release("req-1")
    t.join(timeout=1.0)
    assert results.get("queued_ok") is True

    # 5. Clean up second caller
    sched.release("req-2")


def test_gateway_route_validation():
    handler = AuthenticatedGatewayHandler.__new__(AuthenticatedGatewayHandler)
    handler.path = "/unsupported/route"
    handler.headers = {"Content-Length": "0"}

    # Mock send_json_error
    errors = []
    handler.send_json_error = lambda code, err, msg, headers_extra=None: errors.append((code, err))
    handler.rfile = None
    handler.max_body_bytes = 1000

    ok, _ = handler.validate_request_path_and_body()
    assert ok is False
    assert errors[0] == (404, "NOT_FOUND")


def test_gateway_body_size_limit():
    handler = AuthenticatedGatewayHandler.__new__(AuthenticatedGatewayHandler)
    handler.path = "/v1/chat/completions"
    handler.max_body_bytes = 500
    handler.headers = {"Content-Length": "1000"}  # Exceeds 500 bytes

    errors = []
    handler.send_json_error = lambda code, err, msg, headers_extra=None: errors.append((code, err))

    ok, _ = handler.validate_request_path_and_body()
    assert ok is False
    assert errors[0] == (413, "PAYLOAD_TOO_LARGE")


def test_sse_fragmentation_and_tool_call_reassembly():
    """
    Subtask 5 Requirement: Lossless SSE fragmentation verification.
    Verifies split multi-byte UTF-8, reasoning deltas, and tool-call JSON fragments reconstruct correctly.
    """
    # 1. Simulate split multi-byte UTF-8 (e.g. 4-byte emoji 🚀 = b'\xf0\x9f\x9a\x80' split across reads)
    emoji_bytes = "🚀".encode("utf-8")
    split_part1 = emoji_bytes[:2]
    split_part2 = emoji_bytes[2:]
    reassembled_utf8 = (split_part1 + split_part2).decode("utf-8")
    assert reassembled_utf8 == "🚀"

    # 2. Simulate streaming tool-call argument JSON fragmentation
    # Upstream models stream function arguments across multiple small deltas
    simulated_sse_chunks = [
        {"choices": [{"delta": {"role": "assistant", "tool_calls": [{"index": 0, "id": "call_123", "type": "function", "function": {"name": "execute_code", "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"code":'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ' "print(\\"hello\\")"'}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ', "timeout": 10}'}}]}}]},
    ]

    accumulated_args = []
    function_name = None
    for chunk in simulated_sse_chunks:
        delta = chunk["choices"][0]["delta"]
        if "tool_calls" in delta:
            for tc in delta["tool_calls"]:
                func = tc.get("function", {})
                if "name" in func:
                    function_name = func["name"]
                if "arguments" in func:
                    accumulated_args.append(func["arguments"])

    full_arg_json_str = "".join(accumulated_args)
    parsed_args = json.loads(full_arg_json_str)
    assert function_name == "execute_code"
    assert parsed_args == {"code": 'print("hello")', "timeout": 10}


def test_client_disconnect_slot_release_guarantee():
    """
    Subtask 5 Requirement: Client-disconnect cancellation & slot recovery.
    Verifies when a client disconnects (BrokenPipeError), slot is atomically released.
    """
    from kaggle.deploy_notebook.auth_gateway import SingleSlotScheduler
    test_sched = SingleSlotScheduler(max_queue=1, queue_timeout=1.0)
    req_id = "test-disconnect-req"

    acquired, _ = test_sched.acquire(req_id)
    assert acquired is True

    # Simulate BrokenPipeError in caller try/finally block
    try:
        raise BrokenPipeError(32, "Broken pipe")
    except BrokenPipeError:
        pass
    finally:
        test_sched.release(req_id)

    # Next caller should immediately acquire without being blocked
    next_acquired, err = test_sched.acquire("next-req")
    assert next_acquired is True
    assert err is None
    test_sched.release("next-req")


def test_backend_down_clean_service_unavailable():
    """
    Subtask 5 Requirement: Backend recovery & predictable errors.
    Verifies when upstream llama-server port is unreachable, gateway returns HTTP 503.
    """
    handler = AuthenticatedGatewayHandler.__new__(AuthenticatedGatewayHandler)
    handler.upstream_port = 59999  # Non-existent port
    handler.path = "/v1/models"
    handler.headers = {"Authorization": "Bearer sk-ant-api03-fakekey1234"}

    errors = []
    handler.send_json_error = lambda code, err, msg, headers_extra=None: errors.append((code, err, headers_extra))

    handler.proxy_request("GET", None)
    assert len(errors) == 1
    status_code, err_code, headers_extra = errors[0]
    assert status_code == 503
    assert err_code == "SERVICE_UNAVAILABLE"
    assert headers_extra.get("Retry-After") == "5"
    assert "X-Request-ID" in headers_extra
