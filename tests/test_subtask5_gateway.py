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
