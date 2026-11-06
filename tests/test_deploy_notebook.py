"""Unit tests for deploy_notebook modules and command generation."""
import pytest
from pathlib import Path
from kaggle.config.model_config.catalog import ModelCatalog
from kaggle.deploy_notebook.server_runner import build_native_server_command
from kaggle.deploy_notebook.environment import query_physical_vram


def test_build_native_server_command_flags():
    model = ModelCatalog.get_by_id("qwen38-27b")
    cmd = build_native_server_command(
        binary_path="llama-server",
        model_path="model.gguf",
        profile=model,
        effective_context=65536,
        port=8080,
        host="127.0.0.1",
        mmproj_path=None,
    )
    
    # Assert single-slot assistant flag
    assert "--parallel" in cmd
    idx_p = cmd.index("--parallel")
    assert cmd[idx_p + 1] == "1"

    # Assert no reasoning preservation flag
    assert "--no-reasoning-preserve" in cmd

    # Assert flash-attn on
    assert "--flash-attn" in cmd
    idx_fa = cmd.index("--flash-attn")
    assert cmd[idx_fa + 1] == "on"

    # Assert 64k context size
    assert "--ctx-size" in cmd
    idx_ctx = cmd.index("--ctx-size")
    assert cmd[idx_ctx + 1] == "65536"


def test_query_physical_vram_graceful_fallback():
    # On environments without nvidia-smi (or non-GPU runner), it returns a list gracefully without crashing
    res = query_physical_vram()
    assert isinstance(res, list)


def test_auth_gateway_auth_enforcement():
    from kaggle.deploy_notebook.auth_gateway import AuthenticatedGatewayHandler
    import io

    class DummyHandler(AuthenticatedGatewayHandler):
        def __init__(self, auth_header):
            self.headers = {"Authorization": auth_header}
            self.response_code = None
            self.wfile = io.BytesIO()

        def send_response(self, code):
            self.response_code = code

        def send_header(self, k, v):
            pass

        def end_headers(self):
            pass

    # Reject missing / invalid token
    bad = DummyHandler("Bearer wrong_token")
    assert bad.check_auth() is False
    assert bad.response_code == 401

    # Accept exact master key
    good = DummyHandler("Bearer sk-ant-api03-fakekey1234")
    assert good.check_auth() is True
