"""Check status output without running model prediction or GPU kernels."""

import importlib
import os
from types import SimpleNamespace

import pytest

status = importlib.import_module("flag_gems.flagtune.inference.status")


def test_status_uses_stderr_and_flush(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "builtins.print", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    status.print_status("Model loaded", version="1.0.0")
    args, kwargs = calls[0]
    assert "[FlagTune][" in args[0]
    assert f"[PID={os.getpid()}]" in args[0]
    assert kwargs == {"file": status.sys.stderr, "flush": True}


def test_exception_chain_redacts_url_secrets():
    cause = OSError("https://user:secret@host/model?token=hidden")
    error = RuntimeError("load failed")
    error.__cause__ = cause
    reason = status.exception_reason(error)
    assert "RuntimeError: load failed caused_by=OSError:" in reason
    assert "secret" not in reason
    assert "hidden" not in reason
    cause.__cause__ = error
    assert status.exception_reason(error) == reason


def test_status_closed_stream_does_not_fail(monkeypatch):
    def closed(*args, **kwargs):
        raise BrokenPipeError("closed")

    monkeypatch.setattr("builtins.print", closed)
    status.print_status("Model loaded")


def test_model_loaded_once_per_version(monkeypatch, capsys):
    proposer = pytest.importorskip("triton.flagtune.runtime.proposer")
    identity_module = pytest.importorskip("triton.flagtune.contract.identity")
    cm = importlib.import_module("flag_gems.flagtune.inference.cost_model")
    for name in ("_FLAGTUNE_PROPOSER_POOL", "_FLAGTUNE_VARIANT_INFO_POOL"):
        monkeypatch.setattr(cm, name, {})
    monkeypatch.setattr(cm, "_MODEL_LOAD_STARTED", set())
    loaded = SimpleNamespace(model_version="1.0.0", variant=object())
    monkeypatch.setattr(proposer, "load_model_bundle", lambda *a, **k: loaded)
    monkeypatch.setattr(proposer, "make_config_proposer", lambda *a, **k: object())
    identity = identity_module.ModelIdentity(
        "nvidia-h20", "flaggems/mul", "scalar", "bf16-bf16"
    )
    cm.ensure_proposer(identity)
    cm.ensure_proposer(identity)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("Model loading:") == 1
    assert output.err.count("Model loaded:") == 1
    assert "version=1.0.0" in output.err
    loaded.model_version = "1.0.1"
    cm.ensure_proposer(identity)
    assert "version=1.0.1" in capsys.readouterr().err
