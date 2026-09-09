"""Validate the strict FlagTune device-operation adapter in its focused suite."""

import importlib.util
from types import SimpleNamespace

import pytest

HAS_FLAGTREE_FLAGTUNE = importlib.util.find_spec("triton.flagtune") is not None
pytestmark = pytest.mark.skipif(
    not HAS_FLAGTREE_FLAGTUNE,
    reason="FlagGems FlagTune device-adapter tests require the optional FlagTree package",
)

if HAS_FLAGTREE_FLAGTUNE:
    from triton.flagtune.runtime.device import (  # noqa: E402
        DeviceDescriptor,
        DeviceProbeError,
    )

    from flag_gems.flagtune.runtime.device import (  # noqa: E402
        DeviceRuntime,
        DeviceUnavailableError,
    )


class _FakeDeviceAPI:
    def __init__(self, count=2, available=True):
        self._count = count
        self._available = available
        self.selected = None
        self.synchronize_count = 0

    def device_count(self):
        return self._count

    def is_available(self):
        return self._available

    def set_device(self, index):
        self.selected = index

    def synchronize(self):
        self.synchronize_count += 1


class _FakeTorch:
    bfloat16 = object()

    def __init__(self, api):
        self.cuda = api

    @staticmethod
    def device(device_type, index):
        return (device_type, index)

    @staticmethod
    def empty(shape, *, device, dtype):
        return {"shape": shape, "device": device, "dtype": dtype}


def _descriptor(backend="cuda"):
    return DeviceDescriptor(
        backend=backend,
        vendor="nvidia" if backend == "cuda" else "amd",
        torch_device_type="cuda",
        device_name="Test GPU",
        architecture="sm90" if backend == "cuda" else "gfx942",
        device_index=0,
    )


def test_cuda_runtime_owns_visibility_and_tensor_operations():
    api = _FakeDeviceAPI()
    runtime = DeviceRuntime(_descriptor(), _FakeTorch(api))

    assert runtime.device_count() == 2
    assert runtime.is_available() is True
    assert runtime.visible_device_tokens(
        2, environ={"CUDA_VISIBLE_DEVICES": "4,7"}
    ) == ["4", "7"]
    environment = {}
    runtime.apply_worker_visibility(environment, "7")
    assert environment == {"CUDA_VISIBLE_DEVICES": "7"}

    runtime.set_device(1)
    runtime.synchronize()
    tensor = runtime.make_tensor("empty", (2, 3), dtype=runtime.dtype("bfloat16"))
    assert api.selected == 1
    assert api.synchronize_count == 1
    assert tensor["device"] == ("cuda", 0)
    assert tensor["shape"] == (2, 3)


def test_hip_runtime_uses_rocm_visibility_names_without_changing_torch_api():
    runtime = DeviceRuntime(_descriptor("hip"), _FakeTorch(_FakeDeviceAPI()))
    environment = {}
    runtime.apply_worker_visibility(environment, "3")

    assert environment == {"ROCR_VISIBLE_DEVICES": "3"}
    assert runtime.visible_device_tokens(1, environ={"ROCR_VISIBLE_DEVICES": "6"}) == [
        "6"
    ]


def test_runtime_rejects_missing_or_empty_device_api():
    with pytest.raises(DeviceProbeError, match="torch.cuda"):
        DeviceRuntime(_descriptor(), SimpleNamespace())

    runtime = DeviceRuntime(_descriptor(), _FakeTorch(_FakeDeviceAPI(count=0)))
    with pytest.raises(DeviceUnavailableError, match="no visible devices"):
        runtime.device_count()


def test_runtime_metadata_keeps_architecture_separate_from_platform(monkeypatch):
    runtime = DeviceRuntime(_descriptor(), _FakeTorch(_FakeDeviceAPI()))
    monkeypatch.setattr(
        "flag_gems.flagtune.runtime.device.probe_flagtune_device",
        lambda _index=0: _descriptor(),
    )

    assert runtime.metadata() == {
        "backend": "cuda",
        "vendor": "nvidia",
        "device_name": "Test GPU",
        "architecture": "sm90",
        "platform_key": "nvidia-test-gpu",
    }


def test_runtime_resolves_benchmark_only_through_triton_boundary(monkeypatch):
    runtime = DeviceRuntime(_descriptor(), _FakeTorch(_FakeDeviceAPI()))
    observed = {}
    sentinel = object()

    def fake_resolve(mode, **kwargs):
        observed["mode"] = mode
        observed.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        "triton.flagtune.runtime.benchmark_protocol.resolve_benchmarker",
        fake_resolve,
    )

    assert (
        runtime.resolve_benchmarker(
            "replay",
            warmup_ms=25,
            measurement_ms=100,
            n_retries=10,
        )
        is sentinel
    )
    assert observed == {
        "mode": "replay",
        "warmup_ms": 25,
        "measurement_ms": 100,
        "n_retries": 10,
    }
