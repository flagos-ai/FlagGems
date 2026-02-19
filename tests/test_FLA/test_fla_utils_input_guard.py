import pytest
import torch

import flag_gems
from flag_gems.fused.FLA.utils import input_guard

DEVICE_TYPE = flag_gems.device
DEVICE_COUNT = flag_gems.runtime.device.device_count
_runtime = flag_gems.runtime

_has_current_device = hasattr(_runtime.torch_device_fn, "current_device")


def _device(index=0):
    return torch.device(DEVICE_TYPE, index)


def _non_contiguous(device, shape=(4, 4)):
    nc = torch.randn(*shape, device=device, dtype=torch.float32).t()
    assert not nc.is_contiguous()
    return nc


def _skip_if_no_device():
    if DEVICE_COUNT == 0:
        pytest.skip(f"No {DEVICE_TYPE} devices available")


def _skip_if_fewer_than(n):
    if DEVICE_COUNT < n:
        pytest.skip(f"Need at least {n} {DEVICE_TYPE} devices, got {DEVICE_COUNT}")


_captured: dict = {}


@input_guard
def _capture(*args, **kwargs):
    _captured["args"] = tuple(args)
    _captured["kwargs"] = dict(kwargs)


@input_guard
def _return_current_device(x: torch.Tensor) -> int:
    return _runtime.torch_device_fn.current_device()


@input_guard
def _return_value(x):
    return x


@input_guard
def _two_tensors(a: torch.Tensor, b: torch.Tensor) -> torch.device:
    return a.device


@input_guard
def _no_tensor_fn(scalar: int) -> int:
    return scalar * 2


class TestInputGuardWraps:
    def test_name_preserved(self):
        @input_guard
        def _my_fn(x):
            """My docstring."""

        assert _my_fn.__name__ == "_my_fn"

    def test_doc_preserved(self):
        @input_guard
        def _doc_fn(x):
            """Documented function."""

        assert _doc_fn.__doc__ == "Documented function."

    def test_wrapped_attribute_points_to_original(self):
        def _orig(x):
            pass

        assert input_guard(_orig).__wrapped__ is _orig


class TestInputGuardContiguous:
    def test_single_non_contiguous_arg(self):
        _skip_if_no_device()
        _capture(_non_contiguous(_device()))
        assert _captured["args"][0].is_contiguous()

    def test_multiple_non_contiguous_args(self):
        _skip_if_no_device()
        _capture(_non_contiguous(_device()), _non_contiguous(_device(), shape=(3, 5)))
        assert all(a.is_contiguous() for a in _captured["args"])

    def test_already_contiguous_arg_stays_contiguous(self):
        _skip_if_no_device()
        _capture(torch.randn(4, 4, device=_device(), dtype=torch.float32))
        assert _captured["args"][0].is_contiguous()

    def test_single_non_contiguous_kwarg(self):
        _skip_if_no_device()
        _capture(x=_non_contiguous(_device()))
        assert _captured["kwargs"]["x"].is_contiguous()

    def test_multiple_non_contiguous_kwargs(self):
        _skip_if_no_device()
        _capture(
            x=_non_contiguous(_device()), y=_non_contiguous(_device(), shape=(2, 6))
        )
        assert _captured["kwargs"]["x"].is_contiguous()
        assert _captured["kwargs"]["y"].is_contiguous()

    def test_mixed_args_and_kwargs_all_contiguous(self):
        _skip_if_no_device()
        _capture(_non_contiguous(_device()), y=_non_contiguous(_device()))
        assert _captured["args"][0].is_contiguous()
        assert _captured["kwargs"]["y"].is_contiguous()

    def test_non_tensor_arg_passed_through_unchanged(self):
        _skip_if_no_device()
        _capture(_non_contiguous(_device()), 42, "hello")
        assert _captured["args"][1] == 42
        assert _captured["args"][2] == "hello"

    def test_non_tensor_kwarg_passed_through_unchanged(self):
        _skip_if_no_device()
        _capture(_non_contiguous(_device()), scale=3.14)
        assert _captured["kwargs"]["scale"] == 3.14

    def test_contiguous_version_has_same_data(self):
        _skip_if_no_device()
        nc = _non_contiguous(_device())
        _capture(nc)
        torch.testing.assert_close(_captured["args"][0], nc.contiguous())

    def test_return_value_preserved(self):
        _skip_if_no_device()
        t = torch.ones(2, 2, device=_device(), dtype=torch.float32)
        torch.testing.assert_close(_return_value(t), t)


class TestInputGuardContext:
    @pytest.mark.parametrize("idx", [0, 7])
    def test_output_on_correct_device(self, idx):
        _skip_if_no_device()
        _skip_if_fewer_than(idx + 1)
        x = torch.randn(2, 2, device=_device(idx), dtype=torch.float32)
        _capture(x)
        assert _captured["args"][0].device == x.device

    def test_all_available_devices(self):
        for i in range(min(DEVICE_COUNT, 8)):
            x = torch.ones(1, device=_device(i), dtype=torch.float32)
            _capture(x)
            assert _captured["args"][0].device.index == i

    @pytest.mark.parametrize("idx", [0, 7])
    def test_current_device_matches_input(self, idx):
        if not _has_current_device:
            pytest.skip("Backend has no current_device()")
        _skip_if_fewer_than(idx + 1)
        x = torch.ones(1, device=_device(idx), dtype=torch.float32)
        assert _return_current_device(x) == idx

    def test_ctx_from_first_arg_not_second(self):
        _skip_if_fewer_than(2)
        a = torch.ones(1, device=_device(0), dtype=torch.float32)
        b = torch.ones(1, device=_device(1), dtype=torch.float32)
        assert _two_tensors(a, b) == _device(0)

    def test_ctx_from_first_kwarg_when_no_tensor_in_args(self):
        _skip_if_fewer_than(2)
        _capture(x=torch.ones(1, device=_device(1), dtype=torch.float32))
        assert _captured["kwargs"]["x"].device.index == 1

    def test_nullcontext_when_no_tensor(self):
        assert _no_tensor_fn(21) == 42
