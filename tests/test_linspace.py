# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import random
import time

import pytest
import torch

import flag_gems
from flag_gems.ops.linspace import linspace as generic_linspace

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


@functools.lru_cache(maxsize=None)
def _device_linspace_supports_integral(dtype):
    """Whether torch.linspace on the device handles the given integral dtype.

    Some vendor builds (e.g. xdnn_pytorch_wrapper on Kunlunxin) only implement
    a subset of dtype combinations, so unsupported dtypes fall back to the CPU
    reference.
    """
    try:
        torch.linspace(0, 1, 4, dtype=dtype, device=flag_gems.device)
    except Exception:
        return False
    return True


@pytest.mark.linspace
@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("end", [256, 2048, 4096])
@pytest.mark.parametrize("steps", [1, 256, 512])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("device", [flag_gems.device, None])
@pytest.mark.parametrize("pin_memory", [False, None])
def test_linspace(start, end, steps, dtype, device, pin_memory):
    ref_device = "cpu" if cfg.TO_CPU else device
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype in (torch.int16, torch.int32, torch.int64)
        and not _device_linspace_supports_integral(dtype)
    ):
        # The vendor's xdnn_pytorch_wrapper only implements a subset of dtype
        # combinations (e.g. not int16), so fall back to the CPU reference.
        ref_device = "cpu"

    ref_out = torch.linspace(
        start,
        end,
        steps,
        dtype=dtype,
        layout=None,
        device=ref_device,
        pin_memory=pin_memory,
    )
    if device is None:
        res_out = torch.linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )
    else:
        res_out = flag_gems.linspace(
            start,
            end,
            steps,
            dtype=dtype,
            layout=None,
            device=device,
            pin_memory=pin_memory,
        )

    if ref_device == "cpu" and res_out.device != torch.device("cpu"):
        res_out = res_out.cpu()

    if dtype in [torch.float16, torch.bfloat16, torch.float32, None]:
        utils.gems_assert_close(res_out, ref_out, dtype=dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.linspace
@pytest.mark.skipif(
    flag_gems.linspace is not generic_linspace,
    reason="Generic-only empty fix; vendor overrides unchanged",
)
class TestGenericLinspaceEmpty:
    @staticmethod
    def _endpoints(kind):
        if kind == "real":
            return -2.0, 7.0
        if kind == "complex":
            return 1 + 2j, -3 + 4j
        if kind == "tensor_real":
            return torch.tensor(-2.0, device="cpu"), torch.tensor(7.0, device="cpu")
        return torch.tensor(1 + 2j, device="cpu"), torch.tensor(-3 + 4j, device="cpu")

    @staticmethod
    def _compare(start, end, **kwargs):
        reference_kwargs = dict(kwargs, device="cpu")
        try:
            expected = torch.linspace(start, end, 0, **reference_kwargs)
        except Exception as error:
            with pytest.raises(type(error)):
                generic_linspace(start, end, 0, **kwargs)
            return
        result = generic_linspace(start, end, 0, **kwargs)
        assert result.shape == expected.shape == (0,)
        assert result.dtype == expected.dtype
        assert result.layout == expected.layout
        assert result.stride() == expected.stride()
        assert not result.requires_grad
        target = kwargs.get("device")
        if target is not None:
            assert result.device.type == torch.device(target).type
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "kind", ["real", "complex", "tensor_real", "tensor_complex"]
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            None,
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int64,
            torch.uint8,
            torch.bool,
            torch.complex64,
            torch.complex128,
        ],
    )
    @pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
    def test_dtype_and_endpoints(self, kind, dtype, default_dtype):
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(default_dtype)
            start, end = self._endpoints(kind)
            self._compare(start, end, dtype=dtype, device=flag_gems.device)
        finally:
            torch.set_default_dtype(previous)

    @pytest.mark.parametrize("pin_memory", [None, False, True])
    @pytest.mark.parametrize("layout", [None, torch.strided, torch.sparse_coo])
    def test_cpu_allocation_options(self, pin_memory, layout):
        self._compare(1.0, 2.0, device="cpu", layout=layout, pin_memory=pin_memory)

    def test_default_and_meta_device(self):
        with torch.device("cpu"):
            self._compare(1, 2, device=None)
        with torch.device("meta"):
            result = generic_linspace(1, 2, 0)
            expected = torch.linspace(1, 2, 0)
            assert result.device == expected.device
            assert result.dtype == expected.dtype
            assert result.shape == expected.shape
        result = generic_linspace(1, 2, 0, device="meta")
        assert result.device.type == "meta"
        assert result.shape == (0,)

    @pytest.mark.parametrize(
        "start,end",
        [
            (torch.tensor([1.0], device="cpu"), 2),
            (1, torch.tensor([2.0], device="cpu")),
            (None, 2),
            (1, "2"),
            ([1], 2),
            (1 << 65, 2),
            (-(1 << 64), 2),
        ],
    )
    def test_invalid_zero_endpoints(self, start, end):
        self._compare(start, end, device="cpu")

    @pytest.mark.parametrize("steps", [-1, -7, 0.0, False, True, None, "0"])
    def test_existing_step_rejection(self, steps):
        # This repair does not widen zero-equal non-integer step arguments.
        with pytest.raises((AssertionError, TypeError)):
            generic_linspace(1, 2, steps, device="cpu")

    def test_no_fill_launch_and_fresh_output(self, monkeypatch):
        import importlib

        module = importlib.import_module(generic_linspace.__module__)

        class UnexpectedKernel:
            def __getitem__(self, grid):
                raise AssertionError("empty linspace must not launch a Triton fill")

        monkeypatch.setattr(module, "linspace_kernel", UnexpectedKernel())
        first = generic_linspace(1, 2, 0, device="cpu")
        second = generic_linspace(1, 2, 0, device="cpu")
        assert first is not second
        assert first.untyped_storage()._cdata != second.untyped_storage()._cdata
        assert first.shape == second.shape == (0,)
