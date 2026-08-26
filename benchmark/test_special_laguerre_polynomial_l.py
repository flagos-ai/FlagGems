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

import re
import warnings
from typing import Generator

import pytest
import torch

import flag_gems

from . import base

_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)

_CPU_FALLBACK_RE = re.compile(
    r"(?:fall(?:ing)?\s*back|fallback).{0,120}cpu|"
    r"cpu.{0,120}(?:fall(?:ing)?\s*back|fallback)",
    re.IGNORECASE | re.DOTALL,
)
_native_fallback_seen = False


def _skip_if_native_baseline_falls_back(capfd, make_probe):
    """Skip performance data that would time a native CPU fallback."""
    global _native_fallback_seen
    if _native_fallback_seen:
        pytest.skip("native special_laguerre_polynomial_l falls back to CPU")

    for dtype in _DTYPES:
        args, kwargs = make_probe(dtype)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = torch.special.laguerre_polynomial_l(*args, **kwargs)
                flag_gems.runtime.torch_device_fn.synchronize()
            captured = capfd.readouterr()
            messages = "\n".join(str(item.message) for item in caught)
            messages += "\n" + captured.out + "\n" + captured.err
        except (NotImplementedError, RuntimeError) as error:
            pytest.skip(f"native device baseline is unavailable: {error}")

        input_devices = [arg.device.type for arg in args if torch.is_tensor(arg)]
        silent_cpu_fallback = (
            torch.is_tensor(result)
            and result.device.type == "cpu"
            and any(device != "cpu" for device in input_devices)
        )
        if _CPU_FALLBACK_RE.search(messages) or silent_cpu_fallback:
            _native_fallback_seen = True
            pytest.skip("native special_laguerre_polynomial_l falls back to CPU")


class _LaguerreBenchmark(base.Benchmark):
    _degrees = (0.75, 1.75, 2.75, 8.75)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 19, 7),
            (1024, 1024),
            (20, 320, 15),
            (64, 64, 64),
        ]

    @staticmethod
    def _x(shape, dtype, device):
        return torch.rand(shape, dtype=dtype, device=device) * 2.0 - 1.0


class _TensorTensorBenchmark(_LaguerreBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape, degree in zip(self.shapes, self._degrees):
            x = self._x(shape, cur_dtype, self.device)
            n = torch.tensor(degree, dtype=cur_dtype, device=self.device)
            yield x, n


class _TensorTensorOutBenchmark(_TensorTensorBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for x, n in super().get_input_iter(cur_dtype):
            yield x, n, {"out": torch.empty_like(x)}


class _TensorScalarBenchmark(_LaguerreBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape, degree in zip(self.shapes, self._degrees):
            x = self._x(shape, cur_dtype, self.device)
            yield x, degree


class _TensorScalarOutBenchmark(_TensorScalarBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for x, n in super().get_input_iter(cur_dtype):
            yield x, n, {"out": torch.empty_like(x)}


class _ScalarTensorBenchmark(_LaguerreBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape, degree in zip(self.shapes, self._degrees):
            n = torch.full(shape, degree, dtype=cur_dtype, device=self.device)
            yield 0.25, n


class _ScalarTensorOutBenchmark(_ScalarTensorBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for x, n in super().get_input_iter(cur_dtype):
            yield x, n, {"out": torch.empty_like(n)}


def _probe_tensor_tensor(dtype, out=False):
    x = torch.tensor([0.25, -0.5], dtype=dtype, device=flag_gems.device)
    n = torch.tensor([2.0, 4.0], dtype=dtype, device=flag_gems.device)
    kwargs = {"out": torch.empty_like(x)} if out else {}
    return (x, n), kwargs


def _probe_tensor_scalar(dtype, out=False):
    x = torch.tensor([0.25, -0.5], dtype=dtype, device=flag_gems.device)
    kwargs = {"out": torch.empty_like(x)} if out else {}
    return (x, 4.0), kwargs


def _probe_scalar_tensor(dtype, out=False):
    n = torch.tensor([2.0, 4.0], dtype=dtype, device=flag_gems.device)
    kwargs = {"out": torch.empty_like(n)} if out else {}
    return (0.25, n), kwargs


@pytest.mark.special_laguerre_polynomial_l
def test_special_laguerre_polynomial_l(capfd):
    _skip_if_native_baseline_falls_back(capfd, _probe_tensor_tensor)
    _TensorTensorBenchmark(
        op_name="special_laguerre_polynomial_l",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_out
def test_special_laguerre_polynomial_l_out(capfd):
    _skip_if_native_baseline_falls_back(
        capfd, lambda dtype: _probe_tensor_tensor(dtype, out=True)
    )
    _TensorTensorOutBenchmark(
        op_name="special_laguerre_polynomial_l_out",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_n_scalar
def test_special_laguerre_polynomial_l_n_scalar(capfd):
    _skip_if_native_baseline_falls_back(capfd, _probe_tensor_scalar)
    _TensorScalarBenchmark(
        op_name="special_laguerre_polynomial_l_n_scalar",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_n_scalar_out
def test_special_laguerre_polynomial_l_n_scalar_out(capfd):
    _skip_if_native_baseline_falls_back(
        capfd, lambda dtype: _probe_tensor_scalar(dtype, out=True)
    )
    _TensorScalarOutBenchmark(
        op_name="special_laguerre_polynomial_l_n_scalar_out",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_x_scalar
def test_special_laguerre_polynomial_l_x_scalar(capfd):
    _skip_if_native_baseline_falls_back(capfd, _probe_scalar_tensor)
    _ScalarTensorBenchmark(
        op_name="special_laguerre_polynomial_l_x_scalar",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_x_scalar_out
def test_special_laguerre_polynomial_l_x_scalar_out(capfd):
    _skip_if_native_baseline_falls_back(
        capfd, lambda dtype: _probe_scalar_tensor(dtype, out=True)
    )
    _ScalarTensorOutBenchmark(
        op_name="special_laguerre_polynomial_l_x_scalar_out",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()
