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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="native special_laguerre_polynomial_l benchmark falls back to CPU on Ascend",
)

_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)


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


@pytest.mark.special_laguerre_polynomial_l
def test_special_laguerre_polynomial_l():
    _TensorTensorBenchmark(
        op_name="special_laguerre_polynomial_l",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_out
def test_special_laguerre_polynomial_l_out():
    _TensorTensorOutBenchmark(
        op_name="special_laguerre_polynomial_l_out",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_n_scalar
def test_special_laguerre_polynomial_l_n_scalar():
    _TensorScalarBenchmark(
        op_name="special_laguerre_polynomial_l_n_scalar",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_n_scalar_out
def test_special_laguerre_polynomial_l_n_scalar_out():
    _TensorScalarOutBenchmark(
        op_name="special_laguerre_polynomial_l_n_scalar_out",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_x_scalar
def test_special_laguerre_polynomial_l_x_scalar():
    _ScalarTensorBenchmark(
        op_name="special_laguerre_polynomial_l_x_scalar",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()


@pytest.mark.special_laguerre_polynomial_l_x_scalar_out
def test_special_laguerre_polynomial_l_x_scalar_out():
    _ScalarTensorOutBenchmark(
        op_name="special_laguerre_polynomial_l_x_scalar_out",
        torch_op=torch.special.laguerre_polynomial_l,
        dtypes=_DTYPES,
    ).run()
