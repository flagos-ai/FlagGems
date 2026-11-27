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

from . import base, consts


class LshiftBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Representative vector, matrix, and three-dimensional pointwise inputs.
        self.shapes = [(1024,), (1024, 1024), (16, 512, 256)]
        self.shape_desc = "SHAPE"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            value = torch.randint(0, 100, shape, dtype=cur_dtype, device=self.device)
            shift = torch.randint(0, 8, shape, dtype=cur_dtype, device=self.device)
            yield value, shift


class LshiftScalarBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1024,), (1024, 1024), (16, 512, 256)]
        self.shape_desc = "SHAPE"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            value = torch.randint(0, 100, shape, dtype=cur_dtype, device=self.device)
            yield value, 3


class LshiftOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1024,), (1024, 1024), (16, 512, 256)]
        self.shape_desc = "SHAPE"

    def get_input_iter(self, cur_dtype) -> Generator:
        # The ``.Tensor_out`` overload requires the ``out`` tensor to be passed
        # as a keyword argument; yield it as a dict so the benchmark harness
        # forwards it via kwargs.
        for shape in self.shapes:
            value = torch.randint(0, 100, shape, dtype=cur_dtype, device=self.device)
            shift = torch.randint(0, 8, shape, dtype=cur_dtype, device=self.device)
            out = torch.empty_like(value)
            yield value, shift, {"out": out}


class LshiftScalarOutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1024,), (1024, 1024), (16, 512, 256)]
        self.shape_desc = "SHAPE"

    def get_input_iter(self, cur_dtype) -> Generator:
        # The ``.Scalar_out`` overload also requires the ``out`` keyword arg.
        for shape in self.shapes:
            value = torch.randint(0, 100, shape, dtype=cur_dtype, device=self.device)
            out = torch.empty_like(value)
            yield value, 3, {"out": out}


@pytest.mark.lshift_tensor
def test_lshift():
    bench = LshiftBenchmark(
        op_name="lshift_tensor",
        torch_op=torch.ops.aten.__lshift__.Tensor,
        dtypes=consts.INT_DTYPES + consts.EXTRA_INT_DTYPES,
    )
    bench.run()


@pytest.mark.lshift_scalar
def test_lshift_scalar():
    bench = LshiftScalarBenchmark(
        op_name="lshift_scalar",
        torch_op=torch.ops.aten.__lshift__.Scalar,
        dtypes=consts.INT_DTYPES + consts.EXTRA_INT_DTYPES,
    )
    bench.run()


@pytest.mark.lshift_tensor_out
def test_lshift_tensor_out():
    bench = LshiftOutBenchmark(
        op_name="lshift_tensor_out",
        torch_op=torch.ops.aten.__lshift__.Tensor_out,
        dtypes=consts.INT_DTYPES + consts.EXTRA_INT_DTYPES,
    )
    bench.run()


@pytest.mark.lshift_scalar_out
def test_lshift_scalar_out():
    bench = LshiftScalarOutBenchmark(
        op_name="lshift_scalar_out",
        torch_op=torch.ops.aten.__lshift__.Scalar_out,
        dtypes=consts.INT_DTYPES + consts.EXTRA_INT_DTYPES,
    )
    bench.run()
