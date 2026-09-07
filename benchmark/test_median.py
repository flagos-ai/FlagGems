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

from . import base, consts, utils


class MedianNoDimBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield (utils.generate_tensor_input(shape, cur_dtype, self.device),)


class MedianReductionBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape or [input shape, dim, keepdim]"

    def get_input_iter(self, cur_dtype) -> Generator:
        for case_id, shape_spec in enumerate(self.shapes):
            if shape_spec and isinstance(shape_spec[0], (list, tuple)):
                shape = tuple(shape_spec[0])
                dim = int(shape_spec[1])
                keepdim = bool(shape_spec[2]) if len(shape_spec) > 2 else False
            else:
                shape = shape_spec
                keepdim = case_id % 3 == 0
                if len(shape) == 1:
                    dim = 0
                elif case_id % 2 == 0:
                    dim = len(shape) - 1
                else:
                    dim = 0
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)
            yield inp, dim, {"keepdim": keepdim}


@pytest.mark.median
def test_median():
    bench = MedianNoDimBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median
def test_median_dim():
    bench = MedianReductionBenchmark(
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


class MedianFlatBenchmark(base.GenericBenchmark):
    """Median-family benchmark that reduces the whole tensor (no dim)."""

    def set_shapes(self, shape_file_path=None):
        # Whole-tensor median reduction width is capped by the flag_gems
        # median kernels (fp32/fp16/bf16 key-select limit 16384, int-select
        # limit 16384), so keep numel <= 16384.
        self.shapes = [
            (64,),
            (256,),
            (1024,),
            (4096,),
            (16384,),
        ]
        self.shape_desc = "input shape"


class MedianDimBenchmark(base.GenericBenchmark):
    """Median-family benchmark that reduces over the last dimension."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (256, 256),
            (1024, 1024),
            (256, 4096),
        ]
        self.shape_desc = "M, N"


def _median_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty((), dtype=dtype, device=device)
    yield inp, {"out": out}


def _median_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, {"dim": -1}


def _median_dim_values_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out_shape = tuple(shape[:-1])
    out_values = torch.empty(out_shape, dtype=dtype, device=device)
    out_indices = torch.empty(out_shape, dtype=torch.long, device=device)
    yield inp, {"dim": -1, "out": (out_values, out_indices)}


@pytest.mark.median_out
def test_perf_median_out():
    bench = MedianFlatBenchmark(
        input_fn=_median_out_input_fn,
        op_name="median_out",
        torch_op=torch.ops.aten.median.out,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median_dim
def test_perf_median_dim():
    bench = MedianDimBenchmark(
        input_fn=_median_dim_input_fn,
        op_name="median_dim",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()


@pytest.mark.median_dim_values
def test_perf_median_dim_values():
    bench = MedianDimBenchmark(
        input_fn=_median_dim_values_input_fn,
        op_name="median_dim_values",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    )
    bench.run()
