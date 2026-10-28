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

import pytest
import torch

from . import base, consts, utils


class MedianBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPE_DESC = "input shape"

    def supports_cases(self):
        return True

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        torch.manual_seed(0)
        return (utils.generate_tensor_input(shape, case.dtype, self.device),)


class MedianOutBenchmark(MedianBenchmark):
    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        inp = utils.generate_tensor_input(shape, case.dtype, self.device)
        out = torch.empty((), dtype=case.dtype, device=self.device)
        return inp, {"out": out}


@pytest.mark.median
def test_median():
    MedianBenchmark(
        op_name="median",
        torch_op=torch.median,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


@pytest.mark.median_out
def test_median_out():
    MedianOutBenchmark(
        op_name="median_out",
        torch_op=torch.ops.aten.median.out,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


# The dim overloads previously had correctness suites but no benchmark entry.
_MEDIAN_DIM_CASES = [
    ((9, 7), 0, False),
    ((9, 7), -1, True),
    ((384, 7), 0, False),
    ((5, 9), 1, False),
]


class MedianDimBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(_MEDIAN_DIM_CASES)
        self.shape_desc = "input shape, reduction dim, keepdim"


def _median_dim_case_fn(shape, dtype):
    del dtype
    dims, dim, keepdim = shape
    dims = tuple(dims)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)},
        params={"dim": dim, "keepdim": keepdim},
        builder_args=(dims,),
    )


def _median_dim_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, plan.params["dim"], plan.params["keepdim"]


def _median_dim_values_build_inputs_fn(plan, dtype, device):
    inp, dim, keepdim = _median_dim_build_inputs_fn(plan, dtype, device)
    normalized_dim = dim % inp.ndim
    out_shape = list(inp.shape)
    if keepdim:
        out_shape[normalized_dim] = 1
    else:
        del out_shape[normalized_dim]
    values = torch.empty(out_shape, dtype=dtype, device=device)
    indices = torch.empty(out_shape, dtype=torch.int64, device=device)
    return inp, dim, keepdim, {"values": values, "indices": indices}


@pytest.mark.median_dim
def test_median_dim():
    MedianDimBenchmark(
        op_name="median_dim",
        torch_op=torch.ops.aten.median.dim,
        case_fn=_median_dim_case_fn,
        build_inputs_fn=_median_dim_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()


@pytest.mark.median_dim_values
def test_median_dim_values():
    MedianDimBenchmark(
        op_name="median_dim_values",
        torch_op=torch.ops.aten.median.dim_values,
        case_fn=_median_dim_case_fn,
        build_inputs_fn=_median_dim_values_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES,
    ).run()
