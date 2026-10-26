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


from . import base, consts
from .generated_operator_utils import OperatorBenchmark
from .test_linalg_polar import _PolarBaseline, _native_polar

CORE_SHAPES = [(4, 2), (8, 4), (16, 8), (16, 8, 4)]
COMPREHENSIVE_SHAPES = [(64, 8, 4), (32, 16, 8)]


def _native_polar_out(inp, *, U, H):
    result_u, result_h = _native_polar(inp)
    U.copy_(result_u)
    H.copy_(result_h)
    return U, H


class LinalgPolarOutBenchmark(_PolarBaseline, OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(CORE_SHAPES)
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes.extend(COMPREHENSIVE_SHAPES)
        self.shape_desc = "batch dimensions, rows, columns"


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    h_shape = (*dims[:-2], dims[-1], dims[-1])
    yield base.BenchmarkCasePlan(
        shape={"A": list(dims), "U": list(dims), "H": list(h_shape)},
        params={},
        builder_args=(dims, h_shape),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, h_shape = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    out_u = torch.empty(shape, dtype=dtype, device=device)
    out_h = torch.empty(h_shape, dtype=dtype, device=device)
    return inp, {"U": out_u, "H": out_h}


@pytest.mark.linalg_polar_out
def test_linalg_polar_out():
    LinalgPolarOutBenchmark(
        op_name="linalg_polar_out",
        torch_op=_native_polar_out,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
        is_inplace=True,
        fresh_inputs=True,
    ).run()
