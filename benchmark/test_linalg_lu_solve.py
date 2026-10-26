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

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [(4, 4), (16, 1), (33, 16), (64, 64), (128, 128)]
DTYPES = [torch.float32] + ([torch.float64] if flag_gems.runtime.device.support_fp64 else [])


class LinalgLuSolveBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = SHAPES

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SHAPES)


def _case_fn(shape, dtype):
    del dtype
    n, k = shape
    yield base.BenchmarkCasePlan(
        shape={"LU": [n, n], "pivots": [n], "B": [n, k]},
        params={"left": True, "adjoint": False},
        builder_args=(n, k),
    )


def _build_inputs_fn(plan, dtype, device):
    n, k = plan.builder_args
    torch.manual_seed(0)
    matrix = torch.randn(n, n, dtype=dtype, device=device)
    matrix = matrix @ matrix.mT + torch.eye(n, dtype=dtype, device=device) * n
    lu, pivots = torch.linalg.lu_factor(matrix)
    rhs = torch.randn(n, k, dtype=dtype, device=device)
    return lu, pivots, rhs


@pytest.mark.linalg_lu_solve
def test_linalg_lu_solve():
    LinalgLuSolveBenchmark(
        op_name="linalg_lu_solve",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.linalg_lu_solve,
        dtypes=DTYPES,
    ).run()
