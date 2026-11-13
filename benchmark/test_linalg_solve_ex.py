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

from . import base
from .generated_operator_utils import OperatorBenchmark

SHAPES = [
    (1, 4, 1),
    (1, 8, 1),
    (1, 16, 1),
    (4, 4, 4),
    (4, 8, 4),
    (4, 16, 4),
    (8, 8, 1),
    (8, 16, 1),
    (16, 8, 4),
    (16, 16, 4),
    (32, 8, 1),
]


class LinalgSolveExBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(SHAPES)
        self.shape_desc = "batch, n, nrhs"


def _case_fn(shape, dtype):
    del dtype
    batch, n, nrhs = shape
    yield base.BenchmarkCasePlan(
        shape={"A": [batch, n, n], "B": [batch, n, nrhs]},
        params={"left": True, "check_errors": False},
        builder_args=(batch, n, nrhs),
    )


def _build_inputs_fn(plan, dtype, device):
    batch, n, nrhs = plan.builder_args
    base_matrix = torch.randn((batch, n, n), dtype=dtype, device=device)
    eye = torch.eye(n, dtype=dtype, device=device).unsqueeze(0)
    matrix = base_matrix @ base_matrix.mT + eye * n
    rhs = torch.randn((batch, n, nrhs), dtype=dtype, device=device)
    return matrix, rhs, {"left": True, "check_errors": False}


@pytest.mark.linalg_solve_ex
def test_linalg_solve_ex():
    LinalgSolveExBenchmark(
        op_name="linalg_solve_ex",
        torch_op=torch.linalg.solve_ex,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    ).run()
