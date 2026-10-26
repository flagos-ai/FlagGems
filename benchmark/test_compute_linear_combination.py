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

MNK_SHAPES = [
    (64, 64, 64),
    (128, 256, 512),
    (256, 512, 1024),
    (8, 1024, 4096),
]


class ComputeLinearCombinationBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(MNK_SHAPES)
        self.shape_desc = "(output combinations, input terms, flattened width)"


def _case_fn(shape, dtype):
    del dtype
    m, n, k = map(int, shape)
    yield base.BenchmarkCasePlan(
        shape={"input": [n, k], "coefficients": [m, n]},
        params={},
        builder_args=(m, n, k),
    )


def _build_inputs_fn(plan, dtype, device):
    m, n, k = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn((n, k), dtype=dtype, device=device)
    coefficients = torch.randn((m, n), dtype=dtype, device=device)
    return inp, coefficients


@pytest.mark.compute_linear_combination
@pytest.mark.underscore_compute_linear_combination
def test_compute_linear_combination():
    ComputeLinearCombinationBenchmark(
        op_name="_compute_linear_combination",
        torch_op=torch._compute_linear_combination,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
