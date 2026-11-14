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


class _HermiteBenchmark(base.BinaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    set_shapes = base.Benchmark.set_shapes
    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"x": list(dims), "n": list(dims)},
        params={"degree_kind": "tensor", "degree_max": 9},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    x = base.generate_tensor_input(shape, dtype, device)
    degree = torch.randint(0, 10, shape, device=device).to(dtype)
    return x, degree


@pytest.mark.special_hermite_polynomial_h
def test_special_hermite_polynomial_h():
    bench = _HermiteBenchmark(
        op_name="special_hermite_polynomial_h",
        torch_op=torch.special.hermite_polynomial_h,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.float32],
    )
    bench.run()
