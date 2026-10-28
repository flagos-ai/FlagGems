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

from . import base, consts
from .generated_operator_utils import OperatorBenchmark


class SoftplusBackwardBenchmark(base.UnaryPointwiseBenchmark, OperatorBenchmark):
    """Keep the unary shape policy while exposing the backward operator ABI."""

    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


def _backward_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"grad_output": list(dims), "self": list(dims)},
        params={"beta": 1.0, "threshold": 20.0},
        builder_args=(dims,),
    )


def _backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    inp = torch.randn(shape, dtype=dtype, device=device).clone().requires_grad_(True)
    return torch.randn_like(inp), inp, 1.0, 20.0


@pytest.mark.softplus
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_softplus():
    bench = base.UnaryPointwiseBenchmark(
        op_name="softplus",
        torch_op=torch.nn.functional.softplus,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.softplus_backward
def test_softplus_backward():
    bench = SoftplusBackwardBenchmark(
        op_name="softplus_backward",
        torch_op=torch.ops.aten.softplus_backward,
        case_fn=_backward_case_fn,
        build_inputs_fn=_backward_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
