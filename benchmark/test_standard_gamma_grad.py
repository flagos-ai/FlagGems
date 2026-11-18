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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"self": list(dims), "output": list(dims)},
        params={"domain": "positive"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    if base.vendor_name == "enflame" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(f"Enflame native standard_gamma_grad does not support {dtype}")
    if base.vendor_name == "kunlunxin" and dtype == torch.bfloat16:
        pytest.skip(f"Kunlunxin native standard_gamma_grad does not support {dtype}")
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    alpha = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.5
    output = torch.rand(shape, dtype=dtype, device=device) * 4.0 + 0.5
    return alpha, output


class StandardGammaGradBenchmark(base.BinaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    def set_shapes(self, shape_file_path=None):
        base.Benchmark.set_shapes(self, shape_file_path)

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.standard_gamma_grad
def test_standard_gamma_grad():
    bench = StandardGammaGradBenchmark(
        op_name="standard_gamma_grad",
        torch_op=torch._standard_gamma_grad,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
