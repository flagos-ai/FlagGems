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
        shape={"self": list(dims), "other": list(dims)},
        params={"exponent_dtype": "int32"},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    return inp, exponent


class LdexpBenchmark(base.BinaryPointwiseBenchmark, OperatorBenchmark):
    def supports_cases(self):
        return True

    get_case_iter = OperatorBenchmark.get_case_iter
    build_inputs = OperatorBenchmark.build_inputs


@pytest.mark.ldexp
def test_ldexp():
    bench = LdexpBenchmark(
        op_name="ldexp",
        torch_op=torch.ops.aten.ldexp.Tensor,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _out_build_inputs_fn(plan, dtype, device):
    inp, exponent = _build_inputs_fn(plan, dtype, device)
    out = torch.empty_like(inp)
    return inp, exponent, {"out": out}


@pytest.mark.ldexp_out
def test_ldexp_out():
    LdexpBenchmark(
        op_name="ldexp_out",
        torch_op=torch.ops.aten.ldexp.out,
        case_fn=_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    ).run()
