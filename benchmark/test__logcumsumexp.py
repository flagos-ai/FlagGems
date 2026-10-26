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


def _logcumsumexp_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    yield inp, 1


def _logcumsumexp_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims)}, params={"dim": 1}, builder_args=(dims,)
    )


def _logcumsumexp_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    return next(_logcumsumexp_input_fn(shape, dtype, device))


@pytest.mark.underscore_logcumsumexp
def test__logcumsumexp():
    bench = base.GenericBenchmark2DOnly(
        op_name="_logcumsumexp",
        case_fn=_logcumsumexp_case_fn,
        build_inputs_fn=_logcumsumexp_build_inputs_fn,
        torch_op=torch._logcumsumexp,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _logcumsumexp_out_input_fn(shape, cur_dtype, device):
    inp = torch.randn(shape, dtype=cur_dtype, device=device)
    out = torch.empty_like(inp)
    yield inp, 1, {"out": out}


@pytest.mark.underscore_logcumsumexp_out
def test__logcumsumexp_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="_logcumsumexp_out",
        input_fn=_logcumsumexp_out_input_fn,
        torch_op=torch.ops.aten._logcumsumexp.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
