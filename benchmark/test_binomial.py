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


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"count": list(dims), "prob": list(dims)},
        params={"generator": None},
        builder_args=(dims,),
    )


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"count": list(dims), "prob": list(dims), "out": list(dims)},
        params={"generator": None},
        builder_args=(dims,),
    )


def _build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    count = torch.randint(1, 100, shape, device=device).to(dtype)
    prob = torch.rand(shape, dtype=dtype, device=device)
    return count, prob


def _out_build_inputs_fn(plan, dtype, device):
    if flag_gems.vendor_name == "ascend" and dtype in (torch.float16, torch.bfloat16):
        pytest.skip(
            "Ascend binomial.out falls back to CPU, "
            "which does not support Half/BFloat16"
        )
    (shape,) = plan.builder_args
    torch.manual_seed(0)
    count = torch.randint(1, 100, shape, device=device).to(dtype)
    prob = torch.rand(shape, dtype=dtype, device=device)
    out = torch.empty_like(count)
    return count, prob, {"out": out}


@pytest.mark.binomial
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_binomial():
    bench = base.GenericBenchmark2DOnly(
        op_name="binomial",
        torch_op=torch.binomial,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.binomial_out
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_binomial_out():
    bench = base.GenericBenchmark2DOnly(
        op_name="binomial_out",
        torch_op=torch.ops.aten.binomial.out,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
