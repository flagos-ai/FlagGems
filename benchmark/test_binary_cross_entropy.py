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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    reductions = (1,) if base.Config.bench_level != consts.BenchLevel.COMPREHENSIVE else (0, 1, 2)
    for reduction in reductions:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "target": list(dims)},
            params={"weighted": False, "reduction": reduction},
            builder_args=(dims, False, reduction),
        )
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "target": list(dims), "weight": list(dims)},
            params={"weighted": True, "reduction": 1},
            builder_args=(dims, True, 1),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, weighted, reduction = plan.builder_args
    torch.manual_seed(0)
    inp = torch.sigmoid(utils.generate_tensor_input(shape, dtype, device))
    raw_target = torch.randint(0, 2, shape, device=device)
    # Keep the original GPU RNG sequence while converting binary targets exactly.
    target = (
        raw_target.cpu().to(dtype).to(device)
        if flag_gems.vendor_name == "enflame"
        else raw_target.to(dtype)
    )
    weight = torch.rand(shape, dtype=dtype, device=device) if weighted else None
    return inp, target, weight, reduction


@pytest.mark.binary_cross_entropy
def test_binary_cross_entropy():
    bench = base.GenericBenchmark2DOnly(
        op_name="binary_cross_entropy",
        torch_op=torch.ops.aten.binary_cross_entropy,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def _out_case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    reductions = (
        (1,)
        if base.Config.bench_level != consts.BenchLevel.COMPREHENSIVE
        else (0, 1, 2)
    )
    for reduction in reductions:
        out_shape = dims if reduction == 0 else ()
        yield base.BenchmarkCasePlan(
            shape={
                "self": list(dims),
                "target": list(dims),
                "out": list(out_shape),
            },
            params={"weighted": False, "reduction": reduction},
            builder_args=(dims, out_shape, reduction),
        )


def _out_build_inputs_fn(plan, dtype, device):
    shape, out_shape, reduction = plan.builder_args
    torch.manual_seed(0)
    inp = torch.sigmoid(utils.generate_tensor_input(shape, dtype, device))
    target = torch.randint(0, 2, shape, device=device).to(dtype)
    out = torch.empty(out_shape, dtype=dtype, device=device)
    return inp, target, None, reduction, {"out": out}


@pytest.mark.binary_cross_entropy_out
def test_binary_cross_entropy_out():
    bench = OperatorBenchmark(
        op_name="binary_cross_entropy_out",
        torch_op=torch.ops.aten.binary_cross_entropy.out,
        case_fn=_out_case_fn,
        build_inputs_fn=_out_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
