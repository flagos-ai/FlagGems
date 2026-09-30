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

# aten::output_nr(Tensor self) -> int is an O(1) read of autograd metadata that
# returns a Python int, so no element is ever read and the fixture stays
# uninitialized; allocation happens outside timing. The cases differ only in the
# graph state of the queried tensor: a leaf, a metadata view that starts a
# single-output node, and one slot of a two-output split node, which is the only
# state that reports a nonzero index.

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


# Every dtype the operator accepts. Integer and boolean leaves never carry a graph,
# which is itself one of the benchmarked states.
BENCH_DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.complex64,
        torch.float64,
    )
    if _dtype_supported(dtype)
]


def _case_fn(shape, dtype):
    shape = tuple(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"state": "leaf"},
        builder_args=(shape, "leaf"),
    )
    if not (dtype.is_floating_point or dtype.is_complex):
        return
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"state": "grad_view"},
        builder_args=(shape, "grad_view"),
    )
    if len(shape) >= 1 and shape[0] >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"state": "grad_split_slot"},
            builder_args=(shape, "grad_split_slot"),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, state = plan.builder_args
    if state == "leaf":
        return torch.empty(shape, dtype=dtype, device=device), {}
    leaf = torch.empty(shape, dtype=dtype, device=device).requires_grad_(True)
    if state == "grad_view":
        # A metadata view records a single-output node without computing on any
        # element, so the fp8 dtypes stay covered even though this backend has no
        # arithmetic kernel for them.
        return leaf.view(shape), {}
    # One split point yields exactly two views, so a nonzero slot is measured
    # without creating one tensor per element of a billion-element dimension.
    return torch.split(leaf, [shape[0] // 2, shape[0] - shape[0] // 2], dim=0)[1], {}


@pytest.mark.output_nr
def test_output_nr():
    bench = OperatorBenchmark(
        op_name="output_nr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.output_nr,
        gems_op=getattr(flag_gems, "output_nr", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
