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

# aten::retain_grad(Tensor(a!) self) -> ()
#
# The operator moves no data: it returns None and flips a host-side autograd
# flag, so the timed work is the dispatch plus the flag write. It is only
# defined for a tensor that requires grad, and a leaf is accepted as a silent
# no-op, so a leaf case is as valid a workload as a non-leaf one.

# The three input states a caller can hand to the operator.
_KINDS = ("leaf", "nonleaf", "retained")

# Extra shapes appended on top of the shared default/comprehensive grid (and of
# any caller shape file) so the state change is measured over pointwise,
# reduction-shaped and multi-dimensional buffers.
RETAIN_GRAD_EXTRA_SHAPES = [
    (64, 64),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
]

# The flag is the same static device property the correctness suite reads;
# bfloat16 and the 64-bit, complex and fp8 entries are the only differentiable
# dtypes a backend can lack.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    return flag_name is None or bool(getattr(flag_gems.runtime.device, flag_name))


BENCH_DTYPES = [
    dtype
    for dtype in list(consts.FLOAT_DTYPES)
    + [
        torch.float64,
        torch.complex64,
        torch.complex128,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ]
    if _dtype_supported(dtype)
]


def _case_fn(shape, dtype):
    del dtype
    for kind in _KINDS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"kind": kind},
            builder_args=(shape, kind),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, kind = plan.builder_args
    # The values are irrelevant to the flag, so an empty buffer is enough and
    # its build cost stays out of the measurement.
    leaf = torch.empty(shape, dtype=dtype, device=device).requires_grad_(True)
    if kind == "leaf":
        return leaf, {}
    # A differentiable view keeps the tensor handed to the operator a non-leaf.
    inp = leaf.view(shape)
    if kind == "retained":
        torch.ops.aten.retain_grad(inp)
    return inp, {}


class RetainGradBenchmark(OperatorBenchmark):
    DEFAULT_SHAPE_DESC = "shape"

    def set_shapes(self, shape_file_path=None):
        # A caller shape file keeps its requested workloads; the extras below are
        # appended on top of whatever grid the shared configuration selected.
        super().set_shapes(shape_file_path)
        for shape in RETAIN_GRAD_EXTRA_SHAPES:
            if shape not in self.shapes:
                self.shapes.append(shape)


@pytest.mark.retain_grad
def test_retain_grad():
    bench = RetainGradBenchmark(
        op_name="retain_grad",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.retain_grad,
        gems_op=getattr(flag_gems, "retain_grad", None),
        dtypes=BENCH_DTYPES,
        # Keep the non-leaf graph: fresh_inputs detaches operands and would
        # turn this into the leaf no-op. Repeated timing calls are idempotent;
        # reference-only executes each listed initial state exactly once.
    )
    bench.run()
