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

# aten::mT returns a zero-copy view and never reads or writes an element, so the
# inputs come from torch.empty (allocation stays outside the measured view path)
# and a single input stays valid across iterations: no fresh-input clone is
# needed even though the result aliases the input storage.

_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    return flag_name is None or bool(
        getattr(flag_gems.runtime.device, flag_name, False)
    )


# Every valid metadata dtype: fp16 / fp32 / bf16, the required integers, both
# FP8 types and the useful bool / complex64 additions.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64]
        + [torch.float8_e4m3fn, torch.float8_e5m2]
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
    )
    if _dtype_supported(dtype)
]

# Rank >= 2 boundary shapes appended to the shared grid: minimal matrix, empty
# operand, batch of matrices and the spec's 5-D shape.
MT_BOUNDARY_SHAPES = [
    (),
    (1, 1),
    (0, 3),
    (2, 3, 4),
    (16, 7, 57, 32, 29),
]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    return torch.empty(shape, dtype=dtype, device=device), {}


class MTBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Native rejects rank 1; the deprecated scalar path remains valid.
        self.shapes = list(
            dict.fromkeys(
                [tuple(shape) for shape in self.shapes if len(shape) != 1]
                + MT_BOUNDARY_SHAPES
            )
        )


@pytest.mark.mT
def test_mT():
    bench = MTBenchmark(
        op_name="mT",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mT,
        gems_op=getattr(flag_gems, "mT", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
