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

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# The shared default / comprehensive shape grids stay intact (a caller shape
# file still wins, because set_shapes is not overridden); set_more_shapes only
# appends the rank and empty-size boundaries of a metadata swap, with no size
# filter or cap. swapdims_ reads nothing but sizes and strides, so inputs are
# torch.empty and the values never need generating.
BOUNDARY_SHAPES = [
    (),  # rank-0: a single valid axis
    (1,),  # rank-1: the only pair available is a no-op
    (0, 3),  # empty sizes still swap together with their strides
    (3, 0),
    (16, 7, 57, 32, 29),  # highest rank in the spec grid
]

BENCH_DTYPES = [
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
]


def _dims_for(shape):
    # A rank-0/1 tensor has one axis, so dim1 = -1 addresses the same axis as
    # dim0 and the swap is a no-op there.
    return 0, len(shape) - 1


def _case_fn(shape, dtype):
    del dtype
    dim0, dim1 = _dims_for(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"dim0": dim0, "dim1": dim1},
        builder_args=(shape, dim0, dim1),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, dim0, dim1 = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, dim0, dim1


class SwapDimsBenchmark(OperatorBenchmark):
    def set_more_shapes(self):
        return super().set_more_shapes() + BOUNDARY_SHAPES


@pytest.mark.swapdims_
def test_swapdims_():
    bench = SwapDimsBenchmark(
        op_name="swapdims_",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.swapdims_.default,
        gems_op=getattr(flag_gems, "swapdims_", None),
        dtypes=[
            dtype
            for dtype in BENCH_DTYPES + [torch.float64]
            if _DTYPE_FLAGS.get(dtype, True)
        ],
        is_inplace=True,
        fresh_inputs=True,
    )
    bench.run()
