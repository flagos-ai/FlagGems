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

"""Benchmark for aten::new_empty_strided.

The operator allocates storage, so the timed work is allocation plus dispatch. Every
requested size is timed in each layout the native operator accepts: dense, spaced and a
zero outer stride (valid for every non-scalar rank, including rank 1 and zero-extent
requests). The explicit sizes below are appended to the shared core_shapes /
DEFAULT_SHAPES grid (and to the COMPREHENSIVE extension) instead of replacing it, and a
caller-provided shape file keeps its own shapes as well.

`self` carries dtype/device metadata only, so it is built with torch.empty and its
payload is never read. torch_op is the ATen reference and gems_op the FlagGems
candidate; both receive the same positional (self, size, stride) arguments. The .out
overload cannot share that call form, so it is covered by the correctness file.

The plans are pure metadata, so --list-cases works without a candidate and allocates no
tensor.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

BENCH_EXTRA_SIZES = [
    (),
    (0,),
    (2, 0, 3),
    (256,),
    (2, 19, 7),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64),
    (16, 7, 57, 32),
    (4096, 4096),
]

_DTYPE_CAPABILITY = {
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    # Static capability flags, read at import time: nothing is probed or skipped at run
    # time. A dtype outside the map is a baseline type the backend always handles.
    flag = _DTYPE_CAPABILITY.get(dtype)
    return (
        True if flag is None else bool(getattr(flag_gems.runtime.device, flag, False))
    )


# Every statically supported allocation dtype, deduplicated in order.
BENCH_DTYPES = [
    dtype
    for dtype in dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [
            torch.float64,
            torch.complex128,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]
    )
    if _dtype_supported(dtype)
]


def _contiguous_stride(size):
    running = 1
    strides = []
    for dim in reversed(size):
        strides.append(running)
        running *= dim
    return tuple(reversed(strides))


def _stride_for(size, layout):
    dense = _contiguous_stride(size)
    if layout == "contiguous":
        return dense
    if layout == "spaced":
        return tuple(step * 2 for step in dense)
    if layout == "zero_stride":
        # A 0-dim tensor's stride is the empty tuple; a zero outer stride is valid for
        # every non-scalar rank, rank 1 and zero-extent requests included.
        return () if not size else (0,) + dense[1:]
    raise ValueError("unsupported stride layout " + repr(layout))


def _layouts_for(size):
    if not size:
        return ["contiguous"]
    return ["contiguous", "spaced", "zero_stride"]


def _case_fn(shape, dtype):
    del dtype
    size = tuple(shape)
    for layout in _layouts_for(size):
        stride = _stride_for(size, layout)
        yield base.BenchmarkCasePlan(
            shape={"self": list(size)},
            params={"size": list(size), "stride": list(stride), "layout": layout},
            builder_args=(size, stride),
        )


def _build_inputs_fn(plan, dtype, device):
    size, stride = plan.builder_args
    # `self` supplies dtype/device metadata only: its payload is never read, and the
    # uninitialized result payload is never compared either. unpack_to_args_kwargs
    # turns the returned flat arguments into op(self, size, stride).
    inp = torch.empty(size, dtype=dtype, device=device)
    return inp, list(size), list(stride), {}


class NewEmptyStridedBenchmark(OperatorBenchmark):
    """Two-phase benchmark that keeps the shared shape grid."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Append the explicit allocation sizes instead of replacing the shared grid or a
        # caller-provided shape file. A shape file may deliver plain lists, so normalize
        # each entry before merging.
        shared = [tuple(shape) for shape in (self.shapes or ())]
        self.shapes = list(dict.fromkeys(shared + BENCH_EXTRA_SIZES))


@pytest.mark.new_empty_strided
def test_new_empty_strided():
    bench = NewEmptyStridedBenchmark(
        op_name="new_empty_strided",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.new_empty_strided,
        gems_op=getattr(flag_gems, "new_empty_strided", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
