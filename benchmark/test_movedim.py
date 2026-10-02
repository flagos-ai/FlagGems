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

# Static capability flags, read at import: no tensor is allocated and no
# operator is called at collection time.
_DTYPE_CAPABILITY = {
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


# movedim is dtype-agnostic metadata surgery, so every dtype the native op
# accepts is benchmarked, including the required integers and FP8 types.
BENCH_DTYPES = [
    dtype
    for dtype in (
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [fp8_dtype for fp8_dtype in consts.FP8_DTYPES if fp8_dtype is not None]
    )
    if _dtype_supported(dtype)
]

# The shared grid is used unchanged: core_shapes.yaml has no movedim and no
# GenericBenchmark entry, so consts.DEFAULT_SHAPES applies and any user shape
# file keeps working. movedim accepts any rank, but the shared grid stops at
# rank 3, so one wider-rank shape is merged on top in COMPREHENSIVE mode.
MOVEDIM_MORE_SHAPES = [(2, 19, 7), (8, 16, 32, 64)]


def _moves(rank):
    # Rank 0 and 1 admit only the identity move.
    if rank < 2:
        return ((0, 0),)
    last = rank - 1
    return ((0, last), ([0, last], [last, 0]))


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    for source, destination in _moves(len(shape)):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"source": str(source), "destination": str(destination)},
            builder_args=(shape, source, destination),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, source, destination = plan.builder_args
    # movedim reads no element, so an uninitialized buffer carries the workload
    # while keeping the fixture cost out of the measured region. The list form
    # stays a single positional argument for unpack_to_args_kwargs.
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, source, destination, {}


class MovedimBenchmark(base.GenericBenchmark):
    """Shared shape grid plus a wider-rank shape."""

    def set_more_shapes(self):
        return list(super().set_more_shapes()) + MOVEDIM_MORE_SHAPES


@pytest.mark.movedim
def test_movedim():
    bench = MovedimBenchmark(
        op_name="movedim",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.movedim,
        gems_op=getattr(flag_gems, "movedim", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
