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

# Optional dtypes are gated on static backend capability flags; every baseline
# metadata dtype stays unconditional and no case is filtered at run time.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag_name = _DTYPE_CAPABILITY.get(dtype)
    if flag_name is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag_name, False))


BENCH_DTYPES = [
    dtype
    for dtype in [
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
    if _dtype_supported(dtype)
]

# Rank boundaries appended to the shared default/comprehensive grid; the
# operator accepts every rank, so no shared or caller shape is filtered.
MOVE_AXIS_BOUNDARY_SHAPES = [
    (1024,),
    (1, 4096),
    (128, 512, 256),
    (4, 8, 16, 32, 64),
]


def _case_fn(shape, dtype):
    del dtype
    rank = len(shape)
    axis_pairs = [(0, 0)] if rank <= 1 else [(0, rank - 1), (-1, 0)]
    for source, destination in axis_pairs:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"source": source, "destination": destination},
            builder_args=(shape, source, destination),
        )
    if rank >= 2:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"source": [0, 1], "destination": [1, 0]},
            builder_args=(shape, [0, 1], [1, 0]),
        )
    # int[] no-op pair, valid for every rank including 0-D.
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"source": [], "destination": []},
        builder_args=(shape, [], []),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, source, destination = plan.builder_args
    # The view op and the timing harness never read element values, so an
    # uninitialized allocation keeps the shared shapes cheap while preserving
    # the declared shape. Flat positional args plus a trailing kwargs dict.
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, source, destination, {}


class MoveAxisBenchmark(base.GenericBenchmark):
    # Append the rank boundaries to the superclass shape set instead of
    # replacing it, so the shared default/comprehensive shapes and any caller
    # --shape_file entry stay in effect.
    def set_more_shapes(self):
        return list(super().set_more_shapes()) + MOVE_AXIS_BOUNDARY_SHAPES


@pytest.mark.moveaxis
def test_moveaxis():
    bench = MoveAxisBenchmark(
        op_name="moveaxis",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.moveaxis,
        gems_op=getattr(flag_gems, "moveaxis", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
