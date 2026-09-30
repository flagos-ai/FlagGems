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

# swapaxes is a pure metadata view: no value is read, so cases allocate with
# torch.empty and every dtype the native op accepts can be timed.
_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

BENCH_DTYPES = [
    dtype
    for dtype in (
        *consts.FLOAT_DTYPES,
        torch.float64,
        *consts.EXTRA_INT_DTYPES,
        *consts.INT_DTYPES,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.bool,
        torch.complex64,
    )
    if _DTYPE_FLAGS.get(dtype, True)
]


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    ndim = len(shape)
    # A 0-D or 1-D tensor only accepts the identity swap; wider tensors also
    # exercise negative-axis normalization and the equal-axis shortcut.
    axis_pairs = [(0, 0)] if ndim < 2 else [(0, ndim - 1), (-1, 0), (0, 0)]
    for axis0, axis1 in axis_pairs:
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"axis0": axis0, "axis1": axis1},
            builder_args=(shape, axis0, axis1),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, axis0, axis1 = plan.builder_args
    inp = torch.empty(shape, dtype=dtype, device=device)
    return inp, axis0, axis1


class SwapaxesBenchmark(base.GenericBenchmark):
    """Shared shape grid plus the rank boundaries that grid does not contain."""

    def set_more_shapes(self):
        return super().set_more_shapes() + [
            (),
            (16, 128, 64, 60),
            (16, 7, 57, 32, 29),
        ]


@pytest.mark.swapaxes
def test_swapaxes():
    bench = SwapaxesBenchmark(
        op_name="swapaxes",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.swapaxes,
        gems_op=getattr(flag_gems, "swapaxes", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
