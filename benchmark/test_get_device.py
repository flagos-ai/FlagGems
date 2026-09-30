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

"""Benchmark for ``aten::get_device``.

The operator reads the operand's placement metadata on the host and returns a
Python int, so its cost is dispatch plus an attribute lookup. The shared default
shape grid, the comprehensive extras and any caller shape file are therefore all
kept unchanged: allocation happens outside the timed region and no element is
ever read, so the payload is an uninitialized buffer rather than a filled one.

``torch_op`` is the perf reference and ``gems_op`` the FlagGems candidate; both
share the identical single-tensor call ``op(input)``.
"""

import pytest
import torch

import flag_gems

from . import base, consts

_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.complex128: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]

# A metadata read accepts every dtype group the suite exposes.
BENCH_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + _FP8_DTYPES
    )
)

BENCH_DTYPES = [dtype for dtype in BENCH_DTYPES if _DTYPE_FLAGS.get(dtype, True)]


def _case_fn(shape, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    # Contents are never read, so the whole payload is an empty buffer.
    return torch.empty(plan.builder_args[0], dtype=dtype, device=device), {}


class GetDeviceBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return super().set_more_shapes() + [
            (1,),
            (256,),
            (1024, 1024),
            (20, 320, 15),
            (16, 128, 64, 60),
        ]


@pytest.mark.get_device
def test_get_device():
    bench = GetDeviceBenchmark(
        op_name="get_device",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.get_device,
        gems_op=getattr(flag_gems, "get_device", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
