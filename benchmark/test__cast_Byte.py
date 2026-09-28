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
#
# Benchmark for aten::_cast_Byte over the unary pointwise shape family.

import numbers

import pytest
import torch

import flag_gems

from . import base, consts

# Static backend capability flags: unsupported dtypes are filtered here so the
# same list drives case listing and execution.
_DTYPE_FLAGS = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.float64: flag_gems.runtime.device.support_fp64,
    torch.int64: flag_gems.runtime.device.support_int64,
    torch.float8_e4m3fn: flag_gems.runtime.device.support_fp8,
    torch.float8_e5m2: flag_gems.runtime.device.support_fp8,
}

# Inputs whose cast to uint8 is timed: the pointwise float family plus the
# integral casts, including the uint8 -> uint8 identity path.
BENCH_DTYPES = [
    dtype
    for dtype in (
        *consts.FLOAT_DTYPES,
        torch.uint8,
        torch.int8,
        torch.int32,
        torch.int64,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    )
    if _DTYPE_FLAGS.get(dtype, True)
]

# None omits the argument, so that case measures the schema default.
NON_BLOCKING_CASES = [False, True, None]


def _validate_shape(shape):
    # Shapes are resolved by the framework from the defaults, the core shape
    # list or a caller-supplied shape file, so malformed custom metadata is
    # rejected here at listing time, before any tensor is allocated. A scalar
    # shape and zero extents stay valid.
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, numbers.Integral):
            raise ValueError(f"non-integral shape extent {extent!r} in {shape!r}")
        if extent < 0:
            raise ValueError(f"negative shape extent {extent!r} in {shape!r}")


def _cast_input(shape, dtype, device):
    # FP8 and the integral dtypes have no randn kernel.
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=torch.float32, device=device).to(dtype)
    return torch.randint(0, 128, shape, dtype=torch.int32, device=device).to(dtype)


def _case_fn(shape, dtype):
    # One plan per resolved shape and non_blocking form; the plans carry both
    # the public metadata and the builder arguments, so listing and execution
    # never use different case lists.
    del dtype
    _validate_shape(shape)
    for non_blocking in NON_BLOCKING_CASES:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={
                "non_blocking": "omitted" if non_blocking is None else non_blocking
            },
            builder_args=(shape, non_blocking),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, non_blocking = plan.builder_args
    inp = _cast_input(shape, dtype, device)
    if non_blocking is None:
        return (inp,)
    return inp, {"non_blocking": non_blocking}


class CastByteBenchmark(base.GenericBenchmark):
    # Keep the unary pointwise benchmark's comprehensive shape scales; the
    # default and caller-supplied shape resolution stays with the framework.
    def set_more_shapes(self):
        return base.UnaryPointwiseBenchmark.set_more_shapes(self)


@pytest.mark.cast_Byte
def test_cast_byte():
    bench = CastByteBenchmark(
        op_name="_cast_Byte",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cast_Byte,
        gems_op=getattr(flag_gems, "_cast_Byte", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
