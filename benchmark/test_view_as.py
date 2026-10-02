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


import math

import pytest
import torch

import flag_gems

from . import base

# aten::view_as(self, other) reads only the shape of 'other' and returns a view of
# 'self' when that shape is stride-expressible; otherwise the native operator
# raises instead of copying. A timed case is therefore an (operand layout, target
# shape) pair. Every plan below is derived from the requested shape alone and
# carries the target it was checked against, so listing stays tensor-free and a
# replay by --case-id rebuilds exactly the plans of a normal run.
#
# 'other' is shape-only: no element of it is ever read, so it is allocated with
# torch.empty, which keeps a requested shape real without paying fixture cost.

# Dtype coverage. view_as is dtype-agnostic (only the shape of 'other' is read),
# so the list holds the required integer and FP8 types plus the useful float,
# bool and complex ones. The capability flags are the static ones the device
# layer exposes; they are read while this module is imported.
_BENCH_DTYPES = (
    [torch.float32, torch.float16]
    + ([torch.bfloat16] if flag_gems.runtime.device.support_bf16 else [])
    + ([torch.float64] if flag_gems.runtime.device.support_fp64 else [])
    + [torch.int16, torch.int32, torch.int8, torch.uint8]
    + ([torch.int64] if flag_gems.runtime.device.support_int64 else [])
    + [torch.bool, torch.complex64]
    + ([torch.complex128] if flag_gems.runtime.device.support_fp64 else [])
    + (
        [torch.float8_e4m3fn, torch.float8_e5m2]
        if flag_gems.runtime.device.support_fp8
        else []
    )
)


def _swap_last_two(shape):
    return shape[:-2] + (shape[-1], shape[-2])


def _split_last(shape):
    # Splitting the trailing extent keeps the element order, so the target stays
    # a view for a transposed operand and for a stride-0 one alike.
    if shape and shape[-1] > 1 and shape[-1] % 2 == 0:
        return shape[:-1] + (shape[-1] // 2, 2)
    return shape


def _target_for(operand_shape, layout):
    if layout == "transposed":
        return _split_last(operand_shape)
    # A contiguous operand and a fully expanded one both keep element order, so
    # the flat merge is viewable from either.
    return (math.prod(operand_shape),)


def _make_plan(operand_shape, storage, layout, expand_shape):
    target = _target_for(operand_shape, layout)
    return base.BenchmarkCasePlan(
        shape={"input": list(operand_shape), "target": list(target)},
        params={"layout": layout},
        builder_args=(storage, layout, expand_shape, target),
    )


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(int(extent) for extent in shape)
    yield _make_plan(shape, shape, "asis", None)
    if len(shape) >= 2:
        # A rank-1 operand has no pair of extents to swap.
        yield _make_plan(_swap_last_two(shape), shape, "transposed", None)
    if shape and math.prod(shape) > 1:
        # A fully expanded operand has stride 0 on every extent; expanding a
        # one-element base allocates nothing for the operand itself.
        yield _make_plan(shape, (1,) * len(shape), "expanded", shape)


def _build_inputs_fn(plan, dtype, device):
    storage, layout, expand_shape, target = plan.builder_args
    if layout == "expanded":
        inp = torch.empty(storage, dtype=dtype, device=device).expand(expand_shape)
    else:
        inp = torch.empty(storage, dtype=dtype, device=device)
        if layout == "transposed":
            inp = inp.transpose(-1, -2)
    other = torch.empty(target, dtype=dtype, device=device)
    return inp, other, {}


class ViewAsBenchmark(base.GenericBenchmark):
    """Itemised view_as layouts on top of the shared shapes and shape file."""

    def set_more_shapes(self):
        # Comprehensive level only. The shared defaults have no 0-dim, no rank-4
        # and no zero-element shape, which are the remaining boundaries of this
        # view; all of them are cheap for a metadata operator.
        return super().set_more_shapes() + [
            (),
            (1,),
            (256,),
            (2, 19, 7),
            (16, 128, 64, 60),
            (0, 3),
        ]


@pytest.mark.view_as
def test_view_as():
    bench = ViewAsBenchmark(
        op_name="view_as",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.view_as,
        gems_op=getattr(flag_gems, "view_as", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
