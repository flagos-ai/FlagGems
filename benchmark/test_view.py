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

"""Benchmark for ``aten::view``.

Both overloads of ``torch.ops.aten.view`` are listed and measured through the
single public candidate ``flag_gems.view``: the ``size`` overload on a
contiguous input, and the ``dtype`` overload reinterpreting to ``torch.uint8``
-- a 1-byte target element size divides every trailing dimension, so that
overload adds no shape constraint of its own.

The shapes stay the shared ones (the ``core_shapes.yaml`` class-key entry, the
framework's comprehensive extras and the higher-rank boundaries they lack), so a
caller-supplied shape file keeps working exactly as for every other operator. A
view reads no element, so the builders use ``torch.empty``, and the plans keep
their torch objects in private ``builder_args`` so ``--list-cases`` allocates no
tensor and calls no operator.
"""

import math

import pytest
import torch

import flag_gems

from . import base, consts

# Every dtype the native operator can view. Case selection by level never
# shrinks this list.
_VIEW_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [dtype for dtype in consts.FP8_DTYPES if dtype is not None]
    )
)

# The reinterpret target: torch.uint8's 1-byte element size divides every
# source element size, so the dtype overload needs no shape filtering.
_DTYPE_TARGET = torch.uint8

# Numel-preserving rearrangements of the shared and boundary shapes; an
# unlisted shape from a caller-supplied shape file falls back to a flatten,
# which is always a valid view of a contiguous input.
_TARGET_SHAPES = {
    (1073741824,): (1024, 1024, 1024),
    (64, 64): (8, 512),
    (4096, 4096): (2048, 8192),
    (64, 512, 512): (512, 512, 64),
    (1024, 1024, 1024): (32768, 32768),
    (268435456,): (16384, 16384),
    (10000, 1): (10000,),
    (10000, 256): (400, 6400),
    (10000, 65536): (10000, 256, 256),
    (100, 1, 100): (100, 100),
    (100, 256, 100): (100, 100, 256),
    (100, 65536, 100): (10000, 65536),
    (16, 128, 64, 60): (16, 128, 60, 64),
    (16, 7, 57, 32, 29): (16, 7, 57, 928),
}


def _size_target(shape):
    return _TARGET_SHAPES.get(shape, (math.prod(shape),))


def _case_fn(shape, dtype):
    del dtype
    shape = tuple(shape)
    target = _size_target(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"overload": "size", "target": list(target)},
        builder_args=(shape, "size", target),
    )
    if shape:
        # The dtype overload reinterprets the trailing dimension, so a 0-d
        # input has nothing to reinterpret and yields the size plan only.
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"overload": "dtype", "target": str(_DTYPE_TARGET)},
            builder_args=(shape, "dtype", _DTYPE_TARGET),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, overload, target = plan.builder_args
    # A view moves no element, so uninitialized storage is enough; the
    # allocation happens before timing starts.
    inp = torch.empty(shape, dtype=dtype, device=device)
    # Flat positional arguments keep the call identical for torch_op and
    # gems_op: op(input, (3, 8)) for the size overload, op(input, torch.uint8)
    # for the dtype overload.
    if overload == "size":
        return inp, tuple(target)
    return inp, target


class ViewBenchmark(base.GenericBenchmark):
    """Shared shape grids plus the higher-rank boundaries they lack."""

    def set_more_shapes(self):
        return super().set_more_shapes() + [(16, 128, 64, 60), (16, 7, 57, 32, 29)]


@pytest.mark.view
def test_view():
    bench = ViewBenchmark(
        op_name="view",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.view,
        gems_op=getattr(flag_gems, "view", None),
        dtypes=_VIEW_DTYPES,
    )
    bench.run()
