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

"""Benchmark for ``aten::empty_strided``.

A case is a (size, stride, dtype) triple. The shared default/comprehensive
shapes provide the sizes and the extra stride layouts are derived from each of
them. Planning only touches metadata, so ``--list-cases`` and ``--case-id``
replay allocate nothing.
"""

import pytest
import torch

import flag_gems

from . import base, consts

EMPTY_STRIDED_LAYOUTS = [
    ((), ()),
    ((256,), (1,)),
    ((1024, 1024), (1024, 1)),
    ((1024, 1024), (1, 1024)),
    ((20, 320, 15), (4800, 15, 1)),
    ((16, 128, 64, 60), (491520, 3840, 60, 1)),
    ((512, 512), (1024, 2)),
    ((7, 13, 5), (65, 5, 1)),
    ((0, 3), (3, 1)),
    ((2, 3), (1, 1)),
    ((4 * 1024 * 1024,), (1,)),
]

_LAYOUTS = ("contiguous", "reversed", "padded", "overlap")


def _contiguous_strides(size):
    strides = [0] * len(size)
    running = 1
    for axis in range(len(size) - 1, -1, -1):
        strides[axis] = running
        running *= size[axis]
    return tuple(strides)


def _strides_for(size, layout):
    contiguous = _contiguous_strides(size)
    if layout == "reversed":
        return tuple(reversed(contiguous))
    if layout == "padded":
        # One padding element per axis: a storage larger than numel.
        return tuple(step * 2 for step in contiguous)
    if layout == "overlap":
        # Every axis advances by one element: an overlapping, smaller storage.
        return tuple(1 for _ in contiguous)
    return contiguous


# Every element type the allocator accepts on this backend (native probe: all
# allocate). consts.FP8_DTYPES is not iterated directly because it holds None on
# backends without fp8 support.
_ALLOCATOR_DTYPES = list(
    dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float64, torch.complex32, torch.complex128]
        + [
            dtype
            for dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if flag_gems.runtime.device.support_fp8
        ]
    )
)


def _case_fn(shape, dtype):
    del dtype
    if len(shape) == 2 and isinstance(shape[0], (tuple, list)):
        size, stride = map(tuple, shape)
        yield base.BenchmarkCasePlan(
            shape={"size": list(size), "stride": list(stride)},
            params={"layout": "explicit"},
            builder_args=(size, stride),
        )
        return
    size = tuple(shape)
    seen = set()
    for layout in _LAYOUTS:
        stride = _strides_for(size, layout)
        if stride in seen:
            continue
        seen.add(stride)
        yield base.BenchmarkCasePlan(
            shape={"size": list(size), "stride": list(stride)},
            params={"layout": layout},
            builder_args=(size, stride),
        )


def _build_inputs_fn(plan, dtype, device):
    size, stride = plan.builder_args
    # Positional size/stride plus keyword dtype/device.
    return list(size), list(stride), {"dtype": dtype, "device": device}


class EmptyStridedBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = list(self.shapes) + EMPTY_STRIDED_LAYOUTS


@pytest.mark.empty_strided
def test_empty_strided():
    bench = EmptyStridedBenchmark(
        op_name="empty_strided",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.empty_strided,
        gems_op=getattr(flag_gems, "empty_strided", None),
        dtypes=_ALLOCATOR_DTYPES,
    )
    bench.run()
