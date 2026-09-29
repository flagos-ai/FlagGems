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

"""Benchmark for ``aten::_reshape_copy``.

The curated rows are the copy scales -- flat inputs of 1M/4M/16M elements, a
2-D 1M-element input and the 3-D/4-D spec shapes -- plus non-contiguous
sources (transpose, strided slice) that run the copy with real strides. Every
row carries an explicit ``size`` descriptor whose element count equals the
element count of the input the operator receives.

Listing and execution share the same metadata plans: rows are validated at
import time from integers only, ``--list-cases`` allocates nothing, and
``--case-id`` replay rebuilds the identical input from the stored
``builder_args``.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# (input shape handed to the builder, input layout, size passed to the op).
# ``_logical_shape`` derives the shape the operator actually sees: a transpose
# swaps extents and a strided slice drops elements, so the descriptors below
# are written for that logical input, not for the base allocation.
_BENCH_CASES = [
    ((1 << 20,), "contiguous", (1 << 10, 1 << 10)),
    ((1 << 22,), "contiguous", (1 << 11, 1 << 11)),
    ((1 << 24,), "contiguous", (1 << 12, 1 << 12)),
    ((1024, 1024), "contiguous", (1 << 20,)),
    ((1024, 1024), "transpose", (512, 2048)),
    ((20, 320, 15), "contiguous", (20, 4800)),
    ((20, 320, 15), "transpose", (20, 320, 15)),
    ((20, 320, 15), "slice", (20, 2400)),
    ((16, 128, 64, 60), "contiguous", (16, 491520)),
]

# BF16 is benchmarked only where the backend's static capability flag says it
# is supported, and the same list drives listing and execution.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _numel(extents):
    count = 1
    for extent in extents:
        count *= extent
    return count


def _normalize_extents(extents, label):
    """Validated non-negative integer extents, checked before any product."""
    if isinstance(extents, int) and not isinstance(extents, bool):
        # A bare int is a 1-D shape; validate it through the same loop so
        # negatives are rejected here rather than reaching the operator.
        extents = (extents,)
    if not isinstance(extents, (list, tuple)):
        raise TypeError(f"{label} must be a list or tuple of ints, got {extents!r}.")
    for extent in extents:
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise TypeError(f"{label} must hold ints, got {extent!r}.")
        if extent < 0:
            raise ValueError(f"{label} must hold non-negative extents, got {extent}.")
    return tuple(extents)


def _logical_shape(base_shape, layout):
    """Shape of the tensor the operator receives for one benchmark row."""
    if layout == "contiguous":
        return base_shape
    if len(base_shape) < 2:
        raise ValueError(f"Layout {layout!r} needs rank >= 2, got {base_shape}.")
    if layout == "transpose":
        return (base_shape[1], base_shape[0]) + base_shape[2:]
    if layout == "slice":
        return (base_shape[0], (base_shape[1] + 1) // 2) + base_shape[2:]
    raise ValueError(f"Unknown benchmark input layout {layout!r}.")


def _validated_case(base_shape, layout, size):
    """Normalize one curated row and reject a ``size`` that cannot apply to it."""
    base_shape = _normalize_extents(base_shape, "Benchmark input shape")
    size = _normalize_extents(size, "Benchmark size")
    expected = _numel(_logical_shape(base_shape, layout))
    if _numel(size) != expected:
        raise ValueError(
            f"{layout} input {base_shape} holds {expected} elements but size "
            f"{size} holds {_numel(size)}."
        )
    return base_shape, size


_PLANS_BY_SHAPE = {}
for _row_shape, _row_layout, _row_size in _BENCH_CASES:
    _base, _size = _validated_case(_row_shape, _row_layout, _row_size)
    _PLANS_BY_SHAPE.setdefault(_base, []).append((_row_layout, _size))

_DEFAULT_SHAPES = list(_PLANS_BY_SHAPE)


def _plans_for(shape):
    """Validated base shape and its (layout, size) descriptors.

    Curated shapes keep their rows; a shape supplied only through a custom
    ``--shape-file`` (which may also be a scalar, a bare integer or have
    zero-extent dimensions) falls back to copying a contiguous input into a
    flat tensor with the same element count, a valid ``size`` for any shape.
    """
    base_shape = _normalize_extents(shape, "Benchmark input shape")
    plans = _PLANS_BY_SHAPE.get(base_shape)
    if plans is None:
        return base_shape, [("contiguous", (_numel(base_shape),))]
    return base_shape, plans


def _case_fn(shape, dtype):
    del dtype
    base_shape, plans = _plans_for(shape)
    for layout, size in plans:
        yield base.BenchmarkCasePlan(
            shape={
                "input": list(_logical_shape(base_shape, layout)),
                "input_layout": layout,
            },
            params={"size": list(size)},
            builder_args=(base_shape, layout, size),
        )


def _build_inputs_fn(plan, dtype, device):
    base_shape, layout, size = plan.builder_args
    inp = utils.generate_tensor_input(base_shape, dtype, device)
    if layout == "transpose":
        inp = inp.transpose(0, 1)
    elif layout == "slice":
        inp = inp[:, ::2]
    return inp, list(size)


class ReshapeCopyBenchmark(OperatorBenchmark):
    """Curated copy scales are this operator's default shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)


@pytest.mark.reshape_copy
def test__reshape_copy():
    bench = ReshapeCopyBenchmark(
        op_name="_reshape_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._reshape_copy,
        gems_op=getattr(flag_gems, "_reshape_copy", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
