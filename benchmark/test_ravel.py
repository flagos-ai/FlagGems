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
import operator

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# ravel is a zero-copy flatten when the tensor can be flattened in place and a
# contiguous copy otherwise, so every scale is measured in both geometries: the
# contiguous case covers the view path, the column-major one the copy path.
RAVEL_SHAPES = [
    (1024, 1024),
    (4096, 4096),
    (20, 320, 15),
    (16, 128, 64, 60),
    (16, 7, 57, 32, 29),
]
# A transpose builder listed one shape but ran another, e.g. (20, 320, 15) as
# (320, 20, 15). The column-major layout below keeps the requested shape, so
# the listed shape and stride are exactly the ones executed.
RAVEL_LAYOUTS = ["contiguous", "strided"]

# Comprehensive level only: the degenerate geometries a caller may put in a
# shape file (scalar, zero extent, singleton dims) plus a 1-D scale. They are
# valid ravel inputs, and scalar / empty / singleton tensors stay contiguous,
# so those rows cover the view path only.
RAVEL_EXTRA_SHAPES = [(), (0,), (1,), (1, 1), (1, 9)]

# Static capability flags, never a collection-time probe.
_DTYPE_CAPABILITIES = {
    torch.bfloat16: flag_gems.runtime.device.support_bf16,
    torch.int64: flag_gems.runtime.device.support_int64,
}


def _gated(dtypes):
    """Deduplicate while keeping order, then drop unsupported capabilities."""
    return [
        dtype for dtype in dict.fromkeys(dtypes) if _DTYPE_CAPABILITIES.get(dtype, True)
    ]


_BENCH_DTYPES = _gated(consts.FLOAT_DTYPES + consts.INT_DTYPES + [torch.int64])


def _contiguous_strides(shape):
    """Row-major strides of ``shape`` without allocating anything.

    A zero extent leaves the running product unchanged in PyTorch's layout
    arithmetic -- torch.empty((2, 0, 3)).stride() is (3, 3, 1), not (0, 3, 1) --
    so the listed metadata matches the tensor the builder materializes.
    """
    strides = []
    stride = 1
    for extent in reversed(list(shape)):
        strides.append(stride)
        stride *= max(extent, 1)
    return tuple(reversed(strides))


def _column_major_strides(shape):
    """Column-major strides: the same shape over exactly numel elements."""
    strides = []
    stride = 1
    for extent in shape:
        strides.append(stride)
        stride *= max(extent, 1)
    return tuple(strides)


def _flatten_flavor(shape, strides):
    """Allocation-free prediction of ravel's view-vs-copy decision."""
    if math.prod(shape) <= 1:
        return "view"
    expected = 1
    for extent, stride in zip(reversed(list(shape)), reversed(strides)):
        if extent == 1:
            continue
        if stride != expected:
            return "copy"
        expected *= extent
    return "view"


def _validate_shape(shape):
    """Require true integer, non-negative extents before anything is planned.

    ``operator.index`` performs no coercion, so a float size such as 2.0 is
    rejected here instead of being silently accepted by torch.Size / the
    builder, which raise TypeError for it.
    """
    for extent in shape:
        if isinstance(extent, bool):
            raise ValueError(f"invalid ravel benchmark extent: {extent!r}")
        try:
            normalized = operator.index(extent)
        except TypeError:
            raise ValueError(f"invalid ravel benchmark extent: {extent!r}") from None
        if normalized < 0:
            raise ValueError(f"negative ravel benchmark extent: {extent!r}")


def _case_fn(shape, dtype):
    del dtype
    _validate_shape(shape)
    for layout in RAVEL_LAYOUTS:
        strides = (
            _contiguous_strides(shape)
            if layout == "contiguous"
            else _column_major_strides(shape)
        )
        flavor = _flatten_flavor(shape, strides)
        if layout == "strided" and flavor == "view":
            # Rank < 2, a singleton-only shape such as (1, 9), or a 0/1-element
            # shape is contiguous in both layouts: the row would duplicate the
            # contiguous one instead of covering the copy path.
            continue
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"layout": layout, "stride": str(strides), "flatten": flavor},
            builder_args=(shape, layout),
        )


def _generate_input(shape, dtype, device):
    """int64 is outside the shared generator's dtype branches, which would
    return None and put a non-tensor on the candidate path."""
    if dtype is torch.int64:
        return torch.randint(
            0, 1024, (math.prod(shape),), dtype=torch.int64, device=device
        ).reshape(shape)
    return utils.generate_tensor_input(shape, dtype, device)


def _build_inputs_fn(plan, dtype, device):
    shape, layout = plan.builder_args
    if layout == "contiguous":
        return _generate_input(shape, dtype, device), {}
    # Column-major view over a numel-element buffer: the requested shape is
    # preserved and flattening has to materialize, at no extra allocation.
    backing = _generate_input((math.prod(shape),), dtype, device)
    return torch.as_strided(backing, shape, _column_major_strides(shape)), {}


class RavelBenchmark(OperatorBenchmark):
    """Two-phase benchmark over ravel's view and materializing layouts."""

    DEFAULT_SHAPES = RAVEL_SHAPES

    def set_shapes(self, shape_file_path=None):
        # Shared resolution: a caller's shape file wins for this operator or for
        # this benchmark class, so no valid requested workload is filtered out.
        super().set_shapes(shape_file_path)

    def set_more_shapes(self):
        return RAVEL_EXTRA_SHAPES


@pytest.mark.ravel
def test_ravel():
    bench = RavelBenchmark(
        op_name="ravel",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.ravel,
        gems_op=getattr(flag_gems, "ravel", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
