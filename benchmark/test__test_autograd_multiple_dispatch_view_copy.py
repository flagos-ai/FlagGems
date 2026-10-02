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

"""Benchmark for ``aten::_test_autograd_multiple_dispatch_view_copy``.

The operator flattens its input into a fresh rank-1 contiguous buffer, so the
curated rows pair copy scales with flattenable source layouts: a strided last
dimension (``t[..., ::2]``) and a narrow offset slice (``t[1:3]``) both keep one
contiguous subspace. The curated rows are merged into the shared
core/comprehensive grid -- and into a caller's ``--shape-file`` shapes -- instead
of replacing them, and a shape that is not curated derives its layouts from
static shape facts only, so no impossible combination is generated. Case
listing and execution share these plans.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# (base allocation shape, source layout).
_BENCH_CASES = [
    ((1 << 20,), "contiguous"),
    ((1 << 22,), "contiguous"),
    ((1 << 24,), "contiguous"),
    ((1024, 1024), "contiguous"),
    ((1024, 1024), "last_dim_stride"),
    ((20, 320, 15), "contiguous"),
    ((20, 320, 15), "narrow_offset"),
    ((16, 128, 64, 60), "contiguous"),
    ((16, 128, 64, 60), "last_dim_stride"),
]

_CURATED_SHAPES = [shape for shape, _layout in _BENCH_CASES]

# Static capability flags; the same list drives listing and execution.
_DTYPE_CAPABILITY = {
    torch.bfloat16: "support_bf16",
    torch.float64: "support_fp64",
    torch.complex128: "support_fp64",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}

_BENCH_DTYPES = [
    dtype
    for dtype in dict.fromkeys(
        consts.FLOAT_DTYPES
        + consts.INT_DTYPES
        + consts.EXTRA_INT_DTYPES
        + consts.BOOL_DTYPES
        + consts.COMPLEX_DTYPES
        + [torch.float64, torch.complex128, torch.float8_e4m3fn, torch.float8_e5m2]
    )
    if dtype not in _DTYPE_CAPABILITY
    or getattr(flag_gems.runtime.device, _DTYPE_CAPABILITY[dtype])
]


def _normalize_extents(shape):
    """Tuple extents, so a bare int (1-D) and a list both normalize here."""
    if isinstance(shape, int) and not isinstance(shape, bool):
        return (shape,)
    return tuple(shape)


def _logical_shape(base_shape, layout):
    """Shape of the tensor the operator receives for one benchmark row."""
    if layout == "contiguous":
        return base_shape
    if layout == "last_dim_stride":
        return base_shape[:-1] + ((base_shape[-1] + 1) // 2,)
    if layout == "narrow_offset":
        return (min(2, max(0, base_shape[0] - 1)),) + base_shape[1:]
    raise ValueError(f"Unknown benchmark input layout {layout!r}")


def _plans_for(shape):
    """Validated base shape and the layouts to benchmark for it.

    Curated shapes keep their rows. Any other shape is decided from static shape
    facts alone: a contiguous base only flattens when one contiguous subspace
    remains, which for ``t[..., ::2]`` means an even last extent and for the
    whole-row ``t[1:3]`` slice needs rank >= 2 with a non-singleton first extent.
    Zero-extent and scalar shapes therefore get the contiguous row only.
    """
    base_shape = _normalize_extents(shape)
    curated = [layout for row_shape, layout in _BENCH_CASES if row_shape == base_shape]
    if curated:
        return base_shape, curated
    plans = ["contiguous"]
    if base_shape and base_shape[-1] > 0 and base_shape[-1] % 2 == 0:
        plans.append("last_dim_stride")
    if len(base_shape) >= 2 and base_shape[0] >= 2:
        plans.append("narrow_offset")
    return base_shape, plans


def _case_fn(shape, dtype):
    del dtype
    base_shape, plans = _plans_for(shape)
    for layout in plans:
        yield base.BenchmarkCasePlan(
            shape={
                "input": list(_logical_shape(base_shape, layout)),
                "input_layout": layout,
            },
            params={},
            builder_args=(base_shape, layout),
        )


def _build_inputs_fn(plan, dtype, device):
    base_shape, layout = plan.builder_args
    inp = torch.empty(base_shape, dtype=dtype, device=device)
    if layout == "last_dim_stride":
        inp = inp[..., ::2]
    elif layout == "narrow_offset":
        inp = inp[1:3]
    return (inp,)


class AutogradViewCopyBenchmark(OperatorBenchmark):
    """Shared spec shapes plus the curated flattenable copy rows."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Union the curated rows into the shared core/comprehensive grid (and a
        # caller's --shape-file shapes) instead of replacing it.
        self.shapes = list(
            dict.fromkeys(tuple(shape) for shape in list(self.shapes) + _CURATED_SHAPES)
        )


@pytest.mark.test_autograd_multiple_dispatch_view_copy
def test__test_autograd_multiple_dispatch_view_copy():
    bench = AutogradViewCopyBenchmark(
        op_name="_test_autograd_multiple_dispatch_view_copy",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._test_autograd_multiple_dispatch_view_copy,
        gems_op=getattr(flag_gems, "_test_autograd_multiple_dispatch_view_copy", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
