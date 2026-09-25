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

"""KernelGen benchmark for ``torch.ops.aten._sparse_semi_structured_apply``.

The operator takes a rank-2 ``(M, K)`` activation tile plus a ``uint8``
``(M / 8, K / 8, 8)`` thread mask, so the mask is derived from the tile and is
built in the same phase as the input.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# ``pytest.mark`` refuses attribute access for underscore-prefixed names, so the
# marker is registered on the MarkGenerator directly.
setattr(
    pytest.mark,
    "_sparse_semi_structured_apply",
    MarkDecorator(
        Mark("_sparse_semi_structured_apply", (), {}, _ispytest=True), _ispytest=True
    ),
)

# The native kernel accepts float16 and bfloat16 only, so the dtype set is listed
# explicitly instead of using a convenience list; bfloat16 is gated on the static
# device capability flag.
APPLY_DTYPES = [torch.float16] + (
    [torch.bfloat16] if flag_gems.runtime.device.support_bf16 else []
)

# Performance-relevant legal tiles, from 1.6e4 to 2.1e6 elements.
_APPLY_SHAPES = [
    (128, 128),
    (256, 256),
    (320, 640),
    (512, 1024),
    (1024, 1024),
    (2048, 1024),
]


def _tile_problem(shape):
    """Describe why ``shape`` is not a legal 2:4 activation tile, else ``None``.

    The native kernel takes rank-2 tiles with positive extents, ``M % 32 == 0``
    and ``K % 64 == 0``, and raises for any other shape. A caller-supplied shape
    file can also carry entries written for other operators, so an illegal tile is
    reported while the cases are listed rather than failing only inside the timed
    call. The descriptor is validated before any arithmetic, so a malformed entry
    (a float or bool extent, a string, or something that is not a sequence of two
    extents) yields this diagnostic instead of a TypeError from the modulo below.
    """
    if not isinstance(shape, (list, tuple)):
        return (
            f"the tile must be a list or tuple of extents, got {type(shape).__name__}"
        )
    if len(shape) != 2:
        return f"the native kernel takes a rank-2 (M, K) tile, got rank {len(shape)}"
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int):
            return f"tile extents must be integers, got {shape!r}"
    rows, cols = shape
    if rows <= 0 or cols <= 0:
        return f"the native kernel rejects zero-extent tiles, got {shape!r}"
    if rows % 32:
        return f"M must be a positive multiple of 32, got {rows}"
    if cols % 64:
        return f"K must be a positive multiple of 64, got {cols}"
    return None


def _case_fn(shape, dtype):
    problem = _tile_problem(shape)
    if problem is not None:
        raise ValueError(
            f"illegal _sparse_semi_structured_apply workload {shape!r}: {problem}"
        )
    if dtype not in APPLY_DTYPES:
        raise ValueError(
            f"_sparse_semi_structured_apply takes {APPLY_DTYPES}, got {dtype}"
        )
    rows, cols = shape
    yield base.BenchmarkCasePlan(
        shape={"input": shape},
        params={"mask_rows": rows // 8, "mask_cols": cols // 8},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    masks = torch.randint(
        0, 256, (shape[0] // 8, shape[1] // 8, 8), dtype=torch.uint8, device=device
    )
    return inp, masks


class SparseSemiStructuredApplyBenchmark(OperatorBenchmark):
    """Two-phase benchmark limited to the tiles the native kernel accepts."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_APPLY_SHAPES)


@pytest.mark._sparse_semi_structured_apply
def test__sparse_semi_structured_apply():
    bench = SparseSemiStructuredApplyBenchmark(
        op_name="_sparse_semi_structured_apply",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_semi_structured_apply,
        # flag_gems._sparse_semi_structured_apply is installed by the KernelGen
        # override; Benchmark._candidate_call picks it up at run time.
        gems_op=getattr(flag_gems, "_sparse_semi_structured_apply", None),
        dtypes=APPLY_DTYPES,
    )
    bench.run()
