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

"""Benchmark for ``aten::_sparse_sparse_matmul`` (2-D COO @ 2-D COO).

Both operands are sparse COO matrices, so cases are planned from (m, k, n)
descriptors instead of dense shape lists.  ``nnz_requested_a``/``_b`` record how
many coordinates are drawn per operand; the stored entries are those draws
deduplicated plus the anchor positions, so the stored nnz is not exactly the
requested count.
"""

import pytest
import torch

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

_NNZ_DIVISOR = 16
# Number of small inner indices the coordinate anchor always stores.
_ANCHOR_INNER = 3

# bf16 operands are benchmarked only where the device advertises bf16 support.
# The same static selection drives listing and execution, so the case list seen
# by --list-cases is exactly the one that runs.
_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

# (m, k, n) descriptors: square and rectangular products, both with a large
# shared dimension, which is where sparse x sparse matmul work concentrates.
# The dense core_shapes.yaml has no sparse operand entries, so these descriptors
# are the default unless a shape file configures this operator.
_DEFAULT_SHAPES = [
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 1024, 1024),
    (1024, 2048, 4096),
]


def _requested_nnz(size):
    """Coordinates drawn for an operand with ``size`` positions.  The stored
    entries are the distinct drawn coordinates plus the anchor positions, so the
    stored nnz is not exactly this value."""
    if size <= 0:
        return 0
    return max(1, size // _NNZ_DIVISOR)


def _validated_triple(shape):
    """Strict metadata check for one (m, k, n) descriptor, run in ``set_shapes``
    before any operand is allocated; bools are rejected even though they are
    ints."""
    if not isinstance(shape, (tuple, list)) or len(shape) != 3:
        raise ValueError(f"expected an (m, k, n) descriptor, got {shape!r}")
    extents = []
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(f"invalid shape extent {extent!r} in {shape!r}")
        extents.append(extent)
    return tuple(extents)


def _case_fn(shape, dtype):
    del dtype
    m, k, n = shape
    yield base.BenchmarkCasePlan(
        shape={"mat1": [m, k], "mat2": [k, n]},
        params={
            "nnz_requested_a": _requested_nnz(m * k),
            "nnz_requested_b": _requested_nnz(k * n),
            "stored_nnz": "distinct drawn coordinates plus anchor positions",
        },
        builder_args=(m, k, n),
    )


def _stored_codes(shape, count, device, seed):
    """Distinct linearised positions for one operand: ``count`` seeded random
    draws plus the anchor grid ``(0, 0..K-1)`` and ``(0..K-1, 0)``, all
    deduplicated.  With a nonzero shared dimension K the anchor keeps the small
    inner indices stored in both operands, so those cases do real work instead
    of an empty product; a descriptor with a zero extent is a legitimate custom
    shape whose product is genuinely empty.  Stored entries are the
    deduplicated draws plus the anchor positions, not exactly ``count``."""
    rows, cols = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    drawn = torch.randint(
        0,
        rows * cols,
        (max(count, 1),),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )
    keep = min(_ANCHOR_INNER, max(rows, cols))
    anchor = torch.cat(
        [
            torch.arange(min(cols, keep), dtype=torch.int64, device=device),
            torch.arange(min(rows, keep), dtype=torch.int64, device=device) * cols,
        ]
    )
    return torch.unique(torch.cat([drawn, anchor]))


def _sparse_operand(shape, count, dtype, device, seed):
    rows, cols = shape
    generator = torch.Generator(device=device).manual_seed(seed)
    if count <= 0 or rows == 0 or cols == 0:
        # Without stored positions the index tensor stays empty; randint with a
        # zero high bound is never used.
        index = torch.empty(2, 0, dtype=torch.int64, device=device)
    else:
        codes = _stored_codes(shape, count, device, seed)
        index = torch.stack([codes // cols, codes % cols])
    values = torch.randn(
        index.shape[1], generator=generator, device=device, dtype=torch.float32
    ).to(dtype)
    return torch.sparse_coo_tensor(index, values, shape, device=device)


def _build_inputs_fn(plan, dtype, device):
    m, k, n = plan.builder_args
    mat1 = _sparse_operand((m, k), _requested_nnz(m * k), dtype, device, seed=0)
    mat2 = _sparse_operand((k, n), _requested_nnz(k * n), dtype, device, seed=1)
    # GenericBenchmark unpacks the triple into positional and keyword arguments.
    return mat1, mat2, {}


class SparseSparseMatmulBenchmark(OperatorBenchmark):
    # Dense shape lists cannot describe COO operands, so the operator's own
    # (m, k, n) descriptors are used unless a shape file configures it.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)
        self.shapes = [_validated_triple(shape) for shape in self.shapes]


@pytest.mark.sparse_sparse_matmul
def test__sparse_sparse_matmul():
    bench = SparseSparseMatmulBenchmark(
        op_name="_sparse_sparse_matmul",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_sparse_matmul,
        gems_op=getattr(flag_gems, "_sparse_sparse_matmul", None),
        dtypes=_DTYPES,
    )
    bench.run()
