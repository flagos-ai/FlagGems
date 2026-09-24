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

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# ``_sparse_mm`` starts with an underscore, and ``pytest.mark`` refuses to
# generate a marker via attribute access for such names. Register it directly on
# the MarkGenerator so ``@pytest.mark._sparse_mm`` and ``-m _sparse_mm`` work.
setattr(
    pytest.mark,
    "_sparse_mm",
    MarkDecorator(Mark("_sparse_mm", (), {}, _ispytest=True), _ispytest=True),
)

# (rows, inner, cols, nnz_per_row) workload descriptors: sparse (rows, inner)
# COO @ dense (inner, cols), so the stored nnz is rows * nnz_per_row. The
# nnz_per_row=256 entry stores more entries per row than it has columns, which
# makes the operand uncoalesced.
SPARSE_MM_SHAPES = [
    (1024, 1024, 1024, 32),
    (4096, 4096, 4096, 32),
    (512, 8192, 512, 128),
    (8192, 2048, 1024, 16),
    (512, 128, 512, 256),
]

# COO @ dense is measured with the wide dtypes: the reduced-precision types have
# no native COO spmm kernel on this backend, and the static capability flag
# decides whether float64 is available.
BENCH_DTYPES = [torch.float32]
if flag_gems.runtime.device.support_fp64:
    BENCH_DTYPES.append(torch.float64)


def _resolve_descriptor(rows, inner, cols, nnz_per_row):
    """Return the ``(stored_nnz, coalesced)`` pair the builder will create.

    Planning and construction share this helper, so the metadata reported by
    ``--list-cases`` cannot disagree with the tensors built for that case. Every
    field must be a plain non-negative integer: a fractional, boolean or string
    field is rejected while the case metadata is planned instead of being
    silently truncated when the case runs.
    """
    descriptor = (rows, inner, cols, nnz_per_row)
    for name, value in zip(("rows", "inner", "cols", "nnz_per_row"), descriptor):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be a plain integer: {descriptor}")
        if value < 0:
            raise ValueError(f"{name} must be non-negative: {descriptor}")
    if inner == 0 and nnz_per_row > 0:
        raise ValueError(
            f"inner==0 leaves no column to store into; use nnz_per_row=0: {descriptor}"
        )
    if rows == 0 or inner == 0 or nnz_per_row == 0:
        return 0, True
    # Every row keeps its first ``inner`` entries distinct and repeats the rest,
    # so the support is coalesced only while nnz_per_row <= inner.
    return rows * nnz_per_row, nnz_per_row <= inner


def _case_fn(shape, dtype):
    # Every shape entry is a (rows, inner, cols, nnz_per_row) descriptor; one
    # BenchmarkCasePlan per descriptor keeps the sparse metadata in the case
    # listing and defers tensor construction to the build phase.
    del dtype
    rows, inner, cols, nnz_per_row = shape
    stored_nnz, coalesced = _resolve_descriptor(rows, inner, cols, nnz_per_row)
    yield base.BenchmarkCasePlan(
        shape={
            "rows": rows,
            "inner": inner,
            "cols": cols,
            "nnz_per_row": nnz_per_row,
        },
        params={"stored_nnz": stored_nnz, "coalesced": coalesced},
        builder_args=(rows, inner, cols, nnz_per_row),
    )


def _random_values(shape, dtype, device):
    # benchmark.utils.generate_tensor_input only covers float32/16/bfloat16, so
    # float64 values come from torch directly.
    if dtype in consts.FLOAT_DTYPES:
        return utils.generate_tensor_input(shape, dtype, device)
    return torch.randn(shape, dtype=dtype, device=device)


def _build_inputs_fn(plan, dtype, device):
    rows, inner, cols, nnz_per_row = plan.builder_args
    stored_nnz, coalesced = _resolve_descriptor(rows, inner, cols, nnz_per_row)

    if stored_nnz == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=dtype, device=device)
    else:
        row_idx = torch.arange(rows, device=device).repeat_interleave(nnz_per_row)
        col_idx = (torch.arange(nnz_per_row, device=device) % inner).repeat(rows)
        indices = torch.stack([row_idx, col_idx])
        values = _random_values((stored_nnz,), dtype, device)

    sparse = torch.sparse_coo_tensor(
        indices, values, (rows, inner), device=device, is_coalesced=coalesced
    )
    dense = _random_values((inner, cols), dtype, device)
    return sparse, dense


class SparseMMBenchmark(OperatorBenchmark):
    # core_shapes.yaml has no entry for this operator, so the dedicated
    # (rows, inner, cols, nnz_per_row) descriptors above are the default and a
    # caller-supplied shape file still wins when it names the operator. The
    # generic extra shapes are not applicable and are not merged.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=SPARSE_MM_SHAPES)

    def set_more_shapes(self):
        return []


@pytest.mark._sparse_mm
def test__sparse_mm():
    bench = SparseMMBenchmark(
        op_name="_sparse_mm",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_mm,
        gems_op=getattr(flag_gems, "_sparse_mm", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
