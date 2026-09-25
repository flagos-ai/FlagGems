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

from . import base
from .generated_operator_utils import OperatorBenchmark

# ``_sparse_csr_sum`` starts with an underscore, so ``pytest.mark`` cannot be
# extended by attribute access; register the marker on the MarkGenerator.
setattr(
    pytest.mark,
    "_sparse_csr_sum",
    MarkDecorator(Mark("_sparse_csr_sum", (), {}, _ispytest=True), _ispytest=True),
)

_DEVICE_FLAGS = flag_gems.runtime.device

# Native ``aten::_sparse_csr_sum`` reduces the axes named by ``dim``, so both
# axes are benchmarked: dim=0 keeps the columns (flat, wide output) and dim=1
# keeps the rows. Stored count per row is the performance-relevant knob, and
# the row/column extremes keep both reduction axes on their requested extents.
_CSR_BENCH_SHAPES = [
    ((1024, 1024), 1024),
    ((2048, 2048), 2048),
    ((4096, 4096), 4096),
    ((4096, 16384), 16384),
    ((16384, 4096), 4096),
    ((4096, 4096), 8),
    ((4096, 16384), 8),
    ((16384, 4096), 8),
]

_BENCH_DTYPES = [torch.float16, torch.float32]
if _DEVICE_FLAGS.support_bf16:
    _BENCH_DTYPES.append(torch.bfloat16)
if _DEVICE_FLAGS.support_fp64:
    _BENCH_DTYPES.append(torch.float64)


def _descriptor(entry):
    """Normalize one CSR descriptor to ``((rows, cols), nnz_per_row)``.

    Two forms are accepted: ``(rows, cols)`` builds a dense operand (as many
    stored entries per row as there are columns), and ``((rows, cols),
    nnz_per_row)`` requests an explicit hollow count. Validation happens while
    case metadata is planned, so an invalid request is rejected during listing,
    before any tensor is allocated. Extents and counts must already be
    non-negative integers: casting with ``int()`` would silently truncate a
    fractional request instead of rejecting it.
    """
    if not isinstance(entry, (tuple, list)) or len(entry) != 2:
        raise ValueError(
            f"CSR descriptor {entry!r} must be (rows, cols) or ((rows, cols), nnz_per_row)"
        )
    first, second = entry
    if isinstance(first, (tuple, list)):
        size, nnz_per_row = first, second
    else:
        size, nnz_per_row = entry, None
    if not isinstance(size, (tuple, list)) or len(size) != 2:
        raise ValueError(f"CSR descriptor {entry!r} must carry a (rows, cols) size")
    rows, cols = size
    if nnz_per_row is None:
        nnz_per_row = cols
    for name, value in (("rows", rows), ("cols", cols), ("nnz_per_row", nnz_per_row)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"CSR descriptor {entry!r}: {name}={value!r} must be an integer, "
                f"got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(
                f"CSR descriptor {entry!r}: {name}={value} must be non-negative"
            )
    if nnz_per_row > cols:
        raise ValueError(
            f"CSR descriptor {entry!r}: nnz_per_row={nnz_per_row} exceeds cols={cols}"
        )
    return (rows, cols), nnz_per_row


def _case_fn(shape, dtype):
    del dtype
    (rows, cols), nnz_per_row = _descriptor(shape)
    nnz = rows * nnz_per_row
    # Density is reported exactly as requested; rounding it would misreport the
    # stored count the plan actually builds.
    density = nnz / (rows * cols) if rows * cols else 0.0

    for dim in (0, 1):
        yield base.BenchmarkCasePlan(
            shape={"input": (rows, cols), "nnz_per_row": nnz_per_row},
            params={"dim": dim, "keepdim": True, "nnz": nnz, "density": density},
            builder_args=((rows, cols), nnz_per_row, dim),
        )


def _build_inputs_fn(plan, dtype, device):
    (rows, cols), nnz_per_row, dim = plan.builder_args
    if nnz_per_row == 0:
        crow = torch.zeros(rows + 1, dtype=torch.int64, device=device)
        col = torch.empty(0, dtype=torch.int64, device=device)
        values = torch.empty(0, dtype=dtype, device=device)
    else:
        # Spread columns over the whole column extent: one stride-``step`` grid
        # per row, shifted by a per-row amount smaller than ``step``, so each
        # row stores exactly ``nnz_per_row`` entries with columns strictly
        # increasing, unique and inside the extent. The stride keeps the
        # nonzeros spread across the operand rather than packed into one band,
        # while still exercising the row-major stored order a real operand has.
        step = cols // nnz_per_row
        grid = torch.arange(nnz_per_row, device=device) * step
        row_shift = torch.arange(rows, device=device) % step
        col = (grid[None, :] + row_shift[:, None]).reshape(-1)
        crow = torch.arange(rows + 1, device=device) * nnz_per_row
        values = torch.randn(rows * nnz_per_row, dtype=dtype, device=device)
    inp = torch.sparse_csr_tensor(crow, col, values, size=(rows, cols), device=device)
    return inp, {"dim": dim, "keepdim": True}


class SparseCsrSumBenchmark(OperatorBenchmark):
    """Two-phase benchmark over the (rows, cols, nnz_per_row, dim) grid."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CSR_BENCH_SHAPES)


@pytest.mark._sparse_csr_sum
def test__sparse_csr_sum():
    bench = SparseCsrSumBenchmark(
        op_name="_sparse_csr_sum",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_csr_sum.dim_dtype,
        gems_op=getattr(flag_gems, "_sparse_csr_sum", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
