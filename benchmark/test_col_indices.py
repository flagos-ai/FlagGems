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

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts

# aten::col_indices(Tensor(a) self) -> Tensor(a) returns the batch_dims +
# (nnz,) int64 column index tensor of a sparse row-compressed tensor (CSR or
# BSR) -- a metadata accessor whose result is a view of the input's internal
# col_indices storage. Its cost is proportional to the returned index array
# (nnz per stored matrix) and independent of the stored values, so benchmark a
# spread of logical shapes (2-D, batched 3-D/4-D) and nnz values, plus BSR
# cases with larger blocks. The device-side allocation stays small relative to
# the logical size because only nnz entries are stored.
_COL_CASES = [
    ("csr", (1024, 1024), 65536, None),
    ("csr", (4096, 4096), 1048576, None),
    ("csr", (4096, 65536), 1048576, None),
    ("csr", (16, 1024, 1024), 262144, None),
    ("csr", (8, 4096, 4096), 524288, None),
    ("bsr", (4096, 4096), 65536, (8, 8)),
    ("bsr", (8192, 8192), 16384, (16, 16)),
    ("bsr_batch", (4, 4096, 4096), 16384, (8, 8)),
]


def _make_compressed_indices(batch, n_rows, n_cols, nnz, device):
    """Valid compressed-row structure generated directly on the benchmark device.

    ``crow`` has shape ``batch + (n_rows + 1,)`` (non-decreasing, starting at 0
    and ending at ``nnz``); ``cols`` has shape ``batch + (nnz,)``. Entries are
    sorted by (row, col) so the structure is a well-formed CSR/BSR index set.
    """
    entries = tuple(batch) + (nnz,)
    assert 0 <= nnz <= n_rows * n_cols
    positions = (
        torch.arange(nnz, dtype=torch.long, device=device)
        * (n_rows * n_cols)
        // max(nnz, 1)
    )
    rows = (positions // n_cols).expand(entries).contiguous()
    cols = (positions % n_cols).expand(entries).contiguous()
    counts = torch.zeros(tuple(batch) + (n_rows,), dtype=torch.long, device=device)
    if nnz > 0:
        counts.scatter_add_(
            -1, rows, torch.ones(entries, dtype=torch.long, device=device)
        )
    crow = torch.zeros(tuple(batch) + (n_rows + 1,), dtype=torch.long, device=device)
    crow[..., 1:] = counts.cumsum(-1)
    return crow, cols


def _random_values(shape, dtype, device):
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype, device=device)
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    return torch.randint(-5, 6, shape, dtype=dtype, device=device)


def _build_input(layout, size, nnz, blocks, dtype, device):
    batch, n_rows, n_cols = size[:-2], size[-2], size[-1]
    entries = tuple(batch) + (nnz,)
    if layout == "csr":
        crow, cols = _make_compressed_indices(batch, n_rows, n_cols, nnz, device)
        values = _random_values(entries, dtype, device)
        return torch.sparse_csr_tensor(crow, cols, values, size)
    if layout in ("bsr", "bsr_batch"):
        block_rows, block_cols = blocks
        n_row_blocks = n_rows // block_rows
        n_col_blocks = n_cols // block_cols
        crow, cols = _make_compressed_indices(
            batch, n_row_blocks, n_col_blocks, nnz, device
        )
        values = _random_values(entries + (block_rows, block_cols), dtype, device)
        return torch.sparse_bsr_tensor(crow, cols, values, size)
    raise ValueError(f"unknown layout {layout}")


def _case_fn(shape, dtype):
    del dtype
    layout, size, nnz, blocks = shape
    yield base.BenchmarkCasePlan(
        shape={"input": size},
        params={"nnz": nnz, "layout": layout},
        builder_args=(layout, size, nnz, blocks),
    )


def _build_inputs_fn(plan, dtype, device):
    layout, size, nnz, blocks = plan.builder_args
    inp = _build_input(layout, size, nnz, blocks, dtype, device)
    return inp, {}


class ColIndicesBenchmark(OperatorBenchmark):
    # col_indices is a sparse metadata accessor; there are no meaningful dense
    # shapes in core_shapes.yaml, so benchmark dedicated (layout, size, nnz,
    # blocks) cases instead.
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_COL_CASES)


@pytest.mark.col_indices
def test_col_indices():
    bench = ColIndicesBenchmark(
        op_name="col_indices",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.col_indices,
        gems_op=getattr(flag_gems, "col_indices", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
