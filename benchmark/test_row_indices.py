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
# See the License for the specific language governing permissions or
# limitations under the License.

import pytest
import torch

from . import base, consts

# ``row_indices`` is a sparse *column-compressed* (CSC/BSR) tensor accessor. The
# amount of work is proportional to the number of non-zero entries (``nnz``),
# so the benchmark builds CSC matrices of fixed nnz and measures the cost of
# reading the stored COO row indices. Each entry is reported as the matrix
# shape (nrows, ncols); the underlying nnz is kept modest so the benchmark
# reflects realistic sparse workloads, not a dense copy.
_ROW_INDICES_SHAPES = [
    (64, 64),  # tiny matrix
    (1024, 1024),  # small
    (4096, 4096),  # medium
    (65536, 1024),  # tall, more columns to read
    (1048576, 1024),  # large
    (1048576, 4096),  # largest
]
# Fixed nnz per shape so the copy work scales predictably.
_NNZ_TARGETS = {
    (64, 64): 64,
    (1024, 1024): 4096,
    (4096, 4096): 65536,
    (65536, 1024): 65536,
    (1048576, 1024): 524288,
    (1048576, 4096): 1048576,
}


def _make_csc_with_nnz(nrows, ncols, nnz, dtype, device):
    """Build a valid CSC tensor with exactly ``nnz`` non-zero entries."""
    # Distribute nnz across columns, with sorted row indices within each column
    # so the CSC invariants hold.
    rng = torch.Generator(device="cpu").manual_seed(0)
    per_col = max(1, nnz // ncols)
    ccol = [0]
    row = []
    for _ in range(ncols):
        idx = torch.randint(0, nrows, (per_col,), generator=rng)
        idx = torch.unique(idx)
        if len(idx) == 0:
            idx = torch.tensor([0])
        row.extend(idx.tolist())
        ccol.append(len(row))
    ccol_t = torch.tensor(ccol, dtype=torch.int64, device=device)
    row_t = torch.tensor(row, dtype=torch.int64, device=device)
    values_t = torch.randn(len(row), dtype=dtype, device=device)
    return torch.sparse_csc_tensor(ccol_t, row_t, values_t, size=(nrows, ncols))


class RowIndicesBenchmark(base.GenericBenchmark):
    """Benchmark ``row_indices`` on CSC tensors of varying size."""

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in _ROW_INDICES_SHAPES:
            nnz = _NNZ_TARGETS[shape]
            inp = _make_csc_with_nnz(shape[0], shape[1], nnz, dtype, self.device)
            yield (inp,)


@pytest.mark.row_indices
def test_row_indices():
    bench = RowIndicesBenchmark(
        op_name="row_indices",
        input_fn=None,
        torch_op=torch.ops.aten.row_indices,
        dtypes=consts.FLOAT_DTYPES,
    )
    # CSC requires 2D matrices; the stock DEFAULT_SHAPES contain 1D/3D tuples
    # that are meaningless here, so use the operator-specific shape list.
    bench.shapes = [tuple(s) for s in _ROW_INDICES_SHAPES]
    bench.run()
