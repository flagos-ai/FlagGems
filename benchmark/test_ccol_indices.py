# Copyright 2026, The FlagOS Contributors.
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

from . import base, consts

# ``ccol_indices`` reads the compressed column pointer buffer of a sparse CSC
# tensor, so the workload is a CSC matrix with a fixed nnz; the descriptor is
# the matrix shape (nrows, ncols) and the nnz target scales the pointer count.
CCOL_INDICES_SHAPES = [
    (64, 64),  # tiny matrix
    (1024, 1024),  # small
    (4096, 4096),  # medium
    (65536, 1024),  # tall, more columns to address
    (1048576, 1024),  # large
]
# Fixed nnz per shape so the pointer count scales predictably with columns.
_NNZ_TARGETS = {
    (64, 64): 64,
    (1024, 1024): 4096,
    (4096, 4096): 65536,
    (65536, 1024): 65536,
    (1048576, 1024): 524288,
}


def _make_csc_with_nnz(nrows, ncols, nnz, dtype, device):
    """Build a valid CSC tensor with exactly ``nnz`` non-zero entries."""
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


class CcolIndicesBenchmark(base.Benchmark):
    """Benchmark ``ccol_indices`` on CSC tensors of varying size."""

    def set_shapes(self, shape_file_path=None):
        # CSC requires 2-D matrices, so the operator-specific shape list
        # replaces the stock DEFAULT_SHAPES (which contain 1-D/3-D tuples).
        self.shapes = list(CCOL_INDICES_SHAPES)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            nnz = _NNZ_TARGETS[shape]
            inp = _make_csc_with_nnz(shape[0], shape[1], nnz, cur_dtype, self.device)
            yield (inp,)


@pytest.mark.ccol_indices
def test_ccol_indices():
    bench = CcolIndicesBenchmark(
        op_name="ccol_indices",
        torch_op=torch.ops.aten.ccol_indices,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
