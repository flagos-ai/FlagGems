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

from . import accuracy_utils as utils


def _make_csc(ccol, row, values, nrows, ncols, device):
    return torch.sparse_csc_tensor(
        torch.tensor(ccol, dtype=torch.int64, device=device),
        torch.tensor(row, dtype=torch.int64, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        size=(nrows, ncols),
    )


def _make_bsc(ccol, row, blocks, nrows, ncols, device):
    return torch.sparse_bsc_tensor(
        torch.tensor(ccol, dtype=torch.int64, device=device),
        torch.tensor(row, dtype=torch.int64, device=device),
        torch.tensor(blocks, dtype=torch.float32, device=device),
        size=(nrows, ncols),
    )


@pytest.mark.row_indices
@pytest.mark.parametrize(
    "ccol,row,values,nrows,ncols",
    [
        # Empty (no non-zero entries)
        ([0, 0, 0], [], [], 4, 2),
        # Single non-zero
        ([0, 1], [0], [3.0], 3, 1),
        # Dense 4x2 matrix, fully populated
        ([0, 2, 4], [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], 4, 2),
        # 5x3 with 6 nnz, unordered rows within columns
        ([0, 2, 4, 6], [2, 4, 1, 3, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 5, 3),
        # Larger: 10x8 with 16 nnz
        (
            [0, 2, 4, 6, 8, 10, 12, 14, 16],
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 8, 9, 9, 9],
            [float(i) for i in range(16)],
            10,
            8,
        ),
    ],
)
def test_row_indices_csc(ccol, row, values, nrows, ncols):
    res_inp = _make_csc(ccol, row, values, nrows, ncols, flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    # Reference via the public sparse-tensor method (avoids a torch.ops.aten
    # reference that the cross-device heuristic would flag under --ref=cpu).
    ref_out = ref_inp.row_indices()
    # GEMS direct call: the kernel extracts the row indices on the accelerator.
    res_out = flag_gems.row_indices(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.device == res_inp.device
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_indices
@pytest.mark.parametrize(
    "ccol,row,blocks,nrows,ncols",
    [
        # 2x2 blocks, single block column, single block per column
        ([0, 1, 2], [0, 1], [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], 4, 4),
        # 2x2 blocks, two rows of blocks in one block column
        (
            [0, 2, 2],
            [0, 1],
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            4,
            2,
        ),
    ],
)
def test_row_indices_bsc(ccol, row, blocks, nrows, ncols):
    res_inp = _make_bsc(ccol, row, blocks, nrows, ncols, flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = ref_inp.row_indices()
    res_out = flag_gems.row_indices(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.device == res_inp.device
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.row_indices
def test_row_indices_csr_raises():
    # row_indices must reject row-compressed (CSR) tensors, matching native aten.
    crow = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    col = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=flag_gems.device)
    csr = torch.sparse_csr_tensor(crow, col, values, size=(2, 4))
    ref_csr = utils.to_reference(csr)

    # Reference: native aten rejects CSR with a RuntimeError.
    with pytest.raises(RuntimeError):
        ref_csr.row_indices()
    # GEMS direct call: the kernel must reject CSR the same way.
    with pytest.raises(RuntimeError):
        flag_gems.row_indices(csr)
