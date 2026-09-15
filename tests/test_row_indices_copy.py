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

# Sparse CSC matrix configurations (rows, cols, nnz). row_indices_copy extracts
# the row_indices buffer from a column-compressed sparse tensor (sparse_csc /
# sparse_bsc), so each case exercises a different sparsity pattern / size.
CSC_SHAPES = [
    (4, 4, 6),  # small square
    (8, 16, 20),  # tall
    (16, 8, 20),  # wide
    (32, 32, 256),  # dense-ish
    (1024, 1024, 4096),  # large
    (1, 1, 1),  # minimal
    (3, 3, 0),  # fully empty
]

# Index dtypes supported for sparse buffers.
INDEX_DTYPES = [torch.int32, torch.int64]


def _make_csc(shape, index_dtype, device):
    rows, cols, nnz = shape
    if nnz == 0:
        ccol = torch.zeros(cols + 1, dtype=index_dtype, device=device)
        row = torch.empty(0, dtype=index_dtype, device=device)
        vals = torch.empty(0, dtype=torch.float32, device=device)
    else:
        # Build a valid compressed column structure: distribute nnz across the
        # columns, then choose sorted row indices per column.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1024)
        per_col = torch.randint(0, max(nnz, 1) * 2, (cols,), generator=gen).tolist()
        if sum(per_col) == 0:
            per_col = [1] * min(cols, nnz) if cols else []
        scale = nnz / max(sum(per_col), 1)
        per_col = [max(int(round(c * scale)), 0) for c in per_col]
        if per_col:
            per_col[-1] = max(per_col[-1] + (nnz - sum(per_col)), 0)
        else:
            per_col = [nnz]
        ccol = torch.zeros(cols + 1, dtype=index_dtype, device="cpu")
        acc = 0
        for i in range(cols):
            acc += per_col[i]
            ccol[i + 1] = acc
        ccol = ccol.to(device).to(index_dtype)
        row_list = []
        for i in range(cols):
            n = int(ccol[i + 1].item() - ccol[i].item())
            if n > 0:
                rs = torch.randperm(rows, generator=gen)[:n]
                rs, _ = torch.sort(rs)
                row_list.append(rs)
        if row_list:
            row = torch.cat(row_list).to(index_dtype).to(device)
        else:
            row = torch.empty(0, dtype=index_dtype, device=device)
        vals = torch.randn(row.numel(), dtype=torch.float32, device=device)
    return torch.sparse_csc_tensor(ccol, row, vals, size=(rows, cols))


def _make_bsc(shape, blocksize, index_dtype, device):
    rows, cols = shape
    nblock_rows = rows // blocksize
    nblock_cols = cols // blocksize
    # simple: one block per compressed column, dense
    nblocks = nblock_cols
    ccol = torch.empty(nblock_cols + 1, dtype=index_dtype, device=device)
    for i in range(nblock_cols + 1):
        ccol[i] = i
    row = torch.zeros(nblocks, dtype=index_dtype, device=device)
    # alternate row block indices to be non-trivial
    for i in range(nblocks):
        row[i] = i % nblock_rows
    # sort row indices per column (single block per column, already sorted)
    vals = torch.randn(
        nblocks, blocksize, blocksize, dtype=torch.float32, device=device
    )
    return torch.sparse_bsc_tensor(
        ccol, row, vals, size=(nblock_rows * blocksize, nblock_cols * blocksize)
    )


@pytest.mark.row_indices_copy
@pytest.mark.parametrize("shape", CSC_SHAPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_row_indices_copy_csc(shape, index_dtype):
    device = flag_gems.device
    csc = _make_csc(shape, index_dtype, device)
    ref_csc = csc.to("cpu")

    ref_out = torch.row_indices_copy(ref_csc)
    res_out = flag_gems.row_indices_copy(csc)

    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out.to("cpu"), ref_out)


@pytest.mark.row_indices_copy
def test_row_indices_copy_bsc():
    device = flag_gems.device
    bsc = _make_bsc((8, 8), 2, torch.int64, device)
    ref_bsc = bsc.to("cpu")

    ref_out = torch.row_indices_copy(ref_bsc)
    res_out = flag_gems.row_indices_copy(bsc)

    assert res_out.dtype == ref_out.dtype
    assert res_out.is_contiguous()
    utils.gems_assert_equal(res_out.to("cpu"), ref_out)


@pytest.mark.row_indices_copy
def test_row_indices_copy_wrong_layout():
    device = flag_gems.device
    # CSR is row-compressed, not column-compressed -> should raise.
    crow = torch.tensor([0, 2, 3, 5], device=device)
    col = torch.tensor([0, 1, 0, 1, 2], device=device)
    vals = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=device)
    csr = torch.sparse_csr_tensor(crow, col, vals, size=(3, 3))

    ref_err = False
    try:
        torch.row_indices_copy(csr.to("cpu"))
    except RuntimeError:
        ref_err = True

    gems_err = False
    try:
        flag_gems.row_indices_copy(csr)
    except RuntimeError:
        gems_err = True

    assert ref_err and gems_err


@pytest.mark.row_indices_copy_out
@pytest.mark.parametrize("shape", CSC_SHAPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_row_indices_copy_out(shape, index_dtype):
    # Exercises the .out ATen overload through the registered FlagGems kernel.
    device = flag_gems.device
    csc = _make_csc(shape, index_dtype, device)
    ref_csc = csc.to("cpu")

    ref_out = torch.empty(csc.row_indices().numel(), dtype=index_dtype, device="cpu")
    torch.row_indices_copy(ref_csc, out=ref_out)

    res_out = torch.empty(csc.row_indices().numel(), dtype=index_dtype, device=device)
    res_result = flag_gems.row_indices_copy_out(csc, out=res_out)

    assert res_result.data_ptr() == res_out.data_ptr()
    utils.gems_assert_equal(res_out.to("cpu"), ref_out)


@pytest.mark.row_indices_copy
def test_row_indices_copy_out_resize():
    # PyTorch resizes a mismatched-shape out tensor to the required shape.
    device = flag_gems.device
    csc = _make_csc((8, 8, 10), torch.int64, device)
    ref_csc = csc.to("cpu")

    ref_out = torch.empty(0, dtype=torch.int64, device="cpu")
    torch.row_indices_copy(ref_csc, out=ref_out)

    res_out = torch.empty(0, dtype=torch.int64, device=device)
    res_result = flag_gems.row_indices_copy_out(csc, out=res_out)

    assert res_result.data_ptr() == res_out.data_ptr()
    assert res_out.numel() == csc.row_indices().numel()
    utils.gems_assert_equal(res_out.to("cpu"), ref_out)


@pytest.mark.row_indices_copy
def test_row_indices_copy_out_dtype_mismatch_raises():
    device = flag_gems.device
    csc = _make_csc((4, 4, 6), torch.int32, device)

    # PyTorch refuses to copy int32 indices into an int64 out tensor.
    ref_err = False
    try:
        torch.row_indices_copy(
            csc.to("cpu"), out=torch.empty(6, dtype=torch.int64, device="cpu")
        )
    except RuntimeError:
        ref_err = True

    gems_err = False
    try:
        flag_gems.row_indices_copy_out(
            csc, out=torch.empty(6, dtype=torch.int64, device=device)
        )
    except RuntimeError:
        gems_err = True

    assert ref_err and gems_err
