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

"""Correctness tests for ``aten::_convert_indices_from_csr_to_coo``.

Both operands are integral index tensors, so every value range is a stored
index value; there is no floating-point matrix and no backward (integer
tensors cannot require grad). The decoded table has shape ``(nd + 2, N * nnz)``:
leading planes hold the batch coordinates, then X and Y, with
``(X, Y) = (row, col)`` or ``(col, row)`` when ``transpose=True``.

Every positive case keeps a valid CSR structure: the row pointers are
monotonic and the block's terminal pointer equals its stored entry count.
The native operator does not validate the structure, so malformed layouts are
deliberately not asserted: its out-of-bounds reads make the result
unspecified, and this implementation is not required to reproduce undefined
behaviour. Mismatched batch extents, rank mismatches and unsupported dtypes
are rejected before any read, so those are covered as negative cases.

The decoded table is integral and device-independent: under ``utils.TO_CPU``
the reference operands move to CPU and the suite still compares equal (the
suite passes with ``--ref=cpu``), so no device skip is needed.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# CI derives the pytest marker from the implementation file name via the
# #6359 convention: the stripped marker collides with this operator's own
# registered id, so the derived marker takes the ``underscore_`` prefix.
# Declare both so ``pytest -m <marker>`` selects these tests from either
# entry point (same convention as the merged ``_indices_copy`` suite).


# Index dtypes the operator accepts. The 8-bit forms are added explicitly
# because the shared pool starts at int16; int64 is gated by the device
# capability the runtime publishes.
INDEX_DTYPES = [torch.int8, torch.uint8] + list(utils.ALL_INT_DTYPES)

# (leading batch shape, row count, stored entries per block). The batch shapes
# cover the rank-0 (single block), rank-1/2 static paths and the generic
# higher-rank decomposition; the last entry keeps nnz above the block size.
BLOCK_CASES = [
    ((), 0, 0),
    ((), 1, 1),
    ((), 4, 3),
    ((2,), 4, 3),
    ((2, 3), 8, 5),
    ((2, 2, 2), 8, 5),
    ((2, 2, 2, 2), 16, 32),
    ((2, 2, 2, 2, 2), 4, 3),
]


def _make_csr(lead, rows, nnz, dtype, seed=0):
    """Valid CSR operands: monotonic pointers ending at ``nnz``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    cuts = torch.sort(
        torch.randint(0, nnz + 1, lead + (rows,), generator=gen, dtype=torch.int64),
        dim=-1,
    ).values
    crow = torch.cat([torch.zeros(lead + (1,), dtype=torch.int64), cuts], dim=-1)
    crow[..., -1] = nnz
    col = torch.randint(0, 64, lead + (nnz,), generator=gen, dtype=torch.int64)
    return (
        crow.to(dtype).to(flag_gems.device),
        col.to(dtype).to(flag_gems.device),
    )


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("lead,rows,nnz", BLOCK_CASES)
@pytest.mark.parametrize("dtype", INDEX_DTYPES)
def test_convert_indices_from_csr_to_coo(lead, rows, nnz, dtype):
    crowd, cold = _make_csr(lead, rows, nnz, dtype)
    ref_crowd = utils.to_reference(crowd)
    ref_cold = utils.to_reference(cold)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(ref_crowd, ref_cold)
    res_out = flag_gems._convert_indices_from_csr_to_coo(crowd, cold)

    assert res_out.dtype == torch.int64
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("out_int32", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("lead,rows,nnz", [((), 4, 3), ((2, 3), 8, 5)])
def test_convert_indices_from_csr_to_coo_flags(out_int32, transpose, lead, rows, nnz):
    crowd, cold = _make_csr(lead, rows, nnz, torch.int32, seed=1)
    ref_crowd = utils.to_reference(crowd)
    ref_cold = utils.to_reference(cold)

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crowd, ref_cold, out_int32=out_int32, transpose=transpose
    )
    res_out = flag_gems._convert_indices_from_csr_to_coo(
        crowd, cold, out_int32=out_int32, transpose=transpose
    )

    assert res_out.dtype == (torch.int32 if out_int32 else torch.int64)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo_empty():
    # No stored entries: the table still has one plane per coordinate.
    crowd = torch.tensor([0, 0, 0], dtype=torch.int64, device=flag_gems.device)
    cold = torch.zeros(0, dtype=torch.int64, device=flag_gems.device)
    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        utils.to_reference(crowd), utils.to_reference(cold)
    )
    res_out = flag_gems._convert_indices_from_csr_to_coo(crowd, cold)
    assert res_out.shape == ref_out.shape
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("layout", ["offset_view", "batch_strided"])
def test_convert_indices_from_csr_to_coo_strided_inputs(layout):
    # Both operands may be non-contiguous views; the decoder reads through the
    # declared strides, including the leading batch strides.
    if layout == "offset_view":
        # Row pointers with a non-zero storage offset and unit stride.
        crow = torch.tensor(
            [0, 0, 0, 1, 2, 2, 9, 9],
            dtype=torch.int64,
            device=flag_gems.device,
        ).narrow(0, 0, 5)
        col = torch.tensor([7, 8], dtype=torch.int64, device=flag_gems.device)
    else:
        # A strided leading batch dimension: the view is transposed, so its
        # blocks alternate with a gap. Every block ends at its entry count.
        blocks = [[0, 2, 2], [0, 1, 2], [0, 2, 2], [0, 2, 2]]
        crow = torch.tensor(blocks, dtype=torch.int64).reshape(2, 2, 3)
        crow = crow.transpose(0, 1).to(flag_gems.device)
        col = torch.randint(
            0,
            9,
            (2, 2, 2),
            generator=torch.Generator().manual_seed(3),
            dtype=torch.int64,
        )
        col = col.transpose(0, 1).to(flag_gems.device)
    ref_crow = utils.to_reference(crow)
    ref_col = utils.to_reference(col)
    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(ref_crow, ref_col)
    res_out = flag_gems._convert_indices_from_csr_to_coo(crow, col)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.convert_indices_from_csr_to_coo_out
@pytest.mark.parametrize("out_int32", [False, True])
def test_convert_indices_from_csr_to_coo_out(out_int32):
    crowd, cold = _make_csr((2,), 4, 3, torch.int32, seed=2)
    ref_crowd = utils.to_reference(crowd)
    ref_cold = utils.to_reference(cold)
    out_dtype = torch.int32 if out_int32 else torch.int64

    ref_out = torch.ops.aten._convert_indices_from_csr_to_coo(
        ref_crowd, ref_cold, out_int32=out_int32
    )
    res_out = torch.empty(ref_out.shape, dtype=out_dtype, device=flag_gems.device)
    # The reference buffer follows the reference operands: under --ref=cpu the
    # native .out call needs all of its tensors on the same device.
    ref_buf = torch.empty(ref_out.shape, dtype=out_dtype, device=ref_out.device)

    res_ret = flag_gems._convert_indices_from_csr_to_coo_out(
        crowd, cold, out_int32=out_int32, out=res_out
    )
    torch.ops.aten._convert_indices_from_csr_to_coo.out(
        ref_crowd, ref_cold, out_int32=out_int32, out=ref_buf
    )

    assert res_ret is res_out
    utils.gems_assert_equal(res_ret, ref_buf)


@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize("layout", [0, 1])
def test_convert_indices_from_csr_to_coo_rejects_zero_dim(layout):
    # A rank-0 operand is rejected natively.
    scalar = torch.tensor(0, dtype=torch.int32, device=flag_gems.device)
    vector = torch.tensor([0, 1], dtype=torch.int32, device=flag_gems.device)
    operands = (scalar, vector) if layout == 0 else (vector, scalar)
    with pytest.raises(IndexError):
        flag_gems._convert_indices_from_csr_to_coo(*operands)


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo_rejects_rank_mismatch():
    crow = torch.zeros(3, dtype=torch.int32, device=flag_gems.device)
    col = torch.zeros((2, 2), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_csr_to_coo(crow, col)


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo_rejects_batch_mismatch():
    crow = torch.zeros((2, 3), dtype=torch.int32, device=flag_gems.device)
    col = torch.zeros((3, 2), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_csr_to_coo(crow, col)


# No floating dtype has a kernel: the operator is integral-only.
@pytest.mark.convert_indices_from_csr_to_coo
@pytest.mark.parametrize(
    "dtype",
    # Rejected-dtype matrix: every entry here is a negative case, not a tested
    # precision.
    [
        torch.float16,
        torch.float32,
        torch.bool,
    ]
    # bfloat16 is a rejection case (no kernel), not a tested precision.
    + ([torch.bfloat16] if utils.bf16_is_supported else [])
    + ([torch.float64, torch.complex128] if utils.fp64_is_supported else []),
)
def test_convert_indices_from_csr_to_coo_rejects_unsupported_dtype(dtype):
    crow = torch.zeros(3, dtype=dtype, device=flag_gems.device)
    col = torch.zeros(2, dtype=dtype, device=flag_gems.device)
    with pytest.raises(NotImplementedError):
        flag_gems._convert_indices_from_csr_to_coo(crow, col)


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo_rejects_positional_flags():
    crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=flag_gems.device)
    col = torch.zeros(2, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems._convert_indices_from_csr_to_coo(crow, col, True)


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo_rejects_invalid_flag_type():
    crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=flag_gems.device)
    col = torch.zeros(2, dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(TypeError):
        flag_gems._convert_indices_from_csr_to_coo(crow, col, transpose="yes")


@pytest.mark.convert_indices_from_csr_to_coo
def test_convert_indices_from_csr_to_coo_rejects_out_dtype_mismatch():
    crow = torch.tensor([0, 1, 2], dtype=torch.int32, device=flag_gems.device)
    col = torch.zeros(2, dtype=torch.int32, device=flag_gems.device)
    out = torch.zeros((2, 2), dtype=torch.int32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._convert_indices_from_csr_to_coo_out(crow, col, out=out)
