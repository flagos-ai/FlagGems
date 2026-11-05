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

BSC_UNSAFE_DTYPES = utils.FLOAT_DTYPES

# 2x2 blocks on a (4, 4) canvas: 2 block rows x 2 block cols
BLOCK = (2, 2)
CANVAS = (4, 4)


def _make_components(nnz, dtype, seed=0):
    """Valid BSC components: nnz blocks of shape BLOCK on the (4, 4) canvas."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    ccol = torch.tensor([0, 1, 2, 3, 4][: nnz + 1], dtype=torch.int64)
    if nnz == 4:
        row = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    else:
        row = torch.randint(0, 2, (nnz,), generator=g).to(torch.int64)
    values = torch.randn((nnz,) + BLOCK, dtype=dtype, generator=g)
    return (
        ccol.to(flag_gems.device),
        row.to(flag_gems.device),
        values.to(flag_gems.device),
    )


def _assert_sparse_equal(res, ref):
    assert res.layout == torch.sparse_bsc
    assert ref.layout == torch.sparse_bsc
    assert res.shape == ref.shape
    utils.gems_assert_equal(res.ccol_indices(), ref.ccol_indices())
    utils.gems_assert_equal(res.row_indices(), ref.row_indices())
    utils.gems_assert_equal(res.values(), ref.values())


@pytest.mark.sparse_bsc_tensor_unsafe
@pytest.mark.parametrize("nnz", [0, 1, 4])
@pytest.mark.parametrize("dtype", BSC_UNSAFE_DTYPES)
def test_sparse_bsc_tensor_unsafe(nnz, dtype):
    ccol, row, values = _make_components(nnz, dtype)
    size = (4, 4, 2)  # one dense dim of size 2

    ref_values = utils.to_reference(values)
    ref = torch.ops.aten._sparse_bsc_tensor_unsafe(
        utils.to_reference(ccol),
        utils.to_reference(row),
        ref_values,
        size,
        dtype=ref_values.dtype,
        device=ref_values.device,
    )
    res = flag_gems.sparse_bsc_tensor_unsafe(ccol, row, values, size)

    _assert_sparse_equal(res, ref)
    # without a dtype request the payload aliases the input values storage
    if nnz > 0:
        assert res.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_bsc_tensor_unsafe
@pytest.mark.parametrize("nnz", [1, 4])
@pytest.mark.parametrize("dtype", BSC_UNSAFE_DTYPES)
def test_sparse_bsc_tensor_unsafe_dtype(nnz, dtype):
    """The dtype argument converts the payload through the Triton kernel."""
    ccol, row, values = _make_components(nnz, dtype)
    size = (4, 4, 2)
    target = torch.float64

    ref_values = utils.to_reference(values).to(target)
    ref = torch.ops.aten._sparse_bsc_tensor_unsafe(
        utils.to_reference(ccol),
        utils.to_reference(row),
        ref_values,
        size,
        dtype=target,
        device=ref_values.device,
    )
    res = flag_gems.sparse_bsc_tensor_unsafe(ccol, row, values, size, dtype=target)

    _assert_sparse_equal(res, ref)
    assert res.values().dtype == target


@pytest.mark.sparse_bsc_tensor_unsafe
def test_sparse_bsc_tensor_unsafe_skips_index_validation():
    """The unsafe constructor accepts indices the validated one rejects."""
    # ccol not non-decreasing, a row index outside the block-row range
    ccol = torch.tensor([0, 2, 1], dtype=torch.int64, device=flag_gems.device)
    row = torch.tensor([5, 0], dtype=torch.int64, device=flag_gems.device)
    values = torch.randn(2, *BLOCK, device=flag_gems.device)
    size = CANVAS + (2,)

    ref_values = utils.to_reference(values)
    ref = torch.ops.aten._sparse_bsc_tensor_unsafe(
        utils.to_reference(ccol),
        utils.to_reference(row),
        ref_values,
        size,
        dtype=ref_values.dtype,
        device=ref_values.device,
    )
    res = flag_gems.sparse_bsc_tensor_unsafe(ccol, row, values, size)

    _assert_sparse_equal(res, ref)
