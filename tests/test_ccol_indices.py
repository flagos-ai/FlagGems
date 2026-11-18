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

"""Correctness tests for ``aten::ccol_indices``.

``ccol_indices(self)`` returns the compressed column pointer array of a sparse
CSC tensor as a zero-copy alias of the internal ccol storage, shaped
``batch_dims + (ncols + 1)``. Only column-compressed layouts (CSC and BSC) are
accepted: CSR, BSR, COO and strided inputs are rejected with the native
``RuntimeError`` message, byte-identical.

The stored buffer's dtype is preserved (int32 construction stays int32), and
batched layouts keep their full batch shape. The returned values are layout
metadata, not computed data, so the suite is equally valid under
``utils.TO_CPU`` (it passes with ``--ref=cpu``).
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _make_csc(ccol, row, values, nrows, ncols, device, index_dtype=torch.int64):
    return torch.sparse_csc_tensor(
        torch.tensor(ccol, dtype=index_dtype, device=device),
        torch.tensor(row, dtype=index_dtype, device=device),
        torch.tensor(values, dtype=torch.float32, device=device),
        size=(nrows, ncols),
    )


# CSC fixtures as (ccol pointers, row indices, values, nrows, ncols). The mix
# covers empty, single-entry, fully dense and irregular column occupancy.
CSC_CASES = [
    ([0, 0, 0], [], [], 4, 2),
    ([0, 1], [0], [3.0], 3, 1),
    ([0, 2, 4], [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], 4, 2),
    ([0, 2, 4, 6], [2, 4, 1, 3, 0, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 5, 3),
]


@pytest.mark.ccol_indices
@pytest.mark.parametrize("ccol,row,values,nrows,ncols", CSC_CASES)
def test_ccol_indices(ccol, row, values, nrows, ncols):
    res_inp = _make_csc(ccol, row, values, nrows, ncols, flag_gems.device)
    ref_inp = utils.to_reference(res_inp)

    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(res_inp)

    assert res_out.dtype == torch.int64
    assert res_out.shape == (ncols + 1,)
    utils.gems_assert_equal(res_out, ref_inp.ccol_indices())
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ccol_indices
def test_ccol_indices_bsc():
    # BSC tensors carry their compressed column pointers in the same buffer.
    res_inp = torch.sparse_bsc_tensor(
        torch.tensor([0, 1, 2], device=flag_gems.device),
        torch.tensor([0, 1], device=flag_gems.device),
        torch.randn(1, 2, 2, device=flag_gems.device),
        size=(4, 2),
    )
    ref_inp = utils.to_reference(res_inp)
    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(res_inp)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ccol_indices
def test_ccol_indices_batched():
    ccol = torch.tensor([[0, 2, 3], [0, 1, 2]], device=flag_gems.device)
    row = torch.tensor([[0, 1, 0], [1, 0, 1]], device=flag_gems.device)
    res_inp = torch.sparse_csc_tensor(
        ccol, row, torch.randn(2, 3, device=flag_gems.device), (2, 2, 2)
    )
    ref_inp = utils.to_reference(res_inp)
    ref_out = torch.ops.aten.ccol_indices(ref_inp)
    res_out = flag_gems.ccol_indices(res_inp)
    assert res_out.shape == ref_out.shape == (2, 3)
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ccol_indices
def test_ccol_indices_int32_buffer():
    # The stored buffer's dtype is preserved on the accessor path. The
    # comparison goes through the device-reference helper, which moves both
    # sides consistently under --ref=cpu.
    res_inp = _make_csc(
        [0, 2, 3],
        [0, 1, 0],
        [1.0, 2.0, 3.0],
        2,
        2,
        flag_gems.device,
        index_dtype=torch.int32,
    )
    ref_inp = utils.to_reference(res_inp)
    res_out = flag_gems.ccol_indices(res_inp)
    ref_out = utils.to_reference(ref_inp.ccol_indices())
    assert res_out.dtype == torch.int32
    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.ccol_indices
def test_ccol_indices_alias_and_mutation():
    # The schema ``Tensor(a) self -> Tensor(a)`` makes the result an aliasing
    # view: two reads share storage, and mutating the result mutates the
    # sparse tensor's stored ccol values. A copying implementation breaks both.
    res_inp = _make_csc(
        [0, 2, 4], [0, 1, 2, 3], [1.0, 2.0, 3.0, 4.0], 4, 2, flag_gems.device
    )

    first = flag_gems.ccol_indices(res_inp)
    second = flag_gems.ccol_indices(res_inp)
    assert (
        first.data_ptr() == second.data_ptr()
    ), "ccol_indices must return an aliasing view, not a fresh copy"
    assert first.data_ptr() == res_inp.ccol_indices().data_ptr()

    first[0] = 77
    assert (
        res_inp.ccol_indices()[0].item() == 77
    ), "mutating the returned view must be visible in the sparse tensor"


@pytest.mark.ccol_indices
@pytest.mark.parametrize("layout", ["csr", "bsr"], ids=["csr", "bsr"])
def test_ccol_indices_unsupported_layout_raises(layout):
    # ATen accepts only column-compressed layouts (CSC and BSC) and rejects
    # the row-compressed ones with the native message, byte-identical.
    indptr = torch.tensor([0, 2, 4], dtype=torch.int64, device=flag_gems.device)
    indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=flag_gems.device)
    if layout == "csr":
        inp = torch.sparse_csr_tensor(indptr, indices, values, size=(2, 4))
    else:
        inp = torch.sparse_bsr_tensor(
            indptr,
            indices,
            torch.randn(2, 2, 2, device=flag_gems.device),
            size=(4, 4),
        )

    with pytest.raises(RuntimeError) as ref_err:
        torch.ops.aten.ccol_indices(utils.to_reference(inp))
    with pytest.raises(RuntimeError) as res_err:
        flag_gems.ccol_indices(inp)
    assert str(ref_err.value) == str(
        res_err.value
    ), f"{layout}: message differs: {res_err.value!r} vs {ref_err.value!r}"


@pytest.mark.ccol_indices
@pytest.mark.parametrize("layout", ["coo", "strided"], ids=["coo", "strided"])
def test_ccol_indices_dense_and_coo_rejected(layout):
    # COO and strided inputs are also rejected with the native message.
    if layout == "coo":
        inp = torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [0, 1]], device=flag_gems.device),
            torch.randn(2, device=flag_gems.device),
            (2, 2),
        )
    else:
        inp = torch.randn(2, 3, device=flag_gems.device)
    with pytest.raises(RuntimeError) as ref_err:
        torch.ops.aten.ccol_indices(utils.to_reference(inp))
    with pytest.raises(RuntimeError) as res_err:
        flag_gems.ccol_indices(inp)
    assert str(ref_err.value) == str(res_err.value)
