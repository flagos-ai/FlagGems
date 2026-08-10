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

# 1D / 2D / 3D sparse COO cases as (shape, indices, values). Indices are given in
# ascending order so that the uncoalesced and coalesced forms of each case hold
# identical index data and differ only in the coalesced flag.
SPARSE_CASES = [
    ((4,), [[0, 2]], [1.0, 2.0]),
    ((8,), [[1, 3, 5]], [1.0, 2.0, 3.0]),
    ((2, 2), [[0, 1], [0, 1]], [1.0, 2.0]),
    ((4, 4), [[0, 1, 3], [2, 0, 3]], [1.0, 2.0, 3.0]),
    ((2, 3, 4), [[0, 1], [1, 2], [0, 3]], [1.0, 2.0]),
]


def _make_sparse(shape, indices, values, dtype):
    i = torch.tensor(indices, dtype=torch.int64, device=flag_gems.device)
    v = torch.tensor(values, dtype=dtype, device=flag_gems.device)
    return torch.sparse_coo_tensor(i, v, shape)


@pytest.mark.is_coalesced
@pytest.mark.parametrize("shape,indices,values", SPARSE_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_is_coalesced(shape, indices, values, dtype, caplog):
    # The freshly constructed tensor is not marked coalesced; calling coalesce()
    # sets the flag. Both branches of the boolean result are covered here.
    inp = _make_sparse(shape, indices, values, dtype)
    inp_coalesced = inp.coalesce()

    ref_out = torch.ops.aten.is_coalesced(inp)
    ref_out_coalesced = torch.ops.aten.is_coalesced(inp_coalesced)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_coalesced"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_coalesced(inp)
            res_out_coalesced = torch.ops.aten.is_coalesced(inp_coalesced)

    assert "GEMS IS_COALESCED" in caplog.text
    assert res_out == ref_out, f"Expected {ref_out}, got {res_out}"
    assert res_out_coalesced == ref_out_coalesced
    assert res_out is False
    assert res_out_coalesced is True


@pytest.mark.is_coalesced
def test_is_coalesced_identical_indices(caplog):
    # The flag is a storage state bit rather than a property of the data: these
    # two tensors carry bit-identical indices but opposite flags. This is the
    # case a kernel deriving the answer from the indices would get wrong.
    i = torch.tensor([[0, 1], [0, 1]], dtype=torch.int64, device=flag_gems.device)
    v = torch.tensor([1.0, 2.0], device=flag_gems.device)
    uncoalesced = torch.sparse_coo_tensor(i, v, (2, 2))
    coalesced = torch.sparse_coo_tensor(i, v, (2, 2)).coalesce()

    assert torch.equal(uncoalesced._indices(), coalesced._indices())

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_coalesced"):
        with flag_gems.use_gems():
            res_uncoalesced = torch.ops.aten.is_coalesced(uncoalesced)
            res_coalesced = torch.ops.aten.is_coalesced(coalesced)

    assert "GEMS IS_COALESCED" in caplog.text
    assert res_uncoalesced is False
    assert res_coalesced is True


@pytest.mark.is_coalesced
def test_is_coalesced_duplicate_indices(caplog):
    # Duplicate coordinates are summed by coalesce(), so the coalesced form has
    # fewer stored elements while still reporting the flag as set.
    i = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.int64, device=flag_gems.device)
    v = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
    inp = torch.sparse_coo_tensor(i, v, (2, 2))
    inp_coalesced = inp.coalesce()

    ref_out = torch.ops.aten.is_coalesced(inp)
    ref_out_coalesced = torch.ops.aten.is_coalesced(inp_coalesced)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_coalesced"):
        with flag_gems.use_gems():
            res_out = torch.ops.aten.is_coalesced(inp)
            res_out_coalesced = torch.ops.aten.is_coalesced(inp_coalesced)

    assert "GEMS IS_COALESCED" in caplog.text
    assert res_out == ref_out
    assert res_out_coalesced == ref_out_coalesced
    assert inp_coalesced._nnz() == 2


@pytest.mark.is_coalesced
def test_is_coalesced_empty(caplog):
    # A sparse tensor with no stored elements is trivially coalesced once
    # coalesce() has been called.
    i = torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device)
    v = torch.zeros((0,), device=flag_gems.device)
    inp = torch.sparse_coo_tensor(i, v, (2, 2))
    inp_coalesced = inp.coalesce()

    ref_out_coalesced = torch.ops.aten.is_coalesced(inp_coalesced)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_coalesced"):
        with flag_gems.use_gems():
            res_out_coalesced = torch.ops.aten.is_coalesced(inp_coalesced)

    assert "GEMS IS_COALESCED" in caplog.text
    assert res_out_coalesced == ref_out_coalesced
    assert res_out_coalesced is True


@pytest.mark.is_coalesced
def test_is_coalesced_strided_input(caplog):
    # Strided tensors carry the dense backend key, which this operator is also
    # registered on, so they do reach the implementation and must be rejected
    # there with the same error eager mode raises.
    inp = torch.randn(4, 4, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="sparse coordinate tensor layout"):
        torch.ops.aten.is_coalesced(inp)

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_coalesced"):
        with flag_gems.use_gems():
            with pytest.raises(RuntimeError, match="sparse coordinate tensor layout"):
                torch.ops.aten.is_coalesced(inp)

    assert "GEMS IS_COALESCED" in caplog.text


@pytest.mark.is_coalesced
@pytest.mark.parametrize("to_layout", ["to_sparse_csr", "to_sparse_csc"])
def test_is_coalesced_compressed_layout(to_layout, caplog):
    # Compressed sparse layouts carry their own dispatch keys, which this
    # operator is not registered on, so they keep reaching the native default
    # branch. Asserted here to pin down that the extra registration does not
    # change the error these inputs raise.
    inp = getattr(torch.randn(4, 4, device=flag_gems.device), to_layout)()

    with caplog.at_level("DEBUG", logger="flag_gems.ops.is_coalesced"):
        with flag_gems.use_gems():
            with pytest.raises(RuntimeError, match="sparse coordinate tensor layout"):
                torch.ops.aten.is_coalesced(inp)

    assert "GEMS IS_COALESCED" not in caplog.text
