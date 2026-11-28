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

COMPRESSED_UNSAFE_DTYPES = utils.FLOAT_DTYPES

LAYOUTS = [
    (torch.sparse_csr, "crow", "col", False),
    (torch.sparse_csc, "ccol", "row", False),
    (torch.sparse_bsr, "crow", "col", True),
    (torch.sparse_bsc, "ccol", "row", True),
]
CANVAS = (4, 4)
BLOCK = (2, 2)


def _make_components(nnz, dtype, layout, blocked, seed=0):
    """Valid compressed components for the requested layout on the (4, 4) canvas."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    compressed = torch.tensor([0, 1, 2, 3, 4][: nnz + 1], dtype=torch.int64)
    if nnz == 4:
        plain = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    else:
        plain = torch.randint(0, 2, (nnz,), generator=g).to(torch.int64)
    shape = (nnz,) + BLOCK if blocked else (nnz,)
    values = torch.randn(shape, dtype=dtype, generator=g)
    size = CANVAS + ((2,) if blocked else ())
    return (
        compressed.to(flag_gems.device),
        plain.to(flag_gems.device),
        values.to(flag_gems.device),
        size,
    )


def _getters(layout, blocked):
    if layout == torch.sparse_csr:
        return (lambda t: t.crow_indices(), lambda t: t.col_indices())
    if layout == torch.sparse_csc:
        return (lambda t: t.ccol_indices(), lambda t: t.row_indices())
    if layout == torch.sparse_bsr:
        return (lambda t: t.crow_indices(), lambda t: t.col_indices())
    return (lambda t: t.ccol_indices(), lambda t: t.row_indices())


def _assert_sparse_equal(res, ref, layout, blocked):
    assert res.layout == layout
    assert ref.layout == layout
    assert res.shape == ref.shape
    get_compressed, get_plain = _getters(layout, blocked)
    utils.gems_assert_equal(get_compressed(res), get_compressed(ref))
    utils.gems_assert_equal(get_plain(res), get_plain(ref))
    utils.gems_assert_equal(res.values(), ref.values())


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("nnz", [0, 1, 4])
@pytest.mark.parametrize("layout", [lo for lo, _, _, _ in LAYOUTS])
@pytest.mark.parametrize("dtype", COMPRESSED_UNSAFE_DTYPES)
def test_sparse_compressed_tensor_unsafe(nnz, layout, dtype):
    spec = next(s for s in LAYOUTS if s[0] == layout)
    blocked = spec[3]
    compressed, plain, values, size = _make_components(nnz, dtype, layout, blocked)

    ref_values = utils.to_reference(values)
    ref = torch.ops.aten._sparse_compressed_tensor_unsafe(
        utils.to_reference(compressed),
        utils.to_reference(plain),
        ref_values,
        size,
        dtype=ref_values.dtype,
        layout=layout,
        device=ref_values.device,
    )
    res = flag_gems.sparse_compressed_tensor_unsafe(
        compressed, plain, values, size, layout=layout
    )

    _assert_sparse_equal(res, ref, layout, blocked)
    # without a dtype request the payload aliases the input values storage
    if nnz > 0:
        assert res.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_compressed_tensor_unsafe
@pytest.mark.parametrize("nnz", [1, 4])
@pytest.mark.parametrize("layout", [lo for lo, _, _, _ in LAYOUTS])
@pytest.mark.parametrize("dtype", COMPRESSED_UNSAFE_DTYPES)
def test_sparse_compressed_tensor_unsafe_dtype(nnz, layout, dtype):
    """The dtype argument converts the payload through the Triton kernel."""
    spec = next(s for s in LAYOUTS if s[0] == layout)
    blocked = spec[3]
    compressed, plain, values, size = _make_components(nnz, dtype, layout, blocked)
    target = torch.float64

    ref_values = utils.to_reference(values).to(target)
    ref = torch.ops.aten._sparse_compressed_tensor_unsafe(
        utils.to_reference(compressed),
        utils.to_reference(plain),
        ref_values,
        size,
        dtype=target,
        layout=layout,
        device=ref_values.device,
    )
    res = flag_gems.sparse_compressed_tensor_unsafe(
        compressed, plain, values, size, dtype=target, layout=layout
    )

    _assert_sparse_equal(res, ref, layout, blocked)
    assert res.values().dtype == target


@pytest.mark.sparse_compressed_tensor_unsafe
def test_sparse_compressed_tensor_unsafe_requires_layout():
    """A missing layout is rejected exactly like the native constructor."""
    compressed, plain, values, size = _make_components(
        1, torch.float32, torch.sparse_csr, False
    )
    with pytest.raises(RuntimeError):
        flag_gems.sparse_compressed_tensor_unsafe(compressed, plain, values, size)


@pytest.mark.sparse_compressed_tensor_unsafe
def test_sparse_compressed_tensor_unsafe_skips_index_validation():
    """The unsafe constructor accepts indices the validated one rejects."""
    # compressed axis not non-decreasing, a plain index outside its range
    compressed = torch.tensor([0, 3, 1], dtype=torch.int64, device=flag_gems.device)
    plain = torch.tensor([7, 0], dtype=torch.int64, device=flag_gems.device)
    values = torch.randn(2, device=flag_gems.device)
    size = (4, 4)

    ref_values = utils.to_reference(values)
    ref = torch.ops.aten._sparse_compressed_tensor_unsafe(
        utils.to_reference(compressed),
        utils.to_reference(plain),
        ref_values,
        size,
        dtype=ref_values.dtype,
        layout=torch.sparse_csr,
        device=ref_values.device,
    )
    res = flag_gems.sparse_compressed_tensor_unsafe(
        compressed, plain, values, size, layout=torch.sparse_csr
    )

    _assert_sparse_equal(res, ref, torch.sparse_csr, False)
