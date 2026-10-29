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


def _make_csr(shape, dtype, density=0.5, seed=0, device=None):
    """Build a compressed sparse row tensor on the target device."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, dtype=dtype, generator=g)
    mask = torch.rand(shape, generator=g) < density
    sp = (x * mask).to_sparse_csr()
    if device is not None:
        sp = sp.to(device)
    return sp


def _assert_csr_close(res_out, ref_out, dtype):
    assert res_out.layout == torch.sparse_csr
    assert ref_out.layout == torch.sparse_csr
    assert res_out.shape == ref_out.shape
    assert res_out._nnz() == ref_out._nnz()

    # A fully reduced empty descriptor (crow = [0, 0]) carries a 0-d value
    # buffer that ATen itself cannot convert with to_dense(); compare the
    # components directly there.
    if ref_out.values().dim() == 0:
        res_v = res_out.values()
        ref_v = ref_out.values()
        if not utils.TO_CPU:
            ref_v = ref_v.to(flag_gems.device)
        utils.gems_assert_close(res_v, ref_v, dtype)
        return

    res_dense = res_out.to_dense()
    ref_dense = ref_out.to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, dtype)


DIMS = [[0], [1], [0, 1], []]


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("shape", [(64, 48), (128, 1), (1, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_csr_sum(dim, shape, dtype):
    """Row, column and full reductions match ATen on random CSR inputs."""
    ref_sp = _make_csr(shape, dtype, device="cpu")
    res_sp = ref_sp.to(flag_gems.device)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_sp, dim, True)
    res_out = flag_gems.sparse_csr_sum(res_sp, dim, True)

    _assert_csr_close(res_out, ref_out, ref_out.dtype)


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("dim", DIMS)
def test_sparse_csr_sum_keeps_layout_not_values(dim):
    """Unstored zeros do not participate: structure follows the layout."""
    crow = torch.tensor([0, 2, 3], device=flag_gems.device)
    col = torch.tensor([2, 0, 1], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0], device=flag_gems.device)
    sp = torch.sparse_csr_tensor(crow, col, values, (2, 3))

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(sp.cpu(), dim, True)
    res_out = flag_gems.sparse_csr_sum(sp, dim, True)

    assert res_out._nnz() == ref_out._nnz()
    _assert_csr_close(res_out, ref_out, torch.float32)


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("dim", DIMS)
def test_sparse_csr_sum_explicit_zeros(dim):
    """A stored zero does reach the reduction, exactly like dense."""
    crow = torch.tensor([0, 2, 3], device=flag_gems.device)
    col = torch.tensor([0, 1, 0], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([0.0, 5.0, 7.0], device=flag_gems.device)
    sp = torch.sparse_csr_tensor(crow, col, values, (2, 3))

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(sp.cpu(), dim, True)
    res_out = flag_gems.sparse_csr_sum(sp, dim, True)

    _assert_csr_close(res_out, ref_out, torch.float32)


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("dim", DIMS)
def test_sparse_csr_sum_empty_input(dim):
    """A CSR tensor with no stored entries reduces to an empty descriptor."""
    crow = torch.tensor([0, 0, 0], device=flag_gems.device)
    col = torch.tensor([], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([], device=flag_gems.device)
    sp = torch.sparse_csr_tensor(crow, col, values, (2, 3))

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(sp.cpu(), dim, True)
    res_out = flag_gems.sparse_csr_sum(sp, dim, True)

    assert res_out._nnz() == ref_out._nnz() == 0
    _assert_csr_close(res_out, ref_out, torch.float32)


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("dim", [[0], [1], [0, 1]])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.float16])
def test_sparse_csr_sum_dtype(dim, dtype):
    """An explicit dtype casts both the payload and the accumulator."""
    sp = _make_csr((32, 32), torch.float32, device=flag_gems.device)

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(sp.cpu(), dim, True, dtype=dtype)
    res_out = flag_gems.sparse_csr_sum(sp, dim, True, dtype=dtype)

    assert res_out.dtype == ref_out.dtype == dtype
    _assert_csr_close(res_out, ref_out, dtype)


@pytest.mark.sparse_csr_sum
def test_sparse_csr_sum_unsorted_columns():
    """Column reduction emits columns in ascending order regardless of input."""
    crow = torch.tensor([0, 3, 5], device=flag_gems.device)
    col = torch.tensor([2, 0, 1, 2, 1], dtype=torch.int64, device=flag_gems.device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], device=flag_gems.device)
    sp = torch.sparse_csr_tensor(crow, col, values, (2, 3))

    ref_out = torch.ops.aten._sparse_csr_sum.dim_dtype(sp.cpu(), [0], True)
    res_out = flag_gems.sparse_csr_sum(sp, [0], True)

    _assert_csr_close(res_out, ref_out, torch.float32)


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("dim", [[0], [1]])
def test_sparse_csr_sum_out(dim):
    """The out form rewrites the caller's tensor and returns that same object."""
    ref_sp = _make_csr((16, 16), torch.float32, device="cpu")
    sp = ref_sp.to(flag_gems.device)

    ref_result = torch.ops.aten._sparse_csr_sum.dim_dtype(ref_sp, dim, True)
    out = torch.ops.aten._sparse_csr_sum.dim_dtype(
        sp, dim, True
    )  # descriptor with the right stored-element count

    res_out = flag_gems.sparse_csr_sum_out(sp, dim, True, out=out)

    assert res_out is out
    _assert_csr_close(res_out, ref_result, torch.float32)


@pytest.mark.sparse_csr_sum
@pytest.mark.parametrize("kwargs", [dict(dim=[1], keepdim=False), dict(dim=[2])])
def test_sparse_csr_sum_rejects_bad_request(kwargs):
    """keepdim=False and out-of-range dims raise, like ATen."""
    sp = _make_csr((8, 8), torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.sparse_csr_sum(sp, **kwargs)
