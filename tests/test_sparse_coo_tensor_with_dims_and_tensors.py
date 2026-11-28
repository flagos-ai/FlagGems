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


def _payload(size, sparse_dim, dtype, device, nnz=3, seed=0):
    """Build a deterministic (indices, values) pair for a COO descriptor."""
    g = torch.Generator().manual_seed(seed)
    base = torch.arange(nnz, dtype=torch.int64)
    indices = torch.stack([base % max(int(size[d]), 1) for d in range(sparse_dim)])
    dense_shape = tuple(size[sparse_dim:])
    values = torch.randn((nnz, *dense_shape), dtype=dtype, generator=g).contiguous()
    return indices.to(device), values.to(device)


def _native(
    sparse_dim, dense_dim, size, indices, values, dtype=None, is_coalesced=None
):
    return torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        size,
        indices,
        values,
        dtype=dtype,
        layout=torch.sparse_coo,
        device=indices.device,
        is_coalesced=is_coalesced,
    )


def _assert_coo_close(res_out, ref_out):
    assert res_out.layout == torch.sparse_coo
    assert ref_out.layout == torch.sparse_coo
    assert res_out.shape == ref_out.shape
    assert res_out.sparse_dim() == ref_out.sparse_dim()
    assert res_out.dense_dim() == ref_out.dense_dim()
    assert res_out.dtype == ref_out.dtype
    assert res_out._nnz() == ref_out._nnz()

    res_dense = res_out.coalesce().to_dense()
    ref_dense = ref_out.coalesce().to_dense()
    if not utils.TO_CPU:
        ref_dense = ref_dense.to(flag_gems.device)
    utils.gems_assert_close(res_dense, ref_dense, ref_out.dtype)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize(
    "sparse_dim,dense_dim,size",
    [
        (2, 0, [4, 4]),
        (1, 0, [8]),
        (3, 0, [2, 3, 4]),
        (2, 1, [4, 4, 7]),
        (1, 2, [5, 2, 3]),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors(sparse_dim, dense_dim, size, dtype):
    """The assembled descriptor matches ATen for plain and hybrid shapes."""
    ref_indices, ref_values = _payload(size, sparse_dim, dtype, "cpu")
    indices, values = _payload(size, sparse_dim, dtype, flag_gems.device)
    ref_out = _native(sparse_dim, dense_dim, size, ref_indices, ref_values, dtype)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        size,
        indices,
        values,
        dtype=dtype,
        layout=torch.sparse_coo,
    )

    _assert_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("nnz", [0, 1, 5])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_nnz(nnz, dtype):
    """Empty, single-entry and multi-entry payloads are all forwarded verbatim."""
    ref_indices, ref_values = _payload([4, 4], 2, dtype, "cpu", nnz=nnz)
    indices, values = _payload([4, 4], 2, dtype, flag_gems.device, nnz=nnz)
    ref_out = _native(2, 0, [4, 4], ref_indices, ref_values, dtype)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, dtype=dtype
    )

    assert res_out._nnz() == ref_out._nnz() == nnz
    _assert_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_aliases_payload():
    """The caller's index and value buffers are used directly, with no copy."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, is_coalesced=True
    )

    assert res_out.is_coalesced()
    assert res_out.indices().data_ptr() == indices.data_ptr()
    assert res_out.values().data_ptr() == values.data_ptr()


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_default_uncoalesced():
    """A freshly assembled descriptor keeps the native uncoalesced default."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values
    )
    ref_out = _native(2, 0, [4, 4], indices.cpu(), values.cpu())

    assert res_out.is_coalesced() == ref_out.is_coalesced() is False


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("is_coalesced", [True, False])
def test_sparse_coo_tensor_with_dims_and_tensors_forwards_coalesced(is_coalesced):
    """is_coalesced is recorded verbatim without reordering the payload."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, is_coalesced=is_coalesced
    )

    assert res_out.is_coalesced() is is_coalesced


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_out(dtype):
    """The out form rewrites the caller's tensor and returns that same object."""
    ref_indices, ref_values = _payload([4, 4], 2, dtype, "cpu")
    indices, values = _payload([4, 4], 2, dtype, flag_gems.device)
    out = _native(2, 0, [4, 4], indices, values, dtype)
    ref_out = _native(2, 0, [4, 4], ref_indices, ref_values, dtype)

    res_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors_out(
        2, 0, [4, 4], indices, values, out=out
    )

    assert res_out is out
    _assert_coo_close(res_out, ref_out)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_registration_names():
    implementations = (
        flag_gems.sparse_coo_tensor_with_dims_and_tensors,
        flag_gems.sparse_coo_tensor_with_dims_and_tensors_out,
    )
    names = [
        entry[0] for entry in flag_gems._FULL_CONFIG if entry[1] in implementations
    ]
    assert names == [
        "_sparse_coo_tensor_with_dims_and_tensors",
        "_sparse_coo_tensor_with_dims_and_tensors.out",
    ]


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors
def test_sparse_coo_tensor_with_dims_and_tensors_rejects_bad_layout():
    """A non-COO layout request is rejected before assembling anything."""
    indices, values = _payload([4, 4], 2, torch.float32, flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.sparse_coo_tensor_with_dims_and_tensors(
            2, 0, [4, 4], indices, values, layout=torch.sparse_csr
        )


def _coo_payload(size, sparse_dim, dtype, device, nnz, seed):
    """Deterministic payload with duplicate and unsorted coordinates."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, max(nnz, 1), (nnz,), generator=g, dtype=torch.int64)
    indices = torch.stack([base % max(int(size[d]), 1) for d in range(sparse_dim)]).to(
        device
    )
    dense_shape = tuple(size[sparse_dim:])
    values = torch.randn((nnz,) + dense_shape, dtype=dtype, generator=g).to(device)
    return indices, values


def _native_backward(sparse_dim, dense_dim, size, indices, values, dtype, grad):
    """ATen reference: run the same constructor and take the values gradient."""
    ref_values = utils.to_reference(values.detach()).requires_grad_(True)
    ref_indices = utils.to_reference(indices)
    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        size,
        ref_indices,
        ref_values,
        dtype=dtype,
        layout=torch.sparse_coo,
        device=ref_indices.device,
    )
    (ref_grad,) = torch.autograd.grad(ref_out, ref_values, utils.to_reference(grad))
    return ref_grad


# (sparse_dim, dense_dim, size) triples: plain 1-D/2-D/3-D COO plus hybrid
# payloads that carry trailing dense dimensions.
BACKWARD_LAYOUTS = [
    (1, 0, [64]),
    (2, 0, [16, 32]),
    (3, 0, [4, 8, 16]),
    (2, 1, [16, 32, 7]),
    (1, 2, [32, 3, 5]),
]


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("sparse_dim,dense_dim,size", BACKWARD_LAYOUTS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward(
    sparse_dim, dense_dim, size, dtype
):
    """The values gradient is the dense upstream gradient at each coordinate."""
    indices, values = _coo_payload(
        size, sparse_dim, dtype, flag_gems.device, nnz=6, seed=11
    )
    values = values.requires_grad_(True)
    grad = torch.randn(tuple(size), dtype=dtype, device=flag_gems.device)

    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        sparse_dim,
        dense_dim,
        size,
        indices,
        values,
        dtype=dtype,
        layout=torch.sparse_coo,
    )
    (grad_values,) = torch.autograd.grad(out, values, grad)

    assert out.grad_fn is not None and out.requires_grad
    assert grad_values.shape == values.shape
    assert grad_values.dtype == values.dtype
    ref_grad = _native_backward(
        sparse_dim, dense_dim, size, indices, values.detach(), dtype, grad
    )
    utils.gems_assert_close(grad_values, ref_grad, dtype)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward_duplicates(dtype):
    """Duplicate and unsorted coordinates keep one gradient row per entry."""
    indices = torch.tensor(
        [[0, 0, 1, 1], [1, 1, 0, 2]], dtype=torch.int64, device=flag_gems.device
    )
    values = torch.randn(4, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [3, 3], indices, values, dtype=dtype, layout=torch.sparse_coo
    )
    assert out._nnz() == 4
    assert out.is_coalesced() is False

    grad = torch.randn(3, 3, dtype=dtype, device=flag_gems.device)
    (grad_values,) = torch.autograd.grad(out, values, grad)
    ref_grad = _native_backward(2, 0, [3, 3], indices, values.detach(), dtype, grad)

    utils.gems_assert_close(grad_values, utils.to_reference(ref_grad), dtype)
    # Duplicate coordinates are not summed here; the constructor stores them
    # as separate entries and each keeps its own gathered row.
    utils.gems_assert_close(
        grad_values[:2], utils.to_reference(grad[0].expand(2, 3)[:, 1]), dtype
    )


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward_strided_values(dtype):
    """A strided payload still gets a contiguous logical gradient."""
    storage = torch.randn(2, 12, dtype=dtype, device=flag_gems.device)
    values = storage[:, ::4].requires_grad_(True)
    assert tuple(values.shape) == (2, 3)
    assert not values.is_contiguous()
    indices = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=flag_gems.device)
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 1, [3, 4, 3], indices, values, dtype=dtype, layout=torch.sparse_coo
    )
    grad = torch.randn(3, 4, 3, dtype=dtype, device=flag_gems.device)
    (grad_values,) = torch.autograd.grad(out, values, grad)

    assert grad_values.shape == values.shape
    assert grad_values.is_contiguous()
    ref_grad = _native_backward(2, 1, [3, 4, 3], indices, values.detach(), dtype, grad)
    utils.gems_assert_close(grad_values, ref_grad, dtype)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward_noncontiguous_grad(
    dtype,
):
    """A non-contiguous upstream gradient is read through its own strides."""
    indices = torch.tensor(
        [[0, 1, 2], [2, 0, 1]], dtype=torch.int64, device=flag_gems.device
    )
    values = torch.randn(3, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [3, 3], indices, values, dtype=dtype, layout=torch.sparse_coo
    )
    grad = torch.randn(3, 6, dtype=dtype, device=flag_gems.device)[:, ::2]
    assert not grad.is_contiguous()
    (grad_values,) = torch.autograd.grad(out, values, grad)
    ref_grad = _native_backward(2, 0, [3, 3], indices, values.detach(), dtype, grad)
    utils.gems_assert_close(grad_values, ref_grad, dtype)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward_sparse_grad(dtype):
    """A sparse COO upstream gradient is densified exactly like ATen does."""
    indices = torch.tensor(
        [[0, 1, 1], [2, 0, 2]], dtype=torch.int64, device=flag_gems.device
    )
    values = torch.randn(3, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [3, 3], indices, values, dtype=dtype, layout=torch.sparse_coo
    )
    grad = torch.sparse_coo_tensor(
        indices,
        torch.randn(3, dtype=dtype, device=flag_gems.device),
        (3, 3),
    )
    (grad_values,) = torch.autograd.grad(out, values, grad)
    ref_grad = _native_backward(2, 0, [3, 3], indices, values.detach(), dtype, grad)
    utils.gems_assert_close(grad_values, ref_grad, dtype)


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
def test_sparse_coo_tensor_with_dims_and_tensors_backward_empty_payload():
    """An empty payload yields an empty gradient without launching a kernel."""
    indices = torch.zeros((2, 0), dtype=torch.int64, device=flag_gems.device)
    values = torch.randn(0, dtype=torch.float32, device=flag_gems.device)
    values = values.requires_grad_(True)
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [4, 4], indices, values, dtype=torch.float32, layout=torch.sparse_coo
    )
    (grad_values,) = torch.autograd.grad(
        out, values, torch.randn(4, 4, dtype=torch.float32, device=flag_gems.device)
    )
    assert grad_values.shape == (0,)
    assert grad_values.numel() == 0


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
def test_sparse_coo_tensor_with_dims_and_tensors_backward_indices_not_differentiable():
    """``indices`` are non-differentiable, so no gradient reaches them."""
    indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=flag_gems.device)
    assert not indices.requires_grad
    values = torch.randn(
        2, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [2, 2], indices, values, dtype=torch.float32, layout=torch.sparse_coo
    )
    grad = torch.tensor(
        [[0.0, 1.0], [2.0, 0.0]], dtype=torch.float32, device=flag_gems.device
    )
    (grad_values,) = torch.autograd.grad(out, values, grad)
    # Entry 0 sits at (0, 1) and entry 1 at (1, 0); the gradient follows the
    # payload order, not a sorted or deduplicated coordinate order.
    assert tuple(grad_values.shape) == (2,)
    utils.gems_assert_equal(
        grad_values,
        utils.to_reference(
            torch.tensor([1.0, 2.0], dtype=torch.float32, device=flag_gems.device)
        ),
    )


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward_matches_aten(dtype):
    """The candidate gradient is bitwise equal to ATen's generated formula."""
    indices = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 2]], dtype=torch.int64, device=flag_gems.device
    )
    values = torch.randn(4, dtype=dtype, device=flag_gems.device)
    grad = torch.randn(3, 3, dtype=dtype, device=flag_gems.device)

    ref_values = values.clone().requires_grad_(True)
    ref_out = torch.ops.aten._sparse_coo_tensor_with_dims_and_tensors(
        2,
        0,
        [3, 3],
        indices,
        ref_values,
        dtype=dtype,
        layout=torch.sparse_coo,
        device=flag_gems.device,
    )
    (ref_grad,) = torch.autograd.grad(ref_out, ref_values, grad)

    cand_values = values.clone().requires_grad_(True)
    cand_out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [3, 3], indices, cand_values, dtype=dtype, layout=torch.sparse_coo
    )
    (cand_grad,) = torch.autograd.grad(cand_out, cand_values, grad)

    # ``flag_gems.testing.assert_equal`` moves the candidate to the reference
    # side, so the ATen gradient is converted the same way as the rest of the
    # suite (`--ref cpu` keeps that side on the host).
    utils.gems_assert_equal(cand_grad, utils.to_reference(ref_grad))


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
def test_sparse_coo_tensor_with_dims_and_tensors_backward_kernel_attribution():
    """Exactly one fused gather kernel runs, replacing ATen's sparse_mask."""
    from torch.profiler import ProfilerActivity, profile

    indices = torch.tensor(
        [[0, 1, 1, 2], [1, 0, 2, 2]], dtype=torch.int64, device=flag_gems.device
    )
    values = torch.randn(
        4, dtype=torch.float32, device=flag_gems.device, requires_grad=True
    )
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [3, 3], indices, values, dtype=torch.float32, layout=torch.sparse_coo
    )
    grad = torch.randn(3, 3, dtype=torch.float32, device=flag_gems.device)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        torch.autograd.grad(out, values, grad)
        flag_gems.runtime.torch_device_fn.synchronize()

    names = [event.key for event in prof.key_averages() if event.count]
    # nnz=4 is not a tile multiple, so the masked tail variant serves it; the
    # specialized flat gather takes the ``_gather1/2/3`` names instead.
    fused = (
        "_gather_values_kernel",
        "_gather_tail",
        "_gather1",
        "_gather2",
        "_gather3",
        "_gather_dense",
    )
    assert any(name in n for name in fused for n in names), names
    assert not any("sparse_mask" in name for name in names), names


@pytest.mark.sparse_coo_tensor_with_dims_and_tensors_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_sparse_coo_tensor_with_dims_and_tensors_backward_public_primitive(dtype):
    """The exported backward primitive matches the autograd path."""
    indices = torch.tensor([[0, 2], [1, 0]], dtype=torch.int64, device=flag_gems.device)
    values = torch.randn(2, dtype=dtype, device=flag_gems.device, requires_grad=True)
    grad = torch.randn(3, 3, dtype=dtype, device=flag_gems.device)
    out = flag_gems.sparse_coo_tensor_with_dims_and_tensors(
        2, 0, [3, 3], indices, values, dtype=dtype, layout=torch.sparse_coo
    )
    (grad_values,) = torch.autograd.grad(out, values, grad)

    direct = flag_gems.sparse_coo_tensor_with_dims_and_tensors_backward(
        grad, indices, values.detach(), [3, 3], 2
    )
    utils.gems_assert_equal(direct, utils.to_reference(grad_values))
