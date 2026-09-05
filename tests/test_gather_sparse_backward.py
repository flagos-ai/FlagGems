import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DIMS = [0, 1, 2]


DTYPES = utils.FLOAT_DTYPES


# Core shapes exercised by INPUT_SHAPES (mirrors worktree CI branch).
INPUT_SHAPES = [(512, 128, 32), (1024, 64, 16), (128, 32, 256)]


# Integer grads are also supported by aten::_gather_sparse_backward.
INT_DTYPES = [torch.int64]


def _gather_sparse_backward(inp, dim, index, grad):
    return torch.ops.aten._gather_sparse_backward.default(inp, dim, index, grad)


def _make_gather_index(inp_shape, dim, duplicate_indices):
    """Build an `index` tensor (and matching `grad` shape) for gather backward.

    The index has the same rank as `self`; along `dim` it can be smaller than
    `self`, while the other dims match `self` (standard `torch.gather` shapes).
    """
    index_shape = list(inp_shape)
    index_shape[dim] = max(1, inp_shape[dim] // 2)
    index = torch.empty(tuple(index_shape), dtype=torch.long, device=flag_gems.device)

    size_dim = inp_shape[dim]
    if duplicate_indices:
        index.fill_(0)
        return index

    index_size_dim = index_shape[dim]
    m, n, o = index_shape
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, index_size_dim)
                index[tuple(ii)] = torch.randperm(size_dim, device=flag_gems.device)[
                    :index_size_dim
                ]
    return index


def _assert_sparse_equal(res, ref, dtype, atol=None):
    """Compare two sparse COO tensors element-wise against the uncoalesced
    layout produced by aten::_gather_sparse_backward.

    The kernel builds the sparse tensor in a canonical uncoalesced order
    (values == grad.flatten(), indices[dim] == index.flatten(), the remaining
    index rows are the unraveled linear id), which exactly matches aten, so a
    direct comparison of the uncoalesced indices/values is sufficient.
    """
    res = utils.to_cpu(res, ref)
    # Compare size and nnz.
    assert tuple(res.size()) == tuple(
        ref.size()
    ), f"size mismatch: {tuple(res.size())} vs {tuple(ref.size())}"
    assert res._nnz() == ref._nnz(), f"nnz mismatch: {res._nnz()} vs {ref._nnz()}"

    res_idx = res._indices()
    ref_idx = ref._indices()
    assert res_idx.dtype == torch.int64, f"indices dtype mismatch: {res_idx.dtype}"
    utils.gems_assert_equal(res_idx, ref_idx)

    res_val = res._values()
    ref_val = ref._values().to(dtype)
    if atol is not None:
        utils.gems_assert_close(res_val, ref_val, dtype, atol=atol)
    else:
        utils.gems_assert_close(res_val, ref_val, dtype)


@pytest.mark.gather_sparse_backward
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_gather_sparse_backward_float(inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    index = _make_gather_index(inp_shape, dim, duplicate_indices=False)
    grad = torch.randn(index.shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_grad = utils.to_reference(grad)
    ref_out = _gather_sparse_backward(ref_inp, dim, ref_index, ref_grad)

    res_out = flag_gems._gather_sparse_backward(inp, dim, index, grad)

    _assert_sparse_equal(res_out, ref_out, dtype)


@pytest.mark.gather_sparse_backward
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("duplicate_indices", [False, True])
def test_gather_sparse_backward_duplicate(inp_shape, dim, dtype, duplicate_indices):
    """With duplicate indices the uncoalesced layout still matches aten
    (nnz == index.numel()); the coalesced reduction is left to the downstream
    consumer."""
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    index = _make_gather_index(inp_shape, dim, duplicate_indices)
    grad = torch.randn(index.shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_grad = utils.to_reference(grad)
    ref_out = _gather_sparse_backward(ref_inp, dim, ref_index, ref_grad)

    res_out = flag_gems._gather_sparse_backward(inp, dim, index, grad)

    _assert_sparse_equal(res_out, ref_out, dtype)


@pytest.mark.gather_sparse_backward
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_gather_sparse_backward_int(inp_shape, dim, dtype):
    inp = torch.randn(inp_shape, dtype=torch.float32, device=flag_gems.device)
    index = _make_gather_index(inp_shape, dim, duplicate_indices=False)
    grad = torch.randint(-100, 100, index.shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_grad = utils.to_reference(grad)
    ref_out = _gather_sparse_backward(ref_inp, dim, ref_index, ref_grad)

    res_out = flag_gems._gather_sparse_backward(inp, dim, index, grad)

    _assert_sparse_equal(res_out, ref_out, dtype)


@pytest.mark.gather_sparse_backward
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((16,), 0),
        ((8, 0), 0),
        ((0, 8), 1),
        ((4, 0, 3), 1),
    ],
)
# Edge-case (zero-sized index / 1-D input) coverage uses float32 only to keep
# the case matrix focused on shape edge cases (mirrors worktree CI branch).
@pytest.mark.parametrize("dtype", [torch.float32])
def test_gather_sparse_backward_edge(shape, dim, dtype):
    """Edge cases: 1-D input and zero-sized index (nnz == 0)."""
    ndim = len(shape)
    if dim < 0:
        dim += ndim
    size_dim = shape[dim] if shape[dim] > 0 else 4
    index_shape = list(shape)
    index_shape[dim] = 1 if shape[dim] == 0 else max(1, shape[dim] // 2)
    if all(s == 0 for s in shape):
        index_shape = [0 if s == 0 else s for s in shape]
    # Build a valid index (empty when the gather dim is empty).
    if index_shape[dim] == 0:
        index = torch.empty(
            tuple(index_shape), dtype=torch.long, device=flag_gems.device
        )
        grad = torch.empty(tuple(index_shape), dtype=dtype, device=flag_gems.device)
    else:
        index = torch.randint(
            0, size_dim, tuple(index_shape), dtype=torch.long, device=flag_gems.device
        )
        grad = torch.randn(tuple(index_shape), dtype=dtype, device=flag_gems.device)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_grad = utils.to_reference(grad)
    ref_out = _gather_sparse_backward(ref_inp, dim, ref_index, ref_grad)

    res_out = flag_gems._gather_sparse_backward(inp, dim, index, grad)

    _assert_sparse_equal(res_out, ref_out, dtype)


@pytest.mark.gather_sparse_backward
@pytest.mark.parametrize("inp_shape", INPUT_SHAPES)
@pytest.mark.parametrize("dim", DIMS)
# Negative-dim normalization coverage uses float32 only to isolate dim-handling
# logic (mirrors worktree CI branch).
@pytest.mark.parametrize("dtype", [torch.float32])
def test_gather_sparse_backward_negative_dim(inp_shape, dim, dtype):
    """Negative dim is normalized to the equivalent positive dim."""
    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    index = _make_gather_index(inp_shape, dim, duplicate_indices=False)
    grad = torch.randn(index.shape, dtype=dtype, device=flag_gems.device)
    neg_dim = dim - len(inp_shape)

    ref_inp = utils.to_reference(inp)
    ref_index = utils.to_reference(index)
    ref_grad = utils.to_reference(grad)
    ref_out = _gather_sparse_backward(ref_inp, neg_dim, ref_index, ref_grad)

    res_out = flag_gems._gather_sparse_backward(inp, neg_dim, index, grad)

    _assert_sparse_equal(res_out, ref_out, dtype)
