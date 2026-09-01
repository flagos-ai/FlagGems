import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Sparse COO tensor shapes used for testing: (n_sparse_dims, nnz, size).
# nnz = number of stored (index, value) pairs.
SPARSE_SHAPES = (
    [(2, 8, (4, 4))]
    if utils.QUICK_MODE
    else [
        (2, 1, (2, 2)),
        (2, 4, (3, 2)),
        (2, 16, (8, 8)),
        (2, 64, (16, 16)),
        (3, 32, (5, 7, 9)),
    ]
)


def _make_sparse_coo(nnd, nnz, size, dtype, device, seed=0):
    """Build a sparse COO tensor with unique, sorted indices so that marking it
    ``coalesced`` is a valid assertion (matching the native op's contract)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    # Sample distinct multi-indices from the flattened index space (guaranteed
    # unique) and then sort them lexicographically.
    numel = 1
    for s in size[:nnd]:
        numel *= s
    flat = torch.randperm(numel, generator=gen)[:nnz]
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    for d in reversed(range(nnd)):
        indices[d] = flat % size[d]
        flat = flat // size[d]
    # Lexicographic order: stable-sort by the last row, then the second-to-last,
    # ..., up to the first row.
    perm = torch.arange(nnz)
    for d in reversed(range(nnd)):
        perm = perm[torch.argsort(indices[d][perm], stable=True)]
    indices = indices[:, perm].contiguous()
    values = torch.randn(nnz, dtype=dtype, generator=gen).to(device)
    indices = indices.to(device)
    return torch.sparse_coo_tensor(indices, values, size)


@pytest.mark.coalesced_
@pytest.mark.parametrize("nnd, nnz, size", SPARSE_SHAPES)
@pytest.mark.parametrize("coalesced", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_coalesced_(nnd, nnz, size, coalesced, dtype):
    # Build independent tensors for the reference (native) and the GEMS path so
    # the two in-place mutations don't alias each other.
    inp1 = _make_sparse_coo(nnd, nnz, size, dtype, flag_gems.device, seed=7)
    ref_inp1 = _make_sparse_coo(nnd, nnz, size, dtype, flag_gems.device, seed=7)
    # Move the reference onto the configured reference device (CPU under
    # quick-cpu mode) so ``gems_assert_close`` can compare across devices.
    ref_inp1 = utils.to_reference(ref_inp1)

    ref_out = ref_inp1._coalesced_(coalesced)
    with flag_gems.use_gems():
        out1 = inp1._coalesced_(coalesced)

    # The op is an in-place metadata mutation that returns ``self``.
    assert out1 is inp1
    assert ref_out is ref_inp1
    # The coalesced flag must match the requested value and the reference.
    assert out1.is_coalesced() == coalesced
    assert out1.is_coalesced() == ref_out.is_coalesced()
    # The underlying data is untouched.
    utils.gems_assert_close(inp1._values(), ref_inp1._values(), dtype)
    utils.gems_assert_equal(out1._indices(), ref_out._indices())


@pytest.mark.coalesced_
@pytest.mark.parametrize("nnd, nnz, size", SPARSE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_coalesced__toggle(nnd, nnz, size, dtype):
    # Setting then un-setting must leave the flag as the last value requested.
    inp1 = _make_sparse_coo(nnd, nnz, size, dtype, flag_gems.device, seed=11)
    ref_inp1 = _make_sparse_coo(nnd, nnz, size, dtype, flag_gems.device, seed=11)
    # Move the reference onto the configured reference device (CPU under
    # quick-cpu mode) so ``gems_assert_close`` can compare across devices.
    ref_inp1 = utils.to_reference(ref_inp1)

    with flag_gems.use_gems():
        out1 = inp1._coalesced_(True)._coalesced_(False)._coalesced_(True)
    ref_out = ref_inp1._coalesced_(True)._coalesced_(False)._coalesced_(True)

    assert out1 is inp1
    assert out1.is_coalesced() is True
    assert out1.is_coalesced() == ref_out.is_coalesced()
    utils.gems_assert_close(inp1._values(), ref_inp1._values(), dtype)


@pytest.mark.coalesced_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_coalesced__invalid_layout(dtype):
    # ``_coalesced_`` is only defined for sparse COO tensors, mirroring the
    # native behavior which raises on a strided (dense) tensor.
    inp = torch.randn(4, 4, dtype=dtype, device=flag_gems.device)
    ref_err = None
    try:
        inp_ref = inp.clone()
        inp_ref._coalesced_(True)
    except RuntimeError as e:
        ref_err = e
    assert ref_err is not None, "native _coalesced_ should reject dense tensors"

    with flag_gems.use_gems():
        with pytest.raises(RuntimeError):
            inp._coalesced_(True)
