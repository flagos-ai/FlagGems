import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device
VENDOR = flag_gems.vendor_name

# ``torch._lu_with_info`` is only well-supported with double precision on the
# accelerator in the native baseline; restrict the dtype set per vendor
# (float64 is exercised on nvidia, float32 elsewhere). Half precision is not
# supported by the LU factorization kernels.
if VENDOR == "nvidia":
    # CUDA LU only exposes float32/float64 baselines; half precision unsupported.
    _TEST_DTYPES = [torch.float32, torch.float64]
else:
    _TEST_DTYPES = [torch.float32]

# pivot=False is only supported on CUDA
if utils.TO_CPU:
    _PIVOT_VALUES = [True]
elif DEVICE == "cuda":
    _PIVOT_VALUES = [True, False]
else:
    _PIVOT_VALUES = [True]


def _make_singular(shape, device, dtype):
    """Construct a batch of matrices where one element is exactly singular.

    Setting the entire last row to zero guarantees that, after partial
    pivoting processes all preceding columns, the final pivot is exactly zero
    regardless of floating-point accumulation order. This produces a
    deterministic ``info == k`` (the 1-indexed last position) in both the
    native cuSOLVER reference and the Triton factorization, so the two can be
    compared exactly. (Duplicate-row constructions, by contrast, place the
    zero pivot at an accumulation-order-dependent diagonal position, making an
    exact ``info`` comparison fragile across implementations.)
    """
    a = torch.randn(shape, dtype=dtype, device=device)
    a[..., -1, :] = 0.0
    return a


def _make_input(shape, pivot, device, dtype):
    """Generate a test matrix suitable for the given pivot mode.

    For pivot=True, a random matrix is used (partial pivoting handles stability).
    For pivot=False, the matrix is constructed as L @ U where L has unit diagonal
    to guarantee a stable no-pivot LU factorization exists.
    """
    if pivot:
        return torch.randn(shape, dtype=dtype, device=device)

    *batch, m, n = shape
    k = min(m, n)
    scaling = k**-0.5
    L = (torch.randn(*batch, m, k, dtype=dtype, device=device) * scaling).tril()
    L.diagonal(dim1=-2, dim2=-1).fill_(1.0)
    U = torch.randn(*batch, k, n, dtype=dtype, device=device).triu()
    # Make U's diagonal large for numerical stability
    U.diagonal(dim1=-2, dim2=-1).abs_().add_(1.0)
    return L @ U


def _unpack_lu_no_pivot(lu):
    m, n = lu.shape[-2], lu.shape[-1]
    k = min(m, n)
    ll = lu[..., :, :k].tril()
    diag = torch.arange(k, device=lu.device)
    ll[..., diag, diag] = 1
    u = lu[..., :k, :].triu()
    return ll, u


@pytest.mark.lu_with_info
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize(
    "shape", [(64, 64), (256, 256), (512, 512), (1024, 1024), (8, 128, 128)]
)
@pytest.mark.parametrize("pivot", _PIVOT_VALUES)
def test_lu_with_info(shape, dtype, pivot):
    if DEVICE != "cuda" and not pivot:
        pytest.skip("pivot=False only supported on CUDA")
    inp = _make_input(shape, pivot, DEVICE, dtype)
    ref_inp = utils.to_reference(inp)
    ref_lu, ref_pivots, ref_info = torch._lu_with_info(ref_inp, pivot=pivot)
    res_lu, res_pivots, res_info = flag_gems._lu_with_info(inp, pivot=pivot)
    # LU has the same shape/dtype as the input
    assert res_lu.shape == ref_lu.shape
    assert res_lu.dtype == ref_lu.dtype
    # pivots are int32, shape (batch, min(m,n))
    assert res_pivots.dtype == torch.int32
    assert res_pivots.shape == ref_pivots.shape
    assert torch.all(res_pivots >= 1)
    assert torch.all(res_pivots <= shape[-2])
    # info is int32, shape == batch shape
    assert res_info.dtype == torch.int32
    assert res_info.shape == ref_info.shape
    # info is 0 (non-singular) for these well-conditioned inputs, or in [0, k]
    m, n = shape[-2], shape[-1]
    k = min(m, n)
    assert (res_info >= 0).all()
    assert (res_info <= k).all()
    # Validate the factorization by reconstruction rather than the raw LU
    # storage: different LU implementations (Triton panel-blocked vs cuSOLVER)
    # produce mathematically equivalent factorizations with different floating
    # point accumulation orders, so the stored LU entries need not match the
    # reference bit-for-bit. Reconstructing A = P @ L @ U from each and
    # comparing those is the mathematically meaningful correctness check,
    # matching the precedent in test_linalg_lu_factor.
    torch.backends.cuda.matmul.allow_tf32 = False
    if pivot:
        res_p, res_l, res_u = torch.lu_unpack(res_lu, res_pivots)
        ref_p, ref_l, ref_u = torch.lu_unpack(ref_lu, ref_pivots)
        reconstructed = res_p @ res_l @ res_u
        ref_reconstructed = ref_p @ ref_l @ ref_u
    else:
        res_l, res_u = _unpack_lu_no_pivot(res_lu)
        ref_l, ref_u = _unpack_lu_no_pivot(ref_lu)
        reconstructed = res_l @ res_u
        ref_reconstructed = ref_l @ ref_u
    utils.gems_assert_close(reconstructed, ref_reconstructed, dtype, reduce_dim=k)
    # info must agree with the reference for well-conditioned (non-singular) inputs
    utils.gems_assert_equal(res_info, ref_info)


@pytest.mark.lu_with_info
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("shape", [(64, 64), (8, 128, 128)])
def test_lu_with_info_singular(shape, dtype):
    """Singular matrices must produce a nonzero ``info`` matching the reference."""
    inp = _make_singular(shape, DEVICE, dtype)
    ref_inp = utils.to_reference(inp)
    ref_lu, ref_pivots, ref_info = torch._lu_with_info(ref_inp, pivot=True)
    res_lu, res_pivots, res_info = flag_gems._lu_with_info(inp, pivot=True)
    assert res_info.dtype == torch.int32
    # The all-zero-last-row construction is singular, so every batch element
    # must report a nonzero info (the position of the zero U pivot) in both
    # the reference and GEMS, and the two must agree exactly.
    assert (ref_info > 0).all()
    assert (res_info > 0).all()
    utils.gems_assert_equal(res_info, ref_info)


@pytest.mark.lu_with_info
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("shape", [(6, 6), (4, 6, 6), (2, 3, 6, 6)])
def test_lu_with_info_info_shape(shape, dtype):
    """Verify the info tensor shape for non-batched, batched and higher-rank inputs."""
    full_shape = shape
    inp = torch.randn(full_shape, dtype=dtype, device=DEVICE)
    ref_inp = utils.to_reference(inp)
    ref_lu, ref_pivots, ref_info = torch._lu_with_info(ref_inp, pivot=True)
    res_lu, res_pivots, res_info = flag_gems._lu_with_info(inp, pivot=True)
    assert res_info.shape == ref_info.shape
