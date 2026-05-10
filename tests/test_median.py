import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    ALL_DTYPES = [torch.float32]
    SIZES_SMALL = [(1, 1), (8, 8)]
    SIZES_REGULAR = [(64, 64)]
    SIZES_LARGE = [(1024, 1024)]
    DIMS_2D = [0, -1]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INT_DTYPES = [torch.int32, torch.int64]
    ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES
    SIZES_SMALL = [(8, 8)]
    SIZES_REGULAR = [(64, 64)]
    SIZES_LARGE = [(1024, 1024)]
    DIMS_2D = [0, -1]
    KEEPDIM = [True, False]

SIZES_1D = [(8,), (64,)]
SIZES_2D = SIZES_SMALL + SIZES_REGULAR + SIZES_LARGE
SIZES_3D = [(8, 16, 32)]
SIZES_4D = [(2, 4, 8, 16)]


# ===========================================================================
# median() default: scalar median of flattened input
# ===========================================================================
@pytest.mark.median
@pytest.mark.parametrize("shape", SIZES_1D + SIZES_2D + SIZES_3D[:1])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_default(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)
    ref_out = torch.median(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.median(inp)
    utils.gems_assert_equal(res_out, ref_out)


# ===========================================================================
# median.dim() — input sizes: small (1×1, 8×8)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", SIZES_SMALL)
@pytest.mark.parametrize("dim", DIMS_2D)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_dim_small(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — input sizes: regular (64×64, 256×256)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", SIZES_REGULAR)
@pytest.mark.parametrize("dim", DIMS_2D)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_dim_regular(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — input sizes: large (1024×1024, 4096×4096)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", SIZES_LARGE)
@pytest.mark.parametrize("dim", DIMS_2D)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_dim_large(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — 1D input
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", SIZES_1D)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_1d(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — 3D input (covers dim=0,1,2 and dim=-1,-2,-3)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", SIZES_3D)
@pytest.mark.parametrize("dim", [0, 1, 2, -1, -2, -3])
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_3d(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — 4D input (covers high-dimensional tensors)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", SIZES_4D)
@pytest.mark.parametrize("dim", [0, 2, -1, -4])
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_4d(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — duplicate values (tie-breaking correctness)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", [(16, 32), (8, 64)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_ties(shape, dim, dtype):
    # Use integer values to create guaranteed duplicates
    inp = torch.randint(0, 5, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.median(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim)
    utils.gems_assert_equal(res_out.values, ref_out.values)
    # For ties, index must point to a value equal to the median
    res_idx = res_out.indices
    if res_idx.ndim < inp.ndim:
        res_idx = res_idx.unsqueeze(dim)
    gathered = inp.gather(dim, res_idx.to(inp.device))
    if res_out.values.ndim < gathered.ndim:
        gathered = gathered.squeeze(dim)
    utils.gems_assert_equal(gathered, res_out.values)


# ===========================================================================
# median.dim() — large-dimension path (N > MAX_BITONIC_M, triggers kthvalue)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", [(32, 2048), (16, 4096)])
@pytest.mark.parametrize("dim", [-1])
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_large_n(shape, dim, keepdim, dtype):
    _run_median_dim_test(shape, dim, keepdim, dtype)


# ===========================================================================
# median.dim() — edge cases: N=1 (trivial median)
# ===========================================================================
@pytest.mark.median_dim
@pytest.mark.parametrize("shape", [(64, 1), (1, 128)])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_edge_n1(shape, dim, dtype):
    _run_median_dim_test(shape, dim, keepdim=False, dtype=dtype)


# ===========================================================================
# median.out — out variant of default median
# ===========================================================================
@pytest.mark.median_out
@pytest.mark.parametrize("shape", SIZES_1D + SIZES_2D[:3])
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_out(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    # Reference
    ref_out = torch.empty([], dtype=dtype, device=ref_inp.device)
    torch.ops.aten.median.out(ref_inp, out=ref_out)

    # FlagGems
    out = torch.empty([], dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        torch.ops.aten.median.out(inp, out=out)
    utils.gems_assert_equal(out, ref_out)


# ===========================================================================
# median.dim_values — out variant of dim median
# ===========================================================================
@pytest.mark.median_dim_values
@pytest.mark.parametrize("shape", SIZES_SMALL + SIZES_REGULAR[:1])
@pytest.mark.parametrize("dim", DIMS_2D)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_dim_values(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    # Compute expected output shape
    _dim = dim % inp.ndim
    out_shape = list(inp.shape)
    out_shape[_dim] = 1

    if not keepdim:
        ref_out = torch.median(ref_inp, dim=dim, keepdim=False)
        out_shape_ref = list(inp.shape)
        out_shape_ref.pop(_dim)
    else:
        ref_out = torch.median(ref_inp, dim=dim, keepdim=True)

    # Reference via ATen
    ref_values = torch.empty(
        out_shape if keepdim else out_shape_ref,
        dtype=dtype,
        device=ref_inp.device,
    )
    ref_indices = torch.empty(
        out_shape if keepdim else out_shape_ref,
        dtype=torch.long,
        device=ref_inp.device,
    )
    torch.ops.aten.median.dim_values(
        ref_inp, dim, keepdim, values=ref_values, indices=ref_indices
    )

    # FlagGems
    values = torch.empty(
        out_shape if keepdim else out_shape_ref,
        dtype=dtype,
        device=flag_gems.device,
    )
    indices = torch.empty(
        out_shape if keepdim else out_shape_ref,
        dtype=torch.long,
        device=flag_gems.device,
    )
    with flag_gems.use_gems():
        torch.ops.aten.median.dim_values(
            inp, dim, keepdim, values=values, indices=indices
        )

    utils.gems_assert_equal(values, ref_out.values)
    # Validate index correctness
    res_idx = indices
    if res_idx.ndim < inp.ndim:
        res_idx = res_idx.unsqueeze(dim)
    gathered = inp.gather(dim, res_idx.to(inp.device))
    if not keepdim and gathered.ndim > values.ndim:
        gathered = gathered.squeeze(dim)
    utils.gems_assert_equal(gathered, values)


def _run_median_dim_test(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)
    ref_out = torch.median(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out.values, ref_out.values)
    # Validate index correctness
    res_idx = res_out.indices
    if res_idx.ndim < inp.ndim:
        res_idx = res_idx.unsqueeze(dim)
    gathered = inp.gather(dim, res_idx.to(inp.device))
    if not keepdim and gathered.ndim > res_out.values.ndim:
        gathered = gathered.squeeze(dim)
    utils.gems_assert_equal(gathered, res_out.values)
