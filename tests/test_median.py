import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [0, -1]
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    SHAPES = [(2, 32), (64,)]
else:
    DIM_LIST = [0, 1, -1, -2]
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    # Only use supported int dtypes (int32, int64)
    INT_DTYPES = [dtype for dtype in utils.INT_DTYPES if dtype in (torch.int32, torch.int64)]
    SHAPES = utils.REDUCTION_SHAPES

EMPTY_SHAPES = [(0, 5), (3, 0, 4), (2, 5, 0), (0,)]

# Additional shapes for comprehensive coverage
MEDIAN_SHAPES = (
    [(1,), (8,), (64,), (256, 128), (1024, 1024), (4096,)]
    if not cfg.QUICK_MODE
    else [(8,), (64,)]
)

SPECIAL_SHAPES = [(1,), (2,), (3,), (4,), (7,)] if not cfg.QUICK_MODE else [(1,), (3,)]


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPES + EMPTY_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_float(shape, dim, keepdim, dtype):
    """Test median with floating-point types across various shapes and dims."""
    rank = len(shape)
    is_empty_tensor = any(d == 0 for d in shape)

    if dim is not None:
        if rank == 0 or dim >= rank or dim < -rank:
            pytest.skip(f"Dimension {dim} is out of bound for shape {shape}")

    if is_empty_tensor:
        if dim is None:
            pytest.skip(
                "PyTorch reference requires dim specification for empty tensor."
            )
        dim_index = dim % rank
        if shape[dim_index] == 0:
            pytest.skip(
                f"PyTorch reference prohibits reduction on zero-sized dimension ({dim})."
            )

    if is_empty_tensor:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)

    if dim is not None:
        ref_values, ref_indices = torch.median(ref_inp, dim=dim, keepdim=keepdim)
        with flag_gems.use_gems():
            res_values, res_indices = torch.median(inp, dim=dim, keepdim=keepdim)

        utils.gems_assert_close(res_values, ref_values, dtype)
        # Verify index points to correct value
        if keepdim:
            gathered = torch.gather(inp, dim, res_indices)
        else:
            gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
        assert torch.equal(
            res_values.cpu(), gathered.cpu()
        ), "Index doesn't point to median value"
    else:
        # Global median (dim=None) - PyTorch returns scalar tensor
        ref_out = torch.median(ref_inp)
        with flag_gems.use_gems():
            res_out = torch.median(inp)

        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.median
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", INT_DTYPES)
def test_median_int(shape, dim, dtype):
    """Test median with integer types - values must be exactly equal."""
    rank = len(shape)
    if rank == 0 or dim >= rank or dim < -rank:
        pytest.skip(f"Dimension {dim} is out of bound for shape {shape}")

    min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    inp = torch.randint(min_v, max_v, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    assert torch.equal(
        res_values.cpu(), ref_values.cpu()
    ), f"Values mismatch for int dtype: {res_values} vs {ref_values}"
    # Verify index points to correct value (indices may differ for duplicates)
    gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(
        res_values.cpu(), gathered.cpu()
    ), "Index doesn't point to median value for int dtype"


@pytest.mark.median
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
@pytest.mark.parametrize("dim", [-1])
def test_median_small_shapes(shape, dim):
    """Test with small tensor shapes (1-7 elements)."""
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_values, ref_indices = torch.median(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_median_duplicates(dim):
    """Test with duplicate values to verify index selection.

    For duplicate median values, different implementations may return
    different indices. We verify that:
    1. The returned value is correct
    2. The returned index points to a valid median value in the original tensor
    """
    inp = torch.tensor(
        [[3.0, 3.0, 3.0, 1.0, 2.0], [5.0, 5.0, 1.0, 5.0, 5.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    # Use GPU reference to avoid CPU/GPU index differences
    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    # Values must match exactly
    utils.gems_assert_close(res_values, ref_values, torch.float32)
    # For indices: verify the returned index points to the correct value
    gathered_vals = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(
        res_values.cpu(), gathered_vals.cpu()
    ), f"Index doesn't point to median value: gathered={gathered_vals} vs median={res_values}"


@pytest.mark.median
def test_median_all_same():
    """Test when all elements are the same - any index is valid."""
    inp = torch.full((10, 20), 42.0, dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=1)

    assert torch.all(res_values == 42.0)
    # Any index is valid when all values are the same
    gathered = torch.gather(inp, 1, res_indices.unsqueeze(1)).squeeze(1)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_median_non_contiguous(dim):
    """Test with non-contiguous tensors (transpose)."""
    inp = torch.randn(32, 64, dtype=torch.float32, device=flag_gems.device)
    inp_t = inp.t()  # Make non-contiguous

    ref_values, ref_indices = torch.median(inp_t, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp_t, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    gathered = torch.gather(inp_t, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
def test_median_single_element():
    """Test with single element tensor."""
    inp = torch.tensor([42.0], dtype=torch.float32, device=flag_gems.device)

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=0)

    assert res_values.item() == 42.0
    assert res_indices.item() == 0


@pytest.mark.median
@pytest.mark.parametrize(
    "shape",
    [(2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)],
)
@pytest.mark.parametrize("dim", [0, -1])
def test_median_high_dim(shape, dim):
    """Test with 3D-5D tensors."""
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_with_nan(dim):
    """Test median with NaN values.

    PyTorch behavior: NaN propagates in median (unlike np.nanmedian).
    If any element is NaN, the median should be NaN.
    """
    inp = torch.tensor(
        [1.0, 2.0, float("nan"), 4.0, 5.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    # NaN should propagate
    assert torch.isnan(res_values), f"Expected NaN but got {res_values}"


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_with_inf(dim):
    """Test median with Inf values."""
    inp = torch.tensor(
        [1.0, 2.0, float("inf"), 4.0, 5.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_with_neg_inf(dim):
    """Test median with -Inf values."""
    inp = torch.tensor(
        [1.0, 2.0, float("-inf"), 4.0, 5.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_all_nan(dim):
    """Test median when all elements are NaN."""
    inp = torch.tensor(
        [float("nan"), float("nan"), float("nan")],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    assert torch.isnan(res_values), f"Expected NaN but got {res_values}"


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_negative_values(dim):
    """Test median with negative values."""
    inp = torch.tensor(
        [-5.0, -1.0, -3.0, -2.0, -4.0],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    assert res_values.item() == -3.0, f"Expected -3.0 but got {res_values}"


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_large_range(dim):
    """Test median with very large and very small values."""
    inp = torch.tensor(
        [1e-30, 1e30, 1.0, -1e30, -1e-30],
        dtype=torch.float32,
        device=flag_gems.device,
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_values, ref_values, torch.float32)
    gathered = torch.gather(inp, dim, res_indices.unsqueeze(dim)).squeeze(dim)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_two_elements(dim):
    """Test median with exactly two elements (even case)."""
    inp = torch.tensor([3.0, 1.0], dtype=torch.float32, device=flag_gems.device)

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    # For even N=2, PyTorch returns the smaller element (index 0 in sorted)
    utils.gems_assert_close(res_values, ref_values, torch.float32)
    assert res_values.item() == 1.0


@pytest.mark.median
@pytest.mark.parametrize("dim", [-1])
def test_median_four_elements(dim):
    """Test median with exactly four elements (even case)."""
    inp = torch.tensor(
        [4.0, 1.0, 3.0, 2.0], dtype=torch.float32, device=flag_gems.device
    )

    ref_values, ref_indices = torch.median(inp, dim=dim)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=dim)

    # Sorted: [1, 2, 3, 4], median = 2 (smaller of two middle elements)
    utils.gems_assert_close(res_values, ref_values, torch.float32)
    assert res_values.item() == 2.0


@pytest.mark.median
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_median_half_precision(dtype):
    """Test median with half-precision types."""
    inp = torch.randn(64, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_values, ref_indices = torch.median(ref_inp, dim=-1)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=-1)

    utils.gems_assert_close(res_values, ref_values, dtype)
    gathered = torch.gather(inp, -1, res_indices.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
def test_median_int64():
    """Test median with int64 dtype."""
    inp = torch.randint(
        -1000, 1000, (32, 64), dtype=torch.int64, device=flag_gems.device
    )

    ref_values, ref_indices = torch.median(inp, dim=-1)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=-1)

    assert torch.equal(res_values.cpu(), ref_values.cpu())
    gathered = torch.gather(inp, -1, res_indices.unsqueeze(-1)).squeeze(-1)
    assert torch.equal(res_values.cpu(), gathered.cpu())


@pytest.mark.median
def test_median_int32_boundary():
    """Test median with int32 boundary values."""
    inp = torch.tensor(
        [torch.iinfo(torch.int32).max, torch.iinfo(torch.int32).min, 0, 1, -1],
        dtype=torch.int32,
        device=flag_gems.device,
    )

    ref_values, ref_indices = torch.median(inp, dim=0)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=0)

    assert torch.equal(res_values.cpu(), ref_values.cpu())


@pytest.mark.median
def test_median_keepdim_true():
    """Test keepdim=True preserves reduced dimension."""
    inp = torch.randn(4, 8, 16, dtype=torch.float32, device=flag_gems.device)

    ref_values, ref_indices = torch.median(inp, dim=1, keepdim=True)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=1, keepdim=True)

    assert res_values.shape == (4, 1, 16)
    assert res_indices.shape == (4, 1, 16)
    utils.gems_assert_close(res_values, ref_values, torch.float32)


@pytest.mark.median
def test_median_keepdim_false():
    """Test keepdim=False removes reduced dimension."""
    inp = torch.randn(4, 8, 16, dtype=torch.float32, device=flag_gems.device)

    ref_values, ref_indices = torch.median(inp, dim=1, keepdim=False)
    with flag_gems.use_gems():
        res_values, res_indices = torch.median(inp, dim=1, keepdim=False)

    assert res_values.shape == (4, 16)
    assert res_indices.shape == (4, 16)
    utils.gems_assert_close(res_values, ref_values, torch.float32)
