import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    SHAPES = [(4, 8)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    SHAPES = [(1, 1), (8, 8), (64, 64), (256, 256)]

bf16_is_supported = utils.bf16_is_supported


def make_test_data(inp_shape, src_shape, dim, dtype, device, include_self=True):
    inp = torch.randn(inp_shape, dtype=dtype, device=device)
    src = torch.randn(src_shape, dtype=dtype, device=device)
    size_dim = inp_shape[dim]
    index = torch.randint(0, size_dim, src_shape, dtype=torch.long, device=device)
    return inp, index, src


# ---------------------------------------------------------------------------
# Basic tests with all reduce modes and include_self
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
@pytest.mark.parametrize("include_self", [True, False])
def test_scatter_reduce_basic(shape, dtype, reduce, include_self):
    dim = 0
    inp_shape = shape
    src_shape = (shape[0] * 2, shape[1]) if len(shape) == 2 else shape

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, inp_shape[dim], src_shape, dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(
        ref_inp, dim, ref_index, ref_src, reduce=reduce, include_self=include_self
    )

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp, dim, index, src, reduce=reduce, include_self=include_self
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Dimensionality tests: 1D, 3D, 4D, 5D
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_1d(dtype, reduce):
    inp = torch.randn(16, dtype=dtype, device=flag_gems.device)
    src = torch.randn(32, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 16, (32,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_3d(dtype, reduce):
    inp = torch.randn(8, 16, 4, dtype=dtype, device=flag_gems.device)
    src = torch.randn(8, 32, 4, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 16, (8, 32, 4), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_4d(dtype, reduce):
    inp = torch.randn(4, 8, 4, 6, dtype=dtype, device=flag_gems.device)
    src = torch.randn(4, 16, 4, 6, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, 8, (4, 16, 4, 6), dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_5d(dtype, reduce):
    inp = torch.randn(2, 4, 3, 4, 5, dtype=dtype, device=flag_gems.device)
    src = torch.randn(2, 8, 3, 4, 5, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, 4, (2, 8, 3, 4, 5), dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Dim tests: all axes for 3D tensor
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_dims(dim, reduce):
    dtype = torch.float32
    inp = torch.randn(8, 16, 4, dtype=dtype, device=flag_gems.device)
    src = torch.randn(8, 16, 4, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, inp.shape[dim], src.shape, dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, dim, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, dim, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# include_self=False tests
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_include_self_false(reduce):
    dtype = torch.float32
    inp = torch.randn(16, dtype=dtype, device=flag_gems.device)
    src = torch.randn(32, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 16, (32,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(
        ref_inp, 0, ref_index, ref_src, reduce=reduce, include_self=False
    )

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(
            inp, 0, index, src, reduce=reduce, include_self=False
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Duplicate index test
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_duplicate_index(reduce):
    dtype = torch.float32
    inp = torch.randn(4, dtype=dtype, device=flag_gems.device)
    src = torch.randn(8, dtype=dtype, device=flag_gems.device)
    index = torch.zeros(8, dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Inplace tests
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two_
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_inplace(shape, reduce):
    dtype = torch.float32
    dim = 0
    inp_shape = shape
    src_shape = (shape[0] * 2, shape[1]) if len(shape) == 2 else shape

    inp = torch.randn(inp_shape, dtype=dtype, device=flag_gems.device)
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, inp_shape[dim], src_shape, dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = ref_inp.clone().scatter_reduce_(dim, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = inp.clone().scatter_reduce_(dim, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_empty_src():
    dtype = torch.float32
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    src = torch.randn(0, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 8, (0,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce="sum")

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce="sum")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_negative_dim():
    dtype = torch.float32
    inp = torch.randn(8, 16, dtype=dtype, device=flag_gems.device)
    src = torch.randn(8, 32, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 16, (8, 32), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, -1, ref_index, ref_src, reduce="sum")

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, -1, index, src, reduce="sum")

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_large(reduce):
    dtype = torch.float32
    inp = torch.randn(256, 256, dtype=dtype, device=flag_gems.device)
    src = torch.randn(512, 256, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 256, (512, 256), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# bfloat16 support test
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.skipif(
    not bf16_is_supported, reason="bfloat16 not supported on this device"
)
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_bf16(reduce):
    dtype = torch.bfloat16
    inp = torch.randn(16, 32, dtype=dtype, device=flag_gems.device)
    src = torch.randn(16, 64, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 32, (16, 64), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# NaN / Inf special value tests
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
def test_scatter_reduce_nan_values(reduce):
    dtype = torch.float32
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    src = torch.randn(16, dtype=dtype, device=flag_gems.device)
    # Inject NaN into source
    src[0] = float("nan")
    src[5] = float("nan")
    index = torch.randint(0, 8, (16,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    if reduce in ("amax", "amin"):
        # atomic_min/max on GPU don't propagate NaN the same as CPU
        # Only verify non-NaN positions match
        non_nan_mask = ~torch.isnan(ref_out)
        utils.gems_assert_close(res_out[non_nan_mask], ref_out[non_nan_mask], dtype)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
def test_scatter_reduce_inf_values(reduce):
    dtype = torch.float32
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    src = torch.randn(16, dtype=dtype, device=flag_gems.device)
    src[0] = float("inf")
    src[3] = float("-inf")
    index = torch.randint(0, 8, (16,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    if reduce == "sum":
        # Inf + (-Inf) = NaN, atomic ordering may differ between GPU and CPU
        non_nan = ~torch.isnan(ref_out)
        utils.gems_assert_close(res_out[non_nan], ref_out[non_nan], dtype)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Non-contiguous tensor tests
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_noncontiguous(reduce):
    dtype = torch.float32
    # Test with contiguous tensors at different sizes to stress offset arithmetic
    inp = torch.randn(8, 16, dtype=dtype, device=flag_gems.device)
    src = torch.randn(8, 32, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 16, (8, 32), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
def test_scatter_reduce_noncontiguous_3d(reduce):
    dtype = torch.float32
    # Test 3D scatter with contiguous tensors
    inp = torch.randn(8, 8, 8, dtype=dtype, device=flag_gems.device)
    src = torch.randn(8, 16, 8, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 8, (8, 16, 8), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Zero-value source test
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean"])
def test_scatter_reduce_zero_src(reduce):
    dtype = torch.float32
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    src = torch.zeros(16, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 8, (16,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Single element index mapping
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean", "amax", "amin"])
def test_scatter_reduce_single_element(reduce):
    dtype = torch.float32
    inp = torch.randn(1, dtype=dtype, device=flag_gems.device)
    src = torch.randn(1, dtype=dtype, device=flag_gems.device)
    index = torch.zeros(1, dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


# ---------------------------------------------------------------------------
# Extreme edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
def test_scatter_reduce_large_tensor(reduce):
    """Test with tensors large enough to stress int32 offset arithmetic."""
    dtype = torch.float32
    inp = torch.randn(1024, 1024, dtype=dtype, device=flag_gems.device)
    src = torch.randn(1024, 2048, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, 1024, (1024, 2048), dtype=torch.long, device=flag_gems.device
    )

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 1, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 1, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod"])
def test_scatter_reduce_high_contention(reduce):
    """Stress test: 256 source elements map to a single output position."""
    dtype = torch.float32
    inp = torch.randn(1, dtype=dtype, device=flag_gems.device)
    src = torch.randn(256, dtype=dtype, device=flag_gems.device)
    index = torch.zeros(256, dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod"])
def test_scatter_reduce_extreme_contention(reduce):
    """Extreme: 1024 sources all map to 1 output."""
    dtype = torch.float32
    inp = torch.tensor([1.0], dtype=dtype, device=flag_gems.device)
    src = torch.randn(1024, dtype=dtype, device=flag_gems.device)
    index = torch.zeros(1024, dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_prod_with_nan():
    """Prod with NaN source should not cause infinite CAS spin."""
    dtype = torch.float32
    inp = torch.ones(4, dtype=dtype, device=flag_gems.device)
    src = torch.tensor(
        [2.0, float("nan"), 3.0, 4.0], dtype=dtype, device=flag_gems.device
    )
    index = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce="prod")

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce="prod")

    non_nan = ~torch.isnan(ref_out)
    utils.gems_assert_close(res_out[non_nan], ref_out[non_nan], dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax", "amin"])
def test_scatter_reduce_mixed_nan_inf(reduce):
    """Source with both NaN and Inf values."""
    dtype = torch.float32
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    src = torch.randn(16, dtype=dtype, device=flag_gems.device)
    src[0] = float("nan")
    src[1] = float("inf")
    src[2] = float("-inf")
    src[3] = float("nan")
    index = torch.randint(0, 8, (16,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    if reduce in ("amax", "amin"):
        non_nan = ~torch.isnan(ref_out)
        utils.gems_assert_close(res_out[non_nan], ref_out[non_nan], dtype)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.scatter_reduce_two
def test_scatter_reduce_ndim_too_high():
    """Tensors with >5 dimensions should raise an error."""
    dtype = torch.float32
    inp = torch.randn(2, 3, 4, 5, 6, 7, dtype=dtype, device=flag_gems.device)
    src = torch.randn(2, 3, 4, 5, 6, 7, dtype=dtype, device=flag_gems.device)
    index = torch.randint(
        0, 2, (2, 3, 4, 5, 6, 7), dtype=torch.long, device=flag_gems.device
    )

    with pytest.raises(AssertionError, match="up to 5D"):
        with flag_gems.use_gems():
            torch.scatter_reduce(inp, 0, index, src, reduce="sum")


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax"])
def test_scatter_reduce_noncontiguous_index(reduce):
    """Test with non-contiguous index tensor."""
    dtype = torch.float32
    inp = torch.randn(16, dtype=dtype, device=flag_gems.device)
    src = torch.randn(32, dtype=dtype, device=flag_gems.device)
    full_index = torch.randint(0, 20, (64,), dtype=torch.long, device=flag_gems.device)
    index = full_index[::2]  # shape (32,), non-contiguous
    index = index.clamp(0, 15)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "amax"])
def test_scatter_reduce_all_same_index(reduce):
    """All 64 elements scattered to the same output position."""
    dtype = torch.float32
    inp = torch.randn(4, dtype=dtype, device=flag_gems.device)
    src = torch.randn(64, dtype=dtype, device=flag_gems.device)
    index = torch.full((64,), 2, dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_two
@pytest.mark.parametrize("reduce", ["sum", "prod", "mean"])
def test_scatter_reduce_identity_src(reduce):
    """Source of all ones (identity for prod, additive for sum/mean)."""
    dtype = torch.float32
    inp = torch.randn(8, dtype=dtype, device=flag_gems.device)
    src = torch.ones(16, dtype=dtype, device=flag_gems.device)
    index = torch.randint(0, 8, (16,), dtype=torch.long, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, upcast=True)
    ref_index = utils.to_reference(index)
    ref_src = utils.to_reference(src, upcast=True)
    ref_out = torch.scatter_reduce(ref_inp, 0, ref_index, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(inp, 0, index, src, reduce=reduce)

    utils.gems_assert_close(res_out, ref_out, dtype)
