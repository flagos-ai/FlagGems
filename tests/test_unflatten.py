import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unflatten
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_3d(dim, dtype):
    """Test unflatten with 3D input."""
    shape = (2, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    sizes = (2, 4) if dim == 1 else (2, 8) if dim == 2 else (4, 16)
    ref_out = torch.ops.aten.unflatten(ref_x, dim, sizes)

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, dim, sizes)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.unflatten
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_4d(dim, dtype):
    """Test unflatten with 4D input."""
    shape = (2, 4, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    sizes = (2, 4) if dim == 1 else (4, 8) if dim == 2 else (2, 8)
    ref_out = torch.ops.aten.unflatten(ref_x, dim, sizes)

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, dim, sizes)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.unflatten
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_inferred_dim(dtype):
    """Test unflatten with inferred dimension (-1)."""
    shape = (2, 12, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.unflatten(ref_x, 1, (-1, 4))

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, 1, (-1, 4))

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.unflatten
@pytest.mark.parametrize("dim", [0, 1, -1, -2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_negative_dim(dim, dtype):
    """Test unflatten with negative dimension indices."""
    shape = (2, 6, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    sizes = (2, 3)
    ref_out = torch.ops.aten.unflatten(ref_x, dim, sizes)

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, dim, sizes)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.unflatten
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_multiple_dims(dtype):
    """Test unflatten splitting into multiple dimensions."""
    shape = (2, 24, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.unflatten(ref_x, 1, (2, 3, 4))

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, 1, (2, 3, 4))

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.unflatten
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_size_1(dtype):
    """Test unflatten with size 1."""
    shape = (2, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.unflatten(ref_x, 1, (1, 8))

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, 1, (1, 8))

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.unflatten
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unflatten_all_ones(dtype):
    """Test unflatten with all ones."""
    shape = (2, 8, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.unflatten(ref_x, 1, (1, 1, 1, 8))

    with flag_gems.use_gems():
        act_out = flag_gems.unflatten(x, 1, (1, 1, 1, 8))

    utils.gems_assert_close(act_out, ref_out, dtype)
