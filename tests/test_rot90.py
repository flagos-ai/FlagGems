import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.rot90
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, -1, -2, -3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_2d(k, dtype):
    """Test rot90 with 2D input."""
    shape = (4, 6)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.rot90(ref_x, k)

    with flag_gems.use_gems():
        act_out = flag_gems.rot90(x, k)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.rot90
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dims", [(0, 1), (1, 2), (0, 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_3d(k, dims, dtype):
    """Test rot90 with 3D input and custom dims."""
    shape = (4, 6, 8)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.rot90(ref_x, k, dims)

    with flag_gems.use_gems():
        act_out = flag_gems.rot90(x, k, dims)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.rot90
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dims", [(1, 2), (2, 3), (1, 3)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_4d(k, dims, dtype):
    """Test rot90 with 4D input and custom dims."""
    shape = (2, 4, 6, 8)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.rot90(ref_x, k, dims)

    with flag_gems.use_gems():
        act_out = flag_gems.rot90(x, k, dims)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.rot90
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_negative_dims(k, dtype):
    """Test rot90 with negative dimension indices."""
    shape = (4, 6, 8)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.rot90(ref_x, k, (-2, -1))

    with flag_gems.use_gems():
        act_out = flag_gems.rot90(x, k, (-2, -1))

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.rot90
@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_square(k, dtype):
    """Test rot90 with square input."""
    shape = (8, 8)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.rot90(ref_x, k)

    with flag_gems.use_gems():
        act_out = flag_gems.rot90(x, k)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.rot90
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_rot90_non_contiguous(k, dtype):
    """Test rot90 with non-contiguous input."""
    shape = (4, 6)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x_non_cont = x[:, ::2]  # Make non-contiguous

    ref_x = utils.to_reference(x_non_cont)
    ref_out = torch.ops.aten.rot90(ref_x, k)

    with flag_gems.use_gems():
        act_out = flag_gems.rot90(x_non_cont, k)

    utils.gems_assert_close(act_out, ref_out, dtype)
