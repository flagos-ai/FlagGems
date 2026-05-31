import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.channel_shuffle
@pytest.mark.parametrize("groups", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_channel_shuffle_4d(groups, dtype):
    """Test channel_shuffle with 4D input (N, C, H, W)."""
    C = groups * 4
    shape = (2, C, 16, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.channel_shuffle(ref_x, groups)

    with flag_gems.use_gems():
        act_out = flag_gems.channel_shuffle(x, groups)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.channel_shuffle
@pytest.mark.parametrize("groups", [2, 3, 6])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_channel_shuffle_3d(groups, dtype):
    """Test channel_shuffle with 3D input (C, H, W)."""
    C = groups * 4
    shape = (C, 16, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.channel_shuffle(ref_x, groups)

    with flag_gems.use_gems():
        act_out = flag_gems.channel_shuffle(x, groups)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.channel_shuffle
@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_channel_shuffle_5d(groups, dtype):
    """Test channel_shuffle with 5D input (N, C, D, H, W)."""
    C = groups * 4
    shape = (2, C, 8, 16, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.channel_shuffle(ref_x, groups)

    with flag_gems.use_gems():
        act_out = flag_gems.channel_shuffle(x, groups)

    utils.gems_assert_close(act_out, ref_out, dtype)


@pytest.mark.channel_shuffle
@pytest.mark.parametrize("groups", [2, 4])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_channel_shuffle_non_contiguous(groups, dtype):
    """Test channel_shuffle with non-contiguous input."""
    C = groups * 4
    shape = (2, C, 16, 16)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Make non-contiguous by slicing
    x_non_cont = x[:, :, ::2, ::2]

    ref_x = utils.to_reference(x_non_cont)
    ref_out = torch.ops.aten.channel_shuffle(ref_x, groups)

    with flag_gems.use_gems():
        act_out = flag_gems.channel_shuffle(x_non_cont, groups)

    utils.gems_assert_close(act_out, ref_out, dtype)
