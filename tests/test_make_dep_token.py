import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.make_dep_token
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_make_dep_token(dtype):
    """Test _make_dep_token creates a scalar tensor with correct dtype."""
    ref = utils.to_reference(torch.ops.aten._make_dep_token(dtype=dtype))
    with flag_gems.use_gems():
        res_out = torch.ops.aten._make_dep_token(dtype=dtype)
    utils.gems_assert_close(res_out, ref, dtype, atol=1.0)
    assert res_out.shape == torch.Size(
        []
    ), f"Expected scalar tensor, got shape {res_out.shape}"
    assert res_out.dtype == dtype, f"Expected dtype {dtype}, got {res_out.dtype}"


@pytest.mark.make_dep_token
def test_make_dep_token_default():
    """Test _make_dep_token with default parameters."""
    ref = utils.to_reference(torch.ops.aten._make_dep_token())
    with flag_gems.use_gems():
        res_out = torch.ops.aten._make_dep_token()
    utils.gems_assert_close(res_out, ref, torch.float32, atol=1.0)
    assert (
        res_out.dtype == torch.float32
    ), f"Expected default dtype float32, got {res_out.dtype}"
    assert res_out.shape == torch.Size(
        []
    ), f"Expected scalar tensor, got shape {res_out.shape}"


@pytest.mark.make_dep_token
def test_make_dep_token_gems_impl():
    """Test _make_dep_token with FlagGems implementation directly."""
    res_out = flag_gems.ops._make_dep_token()

    assert res_out.shape == torch.Size(
        []
    ), f"Expected scalar tensor, got shape {res_out.shape}"
    assert (
        res_out.dtype == torch.float32
    ), f"Expected default dtype float32, got {res_out.dtype}"
    assert res_out.device.type == "cuda", f"Expected cuda device, got {res_out.device}"
