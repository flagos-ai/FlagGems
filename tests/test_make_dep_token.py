import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.make_dep_token
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_make_dep_token(dtype):
    """Test _make_dep_token creates a scalar tensor with correct dtype.

    Note: _make_dep_token produces uninitialized values, so we compare
    deterministic structural properties (numel) with gems_assert_close
    instead of comparing tensor values directly.
    """
    ref = utils.to_reference(torch.ops.aten._make_dep_token(dtype=dtype))
    with flag_gems.use_gems():
        res_out = torch.ops.aten._make_dep_token(dtype=dtype)
    ref_numel = torch.tensor(ref.numel(), dtype=torch.int64, device=res_out.device)
    res_numel = torch.tensor(res_out.numel(), dtype=torch.int64, device=res_out.device)
    utils.gems_assert_close(res_numel, ref_numel, torch.int64)
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
    ref_numel = torch.tensor(ref.numel(), dtype=torch.int64, device=res_out.device)
    res_numel = torch.tensor(res_out.numel(), dtype=torch.int64, device=res_out.device)
    utils.gems_assert_close(res_numel, ref_numel, torch.int64)
    assert (
        res_out.dtype == torch.float32
    ), f"Expected default dtype float32, got {res_out.dtype}"
    assert res_out.shape == torch.Size(
        []
    ), f"Expected scalar tensor, got shape {res_out.shape}"


@pytest.mark.make_dep_token
def test_make_dep_token_device():
    """Test _make_dep_token returns a CUDA tensor through FlagGems dispatch."""
    with flag_gems.use_gems():
        res_out = torch.ops.aten._make_dep_token()

    assert res_out.shape == torch.Size(
        []
    ), f"Expected scalar tensor, got shape {res_out.shape}"
    assert (
        res_out.dtype == torch.float32
    ), f"Expected default dtype float32, got {res_out.dtype}"
