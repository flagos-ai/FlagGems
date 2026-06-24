import pytest
import torch

import flag_gems
from flag_gems.ops.special_log1p import special_log1p

from . import accuracy_utils as utils


@pytest.mark.special_log1p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log1p(shape, dtype):
    utils.init_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.special.log1p(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_log1p
@pytest.mark.parametrize("inp", [1.0, 5, -0.5])
def test_special_log1p_non_tensor(inp):
    ref_out = torch.special.log1p(torch.tensor(inp, dtype=torch.float32))
    res_out = special_log1p(inp)
    utils.gems_assert_close(ref_out, res_out, torch.float32)


@pytest.mark.special_log1p_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log1p_out(shape, dtype):
    utils.init_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out_buf = torch.empty(shape, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten.special_log1p.out(ref_inp, out=ref_out_buf)
    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_log1p.out(inp, out=res_out_buf)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_log1p
def test_special_log1p_negative():
    """Test special_log1p with negative inputs, including x <= -1."""
    inp = torch.tensor(
        [-0.5, -0.99, -1.0, -2.0], dtype=torch.float32, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.special.log1p(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.special_log1p
def test_special_log1p_nan_inf():
    """Test special_log1p with NaN and Inf inputs."""
    inp = torch.tensor(
        [float("nan"), float("inf"), float("-inf")],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.special.log1p(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.special_log1p
def test_special_log1p_small_values():
    """Test special_log1p precision for very small values."""
    inp = torch.tensor(
        [1e-15, 1e-10, 1e-8, 1e-5, -1e-5, -1e-8],
        dtype=torch.float64,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.special.log1p(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, torch.float64)
