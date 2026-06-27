import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_logit
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logit_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_out(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        torch.special.logit(inp, eps=1e-6, out=out)
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.special_logit_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_(shape, dtype):
    torch.manual_seed(0)
    base = torch.empty(shape, device=flag_gems.device, dtype=torch.float32).uniform_(
        -4.0, 4.0
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        flag_gems.special_logit_(inp, eps=1e-6)
    utils.gems_assert_close(inp, ref_out, dtype)


# --- Boundary tests for special_logit ---


@pytest.mark.special_logit
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_eps_none(dtype):
    """Test special_logit with eps=None (no clamping)."""
    inp = torch.tensor([0.001, 0.5, 0.999], device=flag_gems.device, dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=None)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=None)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logit
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_extreme_values(dtype):
    """Test special_logit with very large positive and negative logit values."""
    base = torch.tensor(
        [-10.0, -4.0, -1.0, 0.0, 1.0, 4.0, 10.0],
        device=flag_gems.device,
        dtype=torch.float32,
    )
    inp = torch.sigmoid(base).to(dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=1e-6)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_logit
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_nan_input(dtype):
    """Test special_logit with NaN input: expect NaN output."""
    inp = torch.tensor(
        [0.0, float("nan"), 0.5, float("nan"), 1.0],
        device=flag_gems.device,
        dtype=dtype,
    )
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=1e-6)
    # NaN in → NaN out; compare NaN positions
    ref_nan_mask = torch.isnan(ref_out)
    res_nan_mask = torch.isnan(res_out)
    assert torch.equal(
        ref_nan_mask.cpu(), res_nan_mask.cpu()
    ), f"NaN mask mismatch: ref_nan at {ref_nan_mask}, res_nan at {res_nan_mask}"


@pytest.mark.special_logit
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_inf_input(dtype):
    """Test special_logit with Inf input: inf is clamped to [eps, 1-eps] when eps is not None."""
    inp = torch.tensor(
        [0.5, float("inf"), float("-inf")], device=flag_gems.device, dtype=dtype
    )
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=1e-6)
    # NaN mask should match for both outputs
    ref_nan_mask = torch.isnan(ref_out)
    res_nan_mask = torch.isnan(res_out)
    assert torch.equal(
        ref_nan_mask.cpu(), res_nan_mask.cpu()
    ), f"NaN mask mismatch: ref_nan at {ref_nan_mask}, res_nan at {res_nan_mask}"


@pytest.mark.special_logit
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_logit_out_of_range(dtype):
    """Test special_logit with out-of-range values (x < 0 or x > 1)."""
    inp = torch.tensor([-0.5, 0.0, 0.5, 1.0, 1.5], device=flag_gems.device, dtype=dtype)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.logit(ref_inp, eps=1e-6)
    with flag_gems.use_gems():
        res_out = torch.special.logit(inp, eps=1e-6)
    # Use higher tolerance since float32 precision is inherently limited
    # at the singularities x≈0 and x≈1 where logit(x) diverges.
    utils.gems_assert_close(res_out, ref_out, dtype, atol=3e-2)
