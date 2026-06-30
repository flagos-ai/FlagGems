import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_gammainc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# float32 only: gammainc series expansion is numerically unstable in lower precisions
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammainc(shape, dtype):
    # Use positive values for gammainc as it's defined for non-negative inputs
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.special.gammainc(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.special.gammainc(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


# Boundary tests for mathematical correctness
@pytest.mark.special_gammainc
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammainc_boundary_x_zero(dtype):
    """P(a, 0) = 0 for all a > 0."""
    a_vals = torch.tensor(
        [0.5, 1.0, 2.0, 5.0, 10.0], dtype=dtype, device=flag_gems.device
    )
    x_vals = torch.zeros_like(a_vals)

    ref_a = utils.to_reference(a_vals, True)
    ref_x = utils.to_reference(x_vals, True)
    ref_out = torch.special.gammainc(ref_a, ref_x)

    with flag_gems.use_gems():
        res = torch.special.gammainc(a_vals, x_vals)

    utils.gems_assert_close(res, ref_out, dtype)


@pytest.mark.special_gammainc
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammainc_boundary_a_one(dtype):
    """P(1, x) = 1 - exp(-x)."""
    x = torch.linspace(0.1, 20.0, 100, dtype=dtype, device=flag_gems.device)
    a = torch.ones_like(x)

    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.special.gammainc(ref_a, ref_x)

    with flag_gems.use_gems():
        res = torch.special.gammainc(a, x)

    utils.gems_assert_close(res, ref_out, dtype, atol=1e-5)


@pytest.mark.special_gammainc
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_gammainc_boundary_large_x(dtype):
    """P(a, x) -> 1 as x >> a."""
    a = torch.tensor([0.5, 1.0, 2.0], dtype=dtype, device=flag_gems.device)
    x = torch.full_like(a, 100.0)

    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.special.gammainc(ref_a, ref_x)

    with flag_gems.use_gems():
        res = torch.special.gammainc(a, x)

    utils.gems_assert_close(res, ref_out, dtype)
