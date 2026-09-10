import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_scaled_modified_bessel_k1
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# Bessel K1 not supported for Half/BFloat16 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_scaled_modified_bessel_k1(shape, dtype):
    # Generate positive inputs since bessel_k1 is not defined for negative
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.scaled_modified_bessel_k1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.scaled_modified_bessel_k1(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_scaled_modified_bessel_k1_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# Bessel K1 not supported for Half/BFloat16 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_scaled_modified_bessel_k1_out(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, True)
    out_ref = torch.empty_like(ref_inp)
    out = torch.empty_like(inp)
    torch.special.scaled_modified_bessel_k1(ref_inp, out=out_ref)
    with flag_gems.use_gems():
        torch.special.scaled_modified_bessel_k1(inp, out=out)
    utils.gems_assert_close(out, out_ref, dtype)


@pytest.mark.special_scaled_modified_bessel_k1
@pytest.mark.parametrize("dtype", [torch.float32])
def test_special_scaled_modified_bessel_k1_edge_cases(dtype):
    """Test edge cases: zero, tiny, boundary, large, and negative values."""
    # Values covering small region, boundary, and large region
    edge_vals = [
        0.0,  # singularity -> inf
        1e-10,  # near zero
        1e-5,  # very small
        1e-3,  # small
        0.1,  # small region
        1.0,  # middle of small region
        2.0,  # boundary (small formula)
        2.1,  # just over boundary (asymptotic)
        3.0,  # moderate large
        5.0,  # large
        10.0,  # larger
        100.0,  # very large
        -0.1,  # negative -> nan
        -1.0,  # negative -> nan
    ]
    inp = torch.tensor(edge_vals, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.scaled_modified_bessel_k1(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.scaled_modified_bessel_k1(inp)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
