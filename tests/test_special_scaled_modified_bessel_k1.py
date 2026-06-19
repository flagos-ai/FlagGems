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
