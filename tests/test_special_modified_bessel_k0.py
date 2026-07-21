import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_modified_bessel_k0
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# Half/BFloat16 not supported by PyTorch reference for modified_bessel_k0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_modified_bessel_k0(shape, dtype):
    # K0 is only defined for x > 0, so generate positive inputs only
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.special.modified_bessel_k0(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.modified_bessel_k0(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_modified_bessel_k0_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
# Half/BFloat16 not supported by PyTorch reference for modified_bessel_k0
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_special_modified_bessel_k0_out(shape, dtype):
    # K0 is only defined for x > 0, so generate positive inputs only
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 0.1
    ref_x = utils.to_reference(x)
    out_ref = torch.empty_like(ref_x)
    ref_out = torch.ops.aten.special_modified_bessel_k0.out(ref_x, out=out_ref)
    out_act = torch.empty_like(x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_modified_bessel_k0.out(x, out=out_act)
    utils.gems_assert_close(act_out, ref_out, dtype)
    utils.gems_assert_close(out_act, out_ref, dtype)
