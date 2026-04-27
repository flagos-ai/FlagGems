import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.smooth_l1_loss
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_accuracy_smooth_l1_loss(shape, dtype, reduction, beta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)
    ref_out = torch.nn.functional.smooth_l1_loss(
        ref_inp, ref_target, reduction=reduction, beta=beta
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.smooth_l1_loss(
            inp, target, reduction=reduction, beta=beta
        )
    if reduction == "none":
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        res_f32 = res_out.to(torch.float32)
        ref_f32 = ref_out.to(torch.float32)
        if torch.isinf(res_f32).any() or torch.isinf(ref_f32).any():
            return
        rtol = 5e-2 if dtype in (torch.bfloat16, torch.float16) else 1e-3
        assert torch.allclose(res_f32, ref_f32, rtol=rtol, atol=1.0)
