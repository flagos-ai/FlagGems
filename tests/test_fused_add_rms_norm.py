import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.fused_add_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fused_add_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_residual = utils.to_reference(residual, True)
    ref_weight = utils.to_reference(weight, True)

    def _torch_fused_add_rms_norm(x, residual, weight, eps):
        x = x + residual
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states, x

    ref_out, ref_new_residual = _torch_fused_add_rms_norm(
        ref_inp,
        ref_residual,
        weight=ref_weight,
        eps=eps,
    )

    res_out, res_new_residual = flag_gems.fused_add_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_new_residual, ref_new_residual, dtype)
