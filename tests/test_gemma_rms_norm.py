import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


def _torch_gemma_rms_norm(x, weight, eps):
    upcast_x = x.to(torch.float32)
    variance = upcast_x.pow(2).mean(-1, keepdim=True)
    hidden_states = upcast_x * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * hidden_states).to(x.dtype)


def _torch_gemma_fused_add_rms_norm(x, residual, weight, eps):
    x = x + residual
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return ((1.0 + weight.to(torch.float32)) * hidden_states).to(x.dtype), x


@pytest.mark.gemma_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gemma_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [N]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device) * 0.01
    eps = 1e-6

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out = _torch_gemma_rms_norm(ref_inp, ref_weight, eps)
    res_out = flag_gems.gemma_rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.gemma_fused_add_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_gemma_fused_add_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [N]
    inp = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    residual = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device) * 0.01
    eps = 1e-6

    ref_inp = utils.to_reference(inp, True)
    ref_residual = utils.to_reference(residual, True)
    ref_weight = utils.to_reference(weight, True)

    ref_out, ref_new_residual = _torch_gemma_fused_add_rms_norm(
        ref_inp, ref_residual, ref_weight, eps
    )
    res_out, res_new_residual = flag_gems.gemma_fused_add_rms_norm(
        inp, residual, list(layer_shape), weight=weight, eps=eps
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_new_residual, ref_new_residual, dtype)
