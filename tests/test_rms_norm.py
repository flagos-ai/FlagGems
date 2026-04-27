import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    np.random.seed(0)
    np_inp = np.random.uniform(-0.1, 0.1, shape[:2]).astype(np.float32)
    np_grad = np.random.uniform(-0.01, 0.01, shape[:2]).astype(np.float32)
    np_weight = np.random.uniform(-0.1, 0.1, layer_shape).astype(np.float32)

    inp = torch.tensor(np_inp, dtype=dtype, device=flag_gems.device, requires_grad=True)
    weight = torch.tensor(
        np_weight, dtype=dtype, device=flag_gems.device, requires_grad=True
    )

    eps = 1e-5

    ref_inp = utils.to_reference(inp)
    ref_weight = utils.to_reference(weight)

    def _torch_rms_norm(x, weight, eps):
        upcast_x = x.to(torch.float32)
        variance = upcast_x.pow(2).mean(-1, keepdim=True)
        hidden_states = upcast_x * torch.rsqrt(variance + eps).to(torch.float32)
        hidden_states = hidden_states.to(x.dtype)
        return weight * hidden_states

    ref_out = _torch_rms_norm(ref_inp, weight=ref_weight, eps=eps)
    res_out = flag_gems.rms_norm(inp, list(layer_shape), weight=weight, eps=eps)

    res_grad = torch.tensor(
        np_grad, dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    ref_grad = utils.to_reference(res_grad)

    res_grad, res_weight_grad = torch.autograd.grad(res_out, (inp, weight), res_grad)
    ref_grad, ref_weight_grad = torch.autograd.grad(
        ref_out, (ref_inp, ref_weight), ref_grad
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    utils.gems_assert_close(res_weight_grad, ref_weight_grad, dtype, reduce_dim=N)
