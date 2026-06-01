import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.add_rms_norm
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_add_rms_norm(shape, dtype):
    N = shape[1]
    layer_shape = [
        N,
    ]
    inp1 = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape[:2], dtype=dtype, device=flag_gems.device)
    weight = torch.randn(layer_shape, dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)
    ref_weight = utils.to_reference(weight, True)

    def _torch_add_rms_norm(x1, x2, weight, eps):
        x = x1 + x2
        variance = x.pow(2).mean(-1, keepdim=True)
        hidden_states = x * torch.rsqrt(variance + eps)
        return weight * hidden_states

    ref_out = _torch_add_rms_norm(ref_inp1, ref_inp2, weight=ref_weight, eps=eps)

    with flag_gems.use_gems():
        res_out = flag_gems.add_rms_norm(
            inp1, inp2, list(layer_shape), weight=weight, eps=eps
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
