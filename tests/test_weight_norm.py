import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [-1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, -1, 1]


@pytest.mark.weight_norm
@pytest.mark.skip(reason="Issue #2860: fails assertion")
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_weight_norm(shape, dtype, dim):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(42)
        torch.mlu.manual_seed_all(42)

    dim = dim % len(shape)
    v = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    g = torch.randn(
        [1 if i != dim else shape[i] for i in range(v.ndim)],
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=True,
    )
    reduce_size = v.numel() // shape[dim]

    ref_v = utils.to_reference(v, True)
    ref_g = utils.to_reference(g, True)
    ref_w_out = torch._weight_norm(ref_v, ref_g, dim)
    res_w_out = flag_gems.weight_norm(v, g, dim)
    utils.gems_assert_close(res_w_out, ref_w_out, dtype, reduce_dim=reduce_size)

    res_w_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_w_grad = utils.to_reference(res_w_grad, True)

    ref_v_grad, ref_g_grad = torch.autograd.grad(
        ref_w_out, (ref_v, ref_g), grad_outputs=ref_w_grad
    )
    res_v_grad, res_g_grad = torch.autograd.grad(
        res_w_out, (v, g), grad_outputs=res_w_grad
    )
    utils.gems_assert_close(
        res_v_grad, ref_v_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
    utils.gems_assert_close(
        res_g_grad, ref_g_grad, dtype, reduce_dim=reduce_size, equal_nan=True
    )
