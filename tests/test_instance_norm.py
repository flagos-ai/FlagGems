import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    NORM_SHAPES = [
        (2, 1, 2, 1),
    ]
    WEIGTH_BIAS = [True]
    USE_INPUT_BIAS = [True]
    HAS_RUN_STATS = [False]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    NORM_SHAPES = [
        (1, 1, 2, 2),
        (2, 1, 2, 2),
        (2, 3, 2, 2),
        (2, 3, 128, 128),
        (4, 16, 8, 8),
        (2, 3, 1024),
        (2, 3, 2048),
        (2, 3, 4096),
        (2, 3, 8192),
        (2, 3, 10240),
    ]
    WEIGTH_BIAS = [False, True]
    USE_INPUT_BIAS = [False, True]
    HAS_RUN_STATS = [False, True]

device = flag_gems.device


@pytest.mark.instance_norm
@pytest.mark.parametrize("shape", NORM_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("has_weight_bias", WEIGTH_BIAS)
@pytest.mark.parametrize("use_input_stats", USE_INPUT_BIAS)
@pytest.mark.parametrize("has_running_stats", HAS_RUN_STATS)
def test_instance_norm(
    shape, dtype, has_weight_bias, use_input_stats, has_running_stats
):
    if use_input_stats is False and has_running_stats is False:
        return

    B, C = shape[:2]
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)

    weight = None
    bias = None
    if has_weight_bias:
        weight = torch.randn(size=(C,), dtype=dtype, device=device, requires_grad=True)
        bias = torch.randn(size=(C,), dtype=dtype, device=device, requires_grad=True)

    running_mean = None
    running_var = None
    if has_running_stats:
        running_mean = torch.randn(size=(C,), dtype=torch.float32, device=device)
        r = torch.randn(size=(C,), dtype=torch.float32, device=device).abs()
        running_var = r + 1e-5

    momentum = 0.1
    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_running_mean = utils.to_reference(None, True)
    ref_running_var = utils.to_reference(None, True)
    if has_running_stats:
        ref_running_mean = utils.to_reference(running_mean.clone(), True)
        ref_running_var = utils.to_reference(running_var.clone(), True)

    ref_out = torch.nn.functional.instance_norm(
        ref_inp,
        running_mean=ref_running_mean,
        running_var=ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )

    res_out = flag_gems.instance_norm(
        inp,
        weight=weight,
        bias=bias,
        running_mean=running_mean,
        running_var=running_var,
        use_input_stats=use_input_stats,
        momentum=momentum,
        eps=eps,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    if has_running_stats:
        utils.gems_assert_close(running_mean, ref_running_mean, running_mean.dtype)
        utils.gems_assert_close(running_var, ref_running_var, running_var.dtype)

    out_grad = torch.randn_like(inp)
    ref_grad = utils.to_reference(out_grad, True)

    if has_weight_bias:
        (ref_in_grad, ref_weight_grad, ref_bias_grad) = torch.autograd.grad(
            ref_out, (ref_inp, ref_weight, ref_bias), ref_grad
        )
        (res_in_grad, res_weight_grad, res_bias_grad) = torch.autograd.grad(
            res_out, (inp, weight, bias), out_grad
        )
    else:
        (ref_in_grad,) = torch.autograd.grad(ref_out, (ref_inp,), ref_grad)
        (res_in_grad,) = torch.autograd.grad(res_out, (inp,), out_grad)

    M = B * C
    N = inp.numel() // M

    if use_input_stats:
        utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=N)

        if has_weight_bias:
            utils.gems_assert_close(
                res_weight_grad, ref_weight_grad, dtype, reduce_dim=B * N
            )
            utils.gems_assert_close(
                res_bias_grad, ref_bias_grad, dtype, reduce_dim=B * N
            )
