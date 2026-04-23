import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]

SHAPES = [
    (16, 3),
    (32, 32, 32),
    (8, 32, 224, 224),
    (2050, 16, 32, 32),
    (8, 16, 3, 224, 224),
]


@pytest.mark.batch_norm
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm(shape, dtype, affine):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(23)
        torch.mlu.manual_seed_all(23)

    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = None
    bias = None
    if affine:
        weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
        bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)

    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    ref_out = torch.nn.functional.batch_norm(
        ref_inp,
        ref_running_mean,
        ref_running_var,
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.batch_norm(
            inp,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            eps=eps,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(running_mean, ref_running_mean, dtype)
    utils.gems_assert_close(running_var, ref_running_var, dtype)


@pytest.mark.batch_norm_backward
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_backward(shape, dtype, affine):
    C = shape[1]
    res_grad = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    res_inp = torch.randn_like(res_grad)
    res_weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    res_running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    res_running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)
    res_save_mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
    res_save_invstd = torch.randn(C, dtype=torch.float32, device=flag_gems.device)

    ref_grad = utils.to_reference(res_grad, True)
    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_running_mean = utils.to_reference(res_running_mean, True)
    ref_running_var = utils.to_reference(res_running_var, True)
    ref_save_mean = utils.to_reference(res_save_mean, True)
    ref_save_invstd = utils.to_reference(res_save_invstd, True)

    train = True
    eps = 1e-05
    if affine:
        output_mask = [True, True, True]
    else:
        output_mask = [True, False, False]

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_batch_norm_backward(
        ref_grad,
        ref_inp,
        ref_weight,
        ref_running_mean,
        ref_running_var,
        ref_save_mean,
        ref_save_invstd,
        train,
        eps,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_batch_norm_backward(
            res_grad,
            res_inp,
            res_weight,
            res_running_mean,
            res_running_var,
            res_save_mean,
            res_save_invstd,
            train,
            eps,
            output_mask,
        )

    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    if affine:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim
        )
