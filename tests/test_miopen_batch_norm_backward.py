import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.miopen_batch_norm_backward
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_miopen_batch_norm_backward(shape, dtype, affine):
    C = shape[1]
    res_grad = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    res_inp = torch.randn_like(res_grad)
    # miopen_batch_norm_backward requires weight (not optional in schema)
    # Use weight=1 when affine=False to simulate no affine transform
    res_weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
        if affine
        else torch.ones(C, dtype=dtype, device=flag_gems.device)
    )
    res_running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    res_running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)
    res_save_mean = torch.randn(C, dtype=torch.float32, device=flag_gems.device)
    # MIOpen names this argument save_var, but it stores saved inverse std.
    res_save_var = (
        torch.abs(torch.randn(C, dtype=torch.float32, device=flag_gems.device)) + 0.1
    )

    ref_grad = utils.to_reference(res_grad, True)
    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_running_mean = utils.to_reference(res_running_mean, True)
    ref_running_var = utils.to_reference(res_running_var, True)
    ref_save_mean = utils.to_reference(res_save_mean, True)
    ref_save_var = utils.to_reference(res_save_var, True)

    eps = 1e-05

    # PyTorch's MIOpen kernel is unavailable on NVIDIA, so compare with the
    # native backward using the saved inverse std directly.
    ref_save_invstd = ref_save_var
    train = True
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
        ) = torch.ops.aten.miopen_batch_norm_backward(
            res_inp,
            res_grad,
            res_weight,
            res_running_mean,
            res_running_var,
            res_save_mean,
            res_save_var,
            eps,
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
