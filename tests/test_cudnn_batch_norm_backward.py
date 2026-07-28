import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.cudnn_batch_norm_backward
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
def test_cudnn_batch_norm_backward(shape, dtype, affine):
    C = shape[1]
    # Create input and weight tensors (no requires_grad for backward kernel test)
    res_inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    # Always create weight for cudnn_batch_norm_backward
    res_weight = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)
    # For bias, we always create one but only check gradients when affine=True
    res_bias = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-05
    train = True

    # Run forward pass using cudnn_batch_norm to get saved statistics
    # Note: cudnn_batch_norm only supports float32 for the weight parameter
    # so we need to convert to float32 for the forward pass
    res_inp_f32 = res_inp.to(torch.float32)
    res_weight_f32 = res_weight.to(torch.float32)
    # Always use bias for forward pass
    res_bias_f32 = res_bias.to(torch.float32)

    out_cudnn, save_mean, save_var, reserve = torch.ops.aten.cudnn_batch_norm(
        res_inp_f32, res_weight_f32, res_bias_f32, None, None, train, eps, False
    )

    # Convert back to the test dtype
    save_mean = save_mean.to(dtype)
    save_var = save_var.to(dtype)

    # Create grad_output (all ones for simplicity)
    grad_output = torch.ones_like(out_cudnn).to(dtype)
    reserve_space = torch.empty(0, dtype=dtype, device=flag_gems.device)

    # Reference implementation using native_batch_norm_backward
    ref_grad = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_save_mean = utils.to_reference(save_mean, True)
    ref_save_var = utils.to_reference(save_var, True)
    ref_save_invstd = torch.rsqrt(ref_save_var + eps)
    # For native_batch_norm_backward, we still need to output all gradients
    output_mask = [True, True, True]

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_batch_norm_backward(
        ref_grad,
        ref_inp,
        ref_weight,
        None,  # running_mean
        None,  # running_var
        ref_save_mean,
        ref_save_invstd,
        train,
        eps,
        output_mask,
    )

    # Run FlagGems implementation
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.cudnn_batch_norm_backward(
            res_inp,
            grad_output,
            res_weight,
            None,  # running_mean
            None,  # running_var
            save_mean,
            save_var,
            eps,
            reserve_space,
        )

    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    # Only check weight and bias gradients when affine=True
    if affine:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=reduce_dim
        )
