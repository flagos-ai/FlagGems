import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

@pytest.mark.cudnn_batch_norm_backward
@pytest.mark.skipif(
    flag_gems.vendor_name == "cambricon", reason="Issue #5254: Not supported"
)
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
    if (flag_gems.vendor_name == "nvidia" and not utils.TO_CPU and dtype == torch.bfloat16
            and str(torch.__version__) in {"2.8.0a0+5228986c39.nv25.05", "2.8.0a0+5228986c39.nv25.06"}):
        pytest.skip("Native cuDNN batch norm forward/backward does not support BFloat16")
    torch.manual_seed(0)
    C = shape[1]
    # Create input and weight tensors (no requires_grad for backward kernel test)
    res_inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    # Native Half needs FP32 affine/stat parameters. Preserve the original
    # BF16 contract for its CPU oracle; the unavailable CUDA BF16 API is skipped.
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    res_weight = torch.ones(size=(C,), dtype=param_dtype, device=flag_gems.device)
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

    if flag_gems.vendor_name == "enflame":
        # There is no cuDNN forward on Enflame. Freeze native CPU training
        # statistics, then keep the operator's real FP32 target-stat ABI.
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(
            res_inp_f32.cpu(), res_weight_f32.cpu(), res_bias_f32.cpu(),
            None, None, train, 0.1, eps
        )
        save_mean = save_mean.to(device=res_inp.device, dtype=param_dtype)
        save_invstd = save_invstd.to(device=res_inp.device, dtype=param_dtype)
        reserve_space = torch.empty(0, dtype=torch.uint8, device=res_inp.device)
    else:
        _, save_mean, save_invstd, reserve_space = torch.ops.aten.cudnn_batch_norm(
            res_inp_f32, res_weight_f32, res_bias_f32, None, None, train, 0.1, eps
        )
    save_mean = save_mean.to(dtype=param_dtype)
    save_invstd = save_invstd.to(dtype=param_dtype)
    grad_output = torch.ones_like(res_inp)

    # Reference implementation using native_batch_norm_backward
    def reference(value):
        if flag_gems.vendor_name == "enflame":
            return value.to("cpu", dtype=torch.float64)
        return utils.to_reference(value, True)
    ref_grad = reference(grad_output)
    ref_inp = reference(res_inp)
    ref_weight = reference(res_weight)
    ref_save_mean = reference(save_mean)
    ref_save_invstd = reference(save_invstd)
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

    expected_device = res_inp.device
    expected_shapes = (tuple(shape), (C,), (C,))
    expected_dtypes = (dtype, param_dtype, param_dtype)
    # Run FlagGems implementation
    (
        res_in_grad,
        res_weight_grad,
        res_bias_grad,
    ) = flag_gems.cudnn_batch_norm_backward(
        res_inp,
        grad_output,
        res_weight,
        None,  # running_mean
        None,  # running_var
        save_mean,
        save_invstd,
        eps,
        reserve_space,
    )

    outputs = (res_in_grad, res_weight_grad, res_bias_grad)
    for actual, expected_shape, expected_dtype in zip(outputs, expected_shapes, expected_dtypes):
        assert actual.device == expected_device
        assert tuple(actual.shape) == expected_shape
        assert actual.dtype == expected_dtype
    if ref_in_grad.device.type == "cpu":
        res_in_grad, res_weight_grad, res_bias_grad = (x.cpu() for x in outputs)
    reduce_dim = math.prod(shape) // C
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype, reduce_dim=reduce_dim)
    # Only check weight and bias gradients when affine=True
    if affine:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, param_dtype, reduce_dim=reduce_dim
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, param_dtype, reduce_dim=reduce_dim
        )
