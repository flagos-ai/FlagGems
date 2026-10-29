import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _cpu_backward_reference(input, grad_output, weight, save_mean, save_invstd, eps):
    # cuDNN has no CPU dispatch. Native BN uses the same saved inverse standard
    # deviation directly, including the parameter publicly named save_var.
    to_cpu = lambda value: value.detach().to(device="cpu", dtype=torch.float64)
    return torch.ops.aten.native_batch_norm_backward.default(
        to_cpu(grad_output),
        to_cpu(input),
        to_cpu(weight),
        None,
        None,
        to_cpu(save_mean),
        to_cpu(save_invstd),
        True,
        eps,
        [True, True, True],
    )


def _saved_statistics(input, weight, bias, eps):
    if flag_gems.vendor_name == "nvidia" and not utils.TO_CPU:
        _, mean, invstd, reserve = torch.ops.aten.cudnn_batch_norm.default(
            input,
            weight,
            bias,
            None,
            None,
            training=True,
            exponential_average_factor=0.1,
            epsilon=eps,
        )
        return mean, invstd, reserve
    # The target may have neither cuDNN forward nor backward. Construct genuine
    # training statistics on CPU, then retain their required FP32 target ABI.
    _, mean, invstd = torch.ops.aten.native_batch_norm.default(
        input.detach().to(device="cpu", dtype=torch.float32),
        weight.cpu(),
        bias.cpu(),
        None,
        None,
        True,
        0.1,
        eps,
    )
    return (
        mean.to(input.device),
        invstd.to(input.device),
        torch.empty(0, dtype=torch.uint8, device=input.device),
    )


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
    if (
        flag_gems.vendor_name == "nvidia"
        and dtype == torch.bfloat16
        and not utils.TO_CPU
    ):
        pytest.skip(
            "native cuDNN batch-normalization backward does not support bfloat16"
        )
    torch.manual_seed(0)
    channels = shape[1]
    res_inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    # cuDNN half inputs require FP32 scale and saved statistics. Gradients for
    # scale and bias follow that FP32 ABI; input gradients retain input dtype.
    res_weight = torch.ones(channels, dtype=torch.float32, device=flag_gems.device)
    res_bias = torch.ones_like(res_weight)
    eps = 1e-5
    save_mean, save_invstd, reserve = _saved_statistics(
        res_inp, res_weight, res_bias, eps
    )
    grad_output = torch.ones_like(res_inp)
    args = (
        res_inp,
        grad_output,
        res_weight,
        None,
        None,
        save_mean,
        save_invstd,
        eps,
        reserve,
    )
    if flag_gems.vendor_name == "nvidia" and not utils.TO_CPU:
        reference = torch.ops.aten.cudnn_batch_norm_backward.default(*args)
    else:
        reference = _cpu_backward_reference(
            res_inp, grad_output, res_weight, save_mean, save_invstd, eps
        )
    inputs = [res_inp, grad_output, res_weight, save_mean, save_invstd, reserve]
    before = [value.clone() for value in inputs]
    result = flag_gems.cudnn_batch_norm_backward(*args)
    assert isinstance(result, (tuple, list)) and len(result) == 3
    expected_shapes = [res_inp.shape, res_weight.shape, res_weight.shape]
    expected_dtypes = [dtype, res_weight.dtype, res_weight.dtype]
    for actual, expected_shape, expected_dtype in zip(
        result, expected_shapes, expected_dtypes
    ):
        assert isinstance(actual, torch.Tensor)
        assert actual.device == res_inp.device
        assert actual.shape == expected_shape and actual.dtype == expected_dtype
        assert all(
            actual.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
            for value in inputs
        )
    for actual, original in zip(inputs, before):
        assert torch.equal(actual, original)
    reduce_dim = math.prod(shape) // channels
    # Check target ABI before transferring both values for numerical comparison.
    utils.gems_assert_close(
        result[0].cpu(), reference[0].cpu(), dtype, reduce_dim=reduce_dim
    )
    if affine:
        for actual, expected in zip(result[1:], reference[1:]):
            utils.gems_assert_close(
                actual.cpu(), expected.cpu(), res_weight.dtype, reduce_dim=reduce_dim
            )
