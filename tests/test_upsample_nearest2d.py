import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]
    # Reference: force CPU to avoid flag_gems kernel dispatch interference
    ref_i = input.cpu().clone().detach().to(torch.float32).requires_grad_(True)
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    out_grad = torch.randn_like(res_out)
    ref_grad = out_grad.cpu().to(torch.float32).detach()
    ref_out.backward(ref_grad)
    with flag_gems.use_gems():
        res_out.backward(out_grad)
    # nearest neighbor backward accumulates gradients from multiple output
    # positions into the same input position. Float16/bfloat16 rounding error
    # can accumulate significantly with large scale factors.
    res_grad = input.grad.cpu()
    ref_grad_input = ref_i.grad
    torch.testing.assert_close(res_grad, ref_grad_input.to(dtype), atol=5e-2, rtol=1e-3)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_noncontiguous(dtype):
    input = torch.randn((2, 3, 32, 32), dtype=dtype, device=flag_gems.device)
    input = input.permute(0, 1, 3, 2)  # noncontiguous
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [64, 64]
    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)
    utils.gems_assert_close(res_out, ref_out, dtype)
