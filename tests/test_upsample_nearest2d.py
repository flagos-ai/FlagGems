import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

if cfg.QUICK_MODE:
    UPSAMPLE_NEAREST2D_SCALES = [(2, 2)]
else:
    UPSAMPLE_NEAREST2D_SCALES = [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)]


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("scale", UPSAMPLE_NEAREST2D_SCALES)
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d(dtype, shape, scale):
    if flag_gems.vendor_name == "sunrise" and shape[2] * shape[3] >= 1023 * 1025:
        pytest.skip("Issue #3836: Skip for big shape, '--ref cpu' too slow.")
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_i = utils.to_reference(input).to(torch.float32)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size).to(dtype)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward(dtype, shape, scale):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_i = utils.to_reference(input, True)
    output_size = [int(input.shape[i + 2] * scale[i]) for i in range(2)]

    ref_out = torch._C._nn.upsample_nearest2d(ref_i, output_size=output_size)
    with flag_gems.use_gems():
        res_out = torch._C._nn.upsample_nearest2d(input, output_size=output_size)

    out_grad = torch.randn_like(res_out)
    ref_grad = utils.to_reference(out_grad, True)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_i, ref_grad)

    with flag_gems.use_gems():
        (res_in_grad,) = torch.autograd.grad(res_out, input, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
