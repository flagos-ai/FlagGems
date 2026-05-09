import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

random.seed(time.time() // 100)


@pytest.mark.upsample_nearest2d_backward
@pytest.mark.parametrize("scale", [(2, 2), (2.1, 3.7), (1.3, 5.1), (0.3, 0.5)])
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest2d_backward(dtype, shape, scale):
    output_size = [int(shape[i + 2] * scale[i]) for i in range(2)]

    grad_output = torch.randn(
        (shape[0], shape[1], output_size[0], output_size[1]),
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_grad = utils.to_reference(grad_output).to(torch.float32)
    ref_input_size = list(shape)

    ref_out = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad, output_size, ref_input_size, None, None
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, output_size, list(shape), None, None
        )

    assert res_out.shape == shape
    utils.gems_assert_close(res_out, ref_out, dtype)
