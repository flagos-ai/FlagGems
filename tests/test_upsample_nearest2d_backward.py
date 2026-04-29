import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "params",
    [
        {"input_shape": (1, 1, 4, 4), "output_size": (8, 8)},
        {"input_shape": (2, 3, 8, 8), "output_size": (16, 16)},
        {"input_shape": (1, 1, 4, 4), "output_size": (12, 12)},
        {"input_shape": (2, 4, 16, 16), "output_size": (32, 32)},
        {"input_shape": (1, 1, 1, 1), "output_size": (4, 4)},
        {"input_shape": (1, 1, 32, 32), "output_size": (64, 64)},
    ],
)
def test_accuracy_upsample_nearest2d_backward(params, dtype):
    input_shape = params["input_shape"]
    output_size = params["output_size"]
    N, C, IH, IW = input_shape
    OH, OW = output_size

    grad_output = torch.randn((N, C, OH, OW), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad_output, True)
    ref_out = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad, output_size, input_shape, None, None
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, output_size, input_shape, None, None
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.upsample_nearest2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_upsample_nearest2d_backward_with_scales(dtype):
    input_shape = (1, 1, 8, 8)
    N, C, IH, IW = input_shape
    scales_h, scales_w = 2.0, 2.0
    OH, OW = int(IH * scales_h), int(IW * scales_w)

    grad_output = torch.randn((N, C, OH, OW), dtype=dtype, device=flag_gems.device)
    ref_grad = utils.to_reference(grad_output, True)
    ref_out = torch.ops.aten.upsample_nearest2d_backward(
        ref_grad, (OH, OW), input_shape, scales_h, scales_w
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.upsample_nearest2d_backward(
            grad_output, (OH, OW), input_shape, scales_h, scales_w
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
