import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# (time, batch, in_channels, out_channels, kernel_width, pad)
CONV_TBC_SHAPES = [
    (7, 3, 4, 5, 3, 1),
    (10, 2, 6, 3, 4, 0),
    (5, 1, 3, 2, 5, 3),
    (9, 4, 2, 7, 2, 2),
    (32, 8, 16, 32, 3, 1),
    (64, 4, 32, 16, 5, 2),
    (128, 2, 8, 8, 1, 0),
]


@pytest.mark.conv_tbc_backward
@pytest.mark.parametrize("shape", CONV_TBC_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_conv_tbc_backward(shape, dtype):
    ilen, batch, in_c, out_c, kw, pad = shape
    olen = ilen - kw + 1 + 2 * pad

    inp = torch.randn((ilen, batch, in_c), dtype=dtype, device=flag_gems.device)
    weight = torch.randn((kw, in_c, out_c), dtype=dtype, device=flag_gems.device)
    bias = torch.randn((out_c,), dtype=dtype, device=flag_gems.device)
    grad_output = torch.randn(
        (olen, batch, out_c), dtype=dtype, device=flag_gems.device
    )

    ref_grad_output = utils.to_reference(grad_output, True)
    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)

    ref_gi, ref_gw, ref_gb = torch.ops.aten.conv_tbc_backward(
        ref_grad_output, ref_inp, ref_weight, ref_bias, pad
    )
    with flag_gems.use_gems():
        res_gi, res_gw, res_gb = torch.ops.aten.conv_tbc_backward(
            grad_output, inp, weight, bias, pad
        )

    # grad_input / grad_weight accumulate over reduction dims; scale tolerance.
    utils.gems_assert_close(res_gi, ref_gi, dtype, reduce_dim=kw * out_c)
    utils.gems_assert_close(res_gw, ref_gw, dtype, reduce_dim=olen * batch)
    utils.gems_assert_close(res_gb, ref_gb, dtype, reduce_dim=olen * batch)
