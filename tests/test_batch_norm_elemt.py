import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.batch_norm_elemt
@pytest.mark.parametrize(
    "shape",
    [
        (2, 64, 112, 112),
        (8, 128, 56, 56),
        (16, 256, 28, 28),
        (32, 512, 14, 14),
        (64, 1024, 7, 7),
        (128, 256, 32, 32),
        (4, 32, 8, 8, 8),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_batch_norm_elemt(shape, dtype):
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    invstd = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device).abs() + 0.01

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_mean = utils.to_reference(mean, True)
    ref_invstd = utils.to_reference(invstd, True)

    # Reference: manual computation in high precision
    view_shape = [1] * len(shape)
    view_shape[1] = C
    ref_out = (ref_inp - ref_mean.view(view_shape)) * ref_invstd.view(
        view_shape
    ) * ref_weight.view(view_shape) + ref_bias.view(view_shape)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.batch_norm_elemt(inp, weight, bias, mean, invstd, 0.0)

    utils.gems_assert_close(res_out, ref_out, dtype)
