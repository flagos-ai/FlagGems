import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.batch_norm_no_update
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
def test_batch_norm_no_update(shape, dtype, affine):
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = (
        torch.abs(torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)) + 0.1
    )

    eps = 1e-5

    # Compute reference: output = weight * (input - running_mean) / sqrt(running_var + eps) + bias
    w = (
        weight.to(torch.float32)
        if weight is not None
        else torch.ones(C, dtype=torch.float32, device=flag_gems.device)
    )
    b = (
        bias.to(torch.float32)
        if bias is not None
        else torch.zeros(C, dtype=torch.float32, device=flag_gems.device)
    )
    inp_f32 = inp.to(torch.float32)
    running_mean_f32 = running_mean.to(torch.float32)
    running_var_f32 = running_var.to(torch.float32)

    ref_out = w.view(1, C, *([1] * (inp.ndim - 2))) * (
        inp_f32 - running_mean_f32.view(1, C, *([1] * (inp.ndim - 2)))
    ) / torch.sqrt(running_var_f32.view(1, C, *([1] * (inp.ndim - 2))) + eps) + b.view(
        1, C, *([1] * (inp.ndim - 2))
    )
    ref_out = ref_out.to(dtype)

    with flag_gems.use_gems():
        (
            res_out,
            res_save_mean,
            res_save_var,
            res_reserved,
        ) = torch.ops.aten._batch_norm_no_update(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            0.1,
            eps,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
