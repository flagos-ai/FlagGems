import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.native_batch_norm_legit_functional
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
def test_native_batch_norm_legit_functional(shape, dtype, affine):
    if flag_gems.vendor_name == "cambricon":
        torch.manual_seed(23)
        torch.mlu.manual_seed_all(23)
    C = shape[1]

    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    (
        ref_out,
        ref_save_mean,
        ref_save_var,
        ref_running_mean_out,
        ref_running_var_out,
    ) = torch.ops.aten._native_batch_norm_legit_functional.default(
        ref_inp,
        ref_weight,
        ref_bias,
        ref_running_mean,
        ref_running_var,
        True,
        0.1,
        eps,
    )

    with flag_gems.use_gems():
        (
            res_out,
            res_save_mean,
            res_save_var,
            res_running_mean_out,
            res_running_var_out,
        ) = torch.ops.aten._native_batch_norm_legit_functional.default(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            True,
            0.1,
            eps,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(res_save_mean, ref_save_mean, dtype)
    utils.gems_assert_close(res_save_var, ref_save_var, dtype)
    utils.gems_assert_close(res_running_mean_out, ref_running_mean_out, dtype)
    utils.gems_assert_close(res_running_var_out, ref_running_var_out, dtype)
