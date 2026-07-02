import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.miopen_batch_norm
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
def test_miopen_batch_norm(shape, dtype, affine):
    """Test for miopen_batch_norm operator."""
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    # weight is required by miopen_batch_norm schema
    weight = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5
    momentum = 0.1

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    # Reference: use native_batch_norm with use_gems (same as our implementation)
    with flag_gems.use_gems():
        ref_out = torch.ops.aten.native_batch_norm(
            ref_inp,
            ref_weight,
            ref_bias,
            ref_running_mean,
            ref_running_var,
            True,  # training
            momentum,
            eps,
        )[0]

        res_out = torch.ops.aten.miopen_batch_norm(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            True,  # training
            momentum,
            eps,
        )[0]

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(running_mean, ref_running_mean, dtype)
    utils.gems_assert_close(running_var, ref_running_var, dtype)


@pytest.mark.miopen_batch_norm
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("affine", [True, False])
def test_miopen_batch_norm_eval(shape, dtype, affine):
    """Test for miopen_batch_norm operator in eval mode."""
    C = shape[1]
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    # weight is required by miopen_batch_norm schema
    weight = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    # Set running stats to some values for eval mode
    running_mean = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.rand(size=(C,), dtype=dtype, device=flag_gems.device) + 0.1

    eps = 1e-5
    momentum = 0.1

    ref_inp = utils.to_reference(inp, True)
    ref_weight = utils.to_reference(weight, True)
    ref_bias = utils.to_reference(bias, True)
    ref_running_mean = utils.to_reference(running_mean, True)
    ref_running_var = utils.to_reference(running_var, True)

    # Reference: use native_batch_norm with use_gems (same as our implementation)
    with flag_gems.use_gems():
        ref_out = torch.ops.aten.native_batch_norm(
            ref_inp,
            ref_weight,
            ref_bias,
            ref_running_mean,
            ref_running_var,
            False,  # eval mode
            momentum,
            eps,
        )[0]

        res_out = torch.ops.aten.miopen_batch_norm(
            inp,
            weight,
            bias,
            running_mean,
            running_var,
            False,  # eval mode
            momentum,
            eps,
        )[0]

    utils.gems_assert_close(res_out, ref_out, dtype)
