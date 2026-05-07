import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.group_norm
@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (16, 3, 16, 16, 1),
        (32, 32, 32, 32, 8),
        (1, 32, 32, 32, 8),
        (1, 32, 32, 32, 16),
        (1, 64, 32, 32, 16),
        (1, 64, 32, 32, 32),
        (1, 64, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_group_norm(N, C, H, W, num_groups, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    ref_out = torch.nn.functional.group_norm(
        ref_inp, num_groups, weight=ref_weight, bias=ref_bias, eps=eps
    )

    with flag_gems.use_gems():
        res_out = torch.group_norm(
            res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.group_norm_backward
@pytest.mark.parametrize(
    "N, C, H, W, num_groups",
    [
        (16, 3, 16, 16, 1),
        (32, 32, 32, 32, 8),
        (1, 32, 32, 32, 8),
        (1, 32, 32, 32, 16),
        (1, 64, 32, 32, 16),
        (1, 64, 32, 32, 32),
        (1, 64, 32, 32, 64),
    ],
)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_group_norm_backward(N, C, H, W, num_groups, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_mean = torch.randn([N, num_groups], dtype=dtype, device=flag_gems.device)
    res_rstd = torch.randn([N, num_groups], dtype=dtype, device=flag_gems.device)

    if wb_none:
        res_weight = None
        output_mask = [True, False, False]
    else:
        res_weight = torch.randn(C, dtype=dtype, device=flag_gems.device)
        output_mask = [True, True, True]

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_mean = utils.to_reference(res_mean, True)
    ref_rstd = utils.to_reference(res_rstd, True)
    ref_weight = utils.to_reference(res_weight, True)

    group_size = C // num_groups
    HxW = H * W

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_group_norm_backward(
        ref_grad,
        ref_inp,
        ref_mean,
        ref_rstd,
        ref_weight,
        N,
        C,
        HxW,
        num_groups,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_group_norm_backward(
            res_grad,
            res_inp,
            res_mean,
            res_rstd,
            res_weight,
            N,
            C,
            HxW,
            num_groups,
            output_mask,
        )
    utils.gems_assert_close(
        res_in_grad, ref_in_grad, dtype, reduce_dim=group_size * HxW
    )
    if not wb_none:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=N * HxW
        )
        utils.gems_assert_close(res_bias_grad, ref_bias_grad, dtype, reduce_dim=N * HxW)
