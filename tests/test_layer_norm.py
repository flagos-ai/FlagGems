import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    LAYER_NORM_SHAPES = [(1, 40999)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    LAYER_NORM_SHAPES = [(200, 36), (4096, 100), (1, 40999), (100, 40499), (4096, 256)]


@pytest.mark.layer_norm
@pytest.mark.parametrize("shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm(shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    ref_inp = utils.to_reference(res_inp, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    ref_out = torch.layer_norm(
        ref_inp,
        shape[1:],
        weight=ref_weight,
        bias=ref_bias,
        eps=eps,
    )
    with flag_gems.use_gems():
        res_out = torch.layer_norm(
            res_inp,
            shape[1:],
            weight=res_weight,
            bias=res_bias,
            eps=eps,
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.layer_norm_backward
@pytest.mark.parametrize("shape", LAYER_NORM_SHAPES)
@pytest.mark.parametrize("wb_none", [False, True])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_layer_norm_backward(monkeypatch, shape, dtype, wb_none):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if flag_gems.vendor_name == "mthreads":
        # Compatible with older versions of LLVM
        monkeypatch.env("DISABLE_LLVM_OPT", "1")

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad = torch.randn_like(res_inp)
    res_mean = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
    res_rstd = torch.randn(shape[0], dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
        output_mask = [True, False, False]
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        output_mask = [True, True, True]

    normalized_shape = shape[1:]

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad = utils.to_reference(res_grad, True)
    ref_mean = utils.to_reference(res_mean, True)
    ref_rstd = utils.to_reference(res_rstd, True)
    ref_weight = utils.to_reference(res_weight, True)
    ref_bias = utils.to_reference(res_bias, True)

    (
        ref_in_grad,
        ref_weight_grad,
        ref_bias_grad,
    ) = torch.ops.aten.native_layer_norm_backward(
        ref_grad,
        ref_inp,
        normalized_shape,
        ref_mean,
        ref_rstd,
        ref_weight,
        ref_bias,
        output_mask,
    )
    with flag_gems.use_gems():
        (
            res_in_grad,
            res_weight_grad,
            res_bias_grad,
        ) = torch.ops.aten.native_layer_norm_backward(
            res_grad,
            res_inp,
            normalized_shape,
            res_mean,
            res_rstd,
            res_weight,
            res_bias,
            output_mask,
        )

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    if not wb_none:
        utils.gems_assert_close(
            res_weight_grad, ref_weight_grad, dtype, reduce_dim=shape[0]
        )
        utils.gems_assert_close(
            res_bias_grad, ref_bias_grad, dtype, reduce_dim=shape[0]
        )
