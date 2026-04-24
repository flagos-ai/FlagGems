import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)

device = flag_gems.device


@pytest.mark.dropout
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dropout(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
    else:
        utils.init_seed(0)

    if cfg.TO_CPU or shape == (1,):
        shape = (32768,)

    res_inp = torch.randn(
        shape,
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp = utils.to_reference(res_inp)

    p = np.float32(p)
    one_minus_p = np.float32(1.0) - p

    ref_out = torch.nn.functional.dropout(ref_inp, p, True)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.dropout(res_inp, p, True)

    res_out = utils.to_reference(res_out)
    exp_equal = (p * p + one_minus_p * one_minus_p) * res_inp.numel()
    num_equal = torch.sum(torch.isclose(ref_out, res_out)).item()

    if cfg.TO_CPU:
        from flag_gems.testing import RESOLUTION

        zero_equal = torch.eq(res_out, torch.zeros_like(res_out))
        num_zero = torch.sum(zero_equal).item()
        assert abs(num_zero / res_inp.numel() - p) <= 0.05
        scale_equal = torch.isclose(
            res_out, ref_inp / one_minus_p, rtol=RESOLUTION[dtype]
        )
        assert torch.all(torch.logical_or(zero_equal, scale_equal))
    else:
        assert (
            abs(num_equal - exp_equal) / exp_equal <= 0.05
        ), f"num_equal: {num_equal}, exp_equal: {exp_equal}, num_total: {res_inp.numel()}"


@pytest.mark.dropout_backward
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("p", [0.3, 0.6, 0.9])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_dropout_backward(shape, p, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    res_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_mask = torch.randint(0, 2, shape, dtype=torch.bool, device=flag_gems.device)
    ref_grad = utils.to_reference(res_grad)
    ref_mask = utils.to_reference(res_mask)

    scale = 1.0 / (1.0 - p)

    ref_in_grad = torch.ops.aten.native_dropout_backward(ref_grad, ref_mask, scale)
    with flag_gems.use_gems():
        res_in_grad = torch.ops.aten.native_dropout_backward(res_grad, res_mask, scale)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
