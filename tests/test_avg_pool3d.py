import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

AVGPOOL3D_CONFIGS = [
    ((2, 3, 8, 16, 16), 2, 2, 0, False, True, None),
    ((2, 3, 9, 17, 19), 3, 2, 1, False, True, None),
    ((2, 3, 9, 17, 19), 3, 2, 1, False, False, None),
    ((1, 4, 7, 9, 11), (2, 3, 4), (1, 2, 3), (0, 1, 1), False, True, None),
    ((1, 2, 5, 7, 9), (2, 3, 4), (2, 3, 4), 0, True, True, None),
    ((1, 2, 5, 7, 9), (2, 3, 4), (2, 3, 4), 0, True, False, None),
    ((1, 1, 4, 4, 4), 2, 1, 0, False, True, 3),
    ((3, 5, 7, 9), 2, None, 0, False, True, None),
    ((0, 3, 4, 4, 4), 2, 2, 0, False, True, None),
]

INVALID_AVGPOOL3D_CONFIGS = [
    ((1, 1, 4, 4, 4), 2, 0, 0, False, True, None),
    ((1, 1, 4, 4, 4), 2, 1, 2, False, True, None),
    ((1, 1, 4, 4, 4), 0, 1, 0, False, True, None),
    ((1, 1, 4, 4, 4), 2, 1, 0, False, True, 0),
    ((1, 1, 0, 4, 4), 2, 1, 0, False, True, None),
    ((1, 4, 4), 2, 1, 0, False, True, None),
]

random.seed(time.time() // 100)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override",
    AVGPOOL3D_CONFIGS,
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool3d(
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
    dtype,
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=kernel_size,
        stride=[] if stride is None else stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    res_out = flag_gems.avg_pool3d(
        inp,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool3d_non_contiguous(dtype):
    base = torch.randn((2, 3, 5, 7, 18), dtype=dtype, device=flag_gems.device)
    inp = base[..., ::2]
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=(2, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
        ceil_mode=True,
        count_include_pad=False,
        divisor_override=None,
    )

    res_out = flag_gems.avg_pool3d(
        inp,
        kernel_size=(2, 3, 3),
        stride=(1, 2, 2),
        padding=(0, 1, 1),
        ceil_mode=True,
        count_include_pad=False,
        divisor_override=None,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_avg_pool3d_special_values(dtype):
    inp = torch.arange(64, dtype=torch.float32, device=flag_gems.device).reshape(
        1, 1, 4, 4, 4
    )
    inp = inp.to(dtype)
    inp.flatten()[0] = float("nan")
    inp.flatten()[1] = float("inf")
    inp.flatten()[2] = float("-inf")
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=2,
        stride=1,
        padding=1,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )
    res_out = flag_gems.avg_pool3d(
        inp,
        kernel_size=2,
        stride=1,
        padding=1,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.avg_pool3d
def test_avg_pool3d_aten_registration():
    inp = torch.randn((1, 2, 4, 5, 6), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.ops.aten.avg_pool3d(
        ref_inp,
        kernel_size=2,
        stride=[],
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    )

    with flag_gems.use_gems():
        res_out = torch.nn.functional.avg_pool3d(inp, kernel_size=2)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.avg_pool3d
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override",
    INVALID_AVGPOOL3D_CONFIGS,
)
def test_avg_pool3d_invalid_inputs(
    shape,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    count_include_pad,
    divisor_override,
):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.avg_pool3d(
            inp,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
            divisor_override=divisor_override,
        )
