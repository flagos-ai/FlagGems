import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "params",
    [
        {"shape": (1, 1, 4, 4, 4), "kernel_size": 2, "stride": 2, "padding": 0},
        {"shape": (2, 3, 8, 8, 8), "kernel_size": 3, "stride": 1, "padding": 1},
        {"shape": (1, 1, 16, 16, 16), "kernel_size": 4, "stride": 2, "padding": 1},
        {"shape": (2, 4, 6, 6, 6), "kernel_size": 2, "stride": 2, "padding": 0},
        {"shape": (1, 1, 1, 1, 1), "kernel_size": 1, "stride": 1, "padding": 0},
        {"shape": (1, 1, 32, 32, 32), "kernel_size": 2, "stride": 2, "padding": 0},
    ],
)
def test_accuracy_max_pool3d(params, dtype):
    shape = params["shape"]
    kernel_size = params["kernel_size"]
    stride = params["stride"]
    padding = params["padding"]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.max_pool3d(
        ref_inp, kernel_size, stride=stride, padding=padding
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.max_pool3d(
            inp, kernel_size, stride=stride, padding=padding
        )

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_max_pool3d_return_indices(dtype):
    inp = torch.randn((1, 1, 8, 8, 8), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_idx = torch.nn.functional.max_pool3d(
        ref_inp, 2, stride=2, return_indices=True
    )
    with flag_gems.use_gems():
        res_out, res_idx = torch.nn.functional.max_pool3d(
            inp, 2, stride=2, return_indices=True
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_equal(res_idx, ref_idx)


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_max_pool3d_dilation(dtype):
    inp = torch.randn((1, 1, 16, 16, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nn.functional.max_pool3d(
        ref_inp, 3, stride=1, padding=1, dilation=1
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.max_pool3d(
            inp, 3, stride=1, padding=1, dilation=1
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
