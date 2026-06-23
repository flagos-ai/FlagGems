import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


def _assert_mode_matches(inp, dim, keepdim):
    ref_inp = inp.cpu()
    ref_out_value, _ = torch.mode(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.mode(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_value.cpu(), ref_out_value)
    gather_idx = res_out_index.cpu().reshape(
        list(ref_inp.shape[:dim]) + [1] + list(ref_inp.shape[dim + 1 :])
    )
    values_at_index = ref_inp.gather(dim, gather_idx).reshape(res_out_index.shape)
    utils.gems_assert_equal(values_at_index, ref_out_value)


@pytest.mark.mode
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_mode(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )

    _assert_mode_matches(inp, dim, keepdim)


@pytest.mark.mode
@pytest.mark.parametrize("dtype", [torch.float32, torch.int32])
@pytest.mark.parametrize(
    "data, dim, keepdim",
    [
        (
            [
                [1, 2, 2, 3],
                [4, 4, 4, 5],
                [6, 7, 7, 7],
            ],
            1,
            False,
        ),
        (
            [
                [1, 5, 2],
                [1, 5, 3],
                [4, 6, 3],
                [1, 7, 3],
            ],
            0,
            True,
        ),
    ],
)
def test_mode_repeated_values(data, dim, keepdim, dtype):
    inp = torch.tensor(data, dtype=dtype, device=flag_gems.device)

    _assert_mode_matches(inp, dim, keepdim)
