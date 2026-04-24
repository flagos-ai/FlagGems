import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS_LIST = [1]
    KIND_KEEPDIM_DIMS_SHAPE = [("normal", True, 1, utils.REDUCTION_SHAPES[0])]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    KIND_KEEPDIM_DIMS_SHAPE = list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            utils.REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )


@pytest.mark.any
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
@pytest.mark.parametrize("kind", ["normal", "allFalse"])
def test_any(shape, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.any(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.any_dims
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.2"), reason="Skipping Pytorch version."
)
@pytest.mark.parametrize("kind, keepdim, dim, shape", KIND_KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + [torch.bool])
def test_any_dims(shape, dim, keepdim, dtype, kind):
    if kind == "allFalse":
        inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 2, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.any(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.any(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)
