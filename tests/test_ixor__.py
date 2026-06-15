import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.ixor__
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_ixor__(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.ops.aten.__ixor__.Tensor(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__ixor__.Tensor(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp1, ref_inp1)


@pytest.mark.ixor___scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_ixor___scalar(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = utils.to_reference(inp1)

    ref_out = torch.ops.aten.__ixor__.Scalar(ref_inp1, inp2)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.__ixor__.Scalar(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp1, ref_inp1)
