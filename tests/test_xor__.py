import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# __xor__ only supports integer and boolean dtypes
INT_DTYPES = [torch.int16, torch.int32, torch.int64]
BOOL_TYPES = [torch.bool]


@pytest.mark.xor__
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_xor__(shape, dtype):
    if dtype in BOOL_TYPES:
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

    ref_out = ref_inp1 ^ ref_inp2
    with flag_gems.use_gems():
        res_out = inp1 ^ inp2

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.xor__
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_xor__inplace(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.__ixor__(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.__ixor__(inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.xor__
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_xor__scalar(shape, dtype):
    if dtype in BOOL_TYPES:
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

    ref_out = ref_inp1 ^ inp2
    with flag_gems.use_gems():
        res_out = inp1 ^ inp2

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.xor__
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES + BOOL_TYPES)
def test_xor__scalar_inplace(shape, dtype):
    if dtype in BOOL_TYPES:
        inp1 = torch.randint(0, 2, size=shape, dtype=dtype, device=flag_gems.device)
        inp2 = bool(random.randint(0, 2))
    else:
        inp1 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
        inp2 = 0x00FF
    ref_inp1 = utils.to_reference(inp1.clone())

    ref_out = ref_inp1.__ixor__(inp2)
    with flag_gems.use_gems():
        res_out = inp1.__ixor__(inp2)

    utils.gems_assert_equal(res_out, ref_out)
