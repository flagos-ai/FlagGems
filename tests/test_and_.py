import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.and_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_and_tensor(shape, dtype):
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

    ref_out = ref_inp1.__and__(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.__and__(inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_
def test_and_bool_broadcast():
    inp1 = torch.tensor(
        [[True, False, True]], dtype=torch.bool, device=flag_gems.device
    )
    inp2 = torch.tensor(
        [[False], [False], [True]], dtype=torch.bool, device=flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.__and__(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.__and__(inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_and_integer_edge_cases(dtype):
    inp1 = torch.tensor([-1, -1, 0, 123, -123], dtype=dtype, device=flag_gems.device)
    inp2 = torch.tensor(
        [0x00FF, -1, 123, 0, -123], dtype=dtype, device=flag_gems.device
    )
    ref_inp1 = utils.to_reference(inp1)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = ref_inp1.__and__(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.__and__(inp2)

    utils.gems_assert_equal(res_out, ref_out)

@pytest.mark.and_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_and_scalar(shape, dtype):
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

    ref_out = ref_inp1.__and__(inp2)
    with flag_gems.use_gems():
        res_out = inp1.__and__(inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.and_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_and_scalar_tensor(shape, dtype):
    if dtype in utils.BOOL_TYPES:
        inp1 = bool(random.randint(0, 2))
        inp2 = torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    else:
        inp1 = 0x00FF
        inp2 = torch.randint(
            low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    ref_inp2 = utils.to_reference(inp2)

    ref_out = torch.bitwise_and(inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_and(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)
