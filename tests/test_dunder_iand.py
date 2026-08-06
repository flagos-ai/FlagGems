import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.dunder_iand_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_dunder_iand(shape, dtype):
    if dtype in utils.BOOL_TYPES:
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

    ref_inp1 &= ref_inp2
    with flag_gems.use_gems():
        inp1 &= inp2

    utils.gems_assert_equal(inp1, ref_inp1)


@pytest.mark.dunder_iand_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("scalar", [0, 0x00FF, -1])
def test_dunder_iand_scalar_int(shape, dtype, scalar):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())

    ref_inp1 &= scalar
    with flag_gems.use_gems():
        inp1 &= scalar

    utils.gems_assert_equal(inp1, ref_inp1)


@pytest.mark.dunder_iand_scalar
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", [False, True])
def test_dunder_iand_scalar_bool(shape, scalar):
    inp1 = torch.randint(0, 2, size=shape, dtype=torch.bool, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())

    ref_inp1 &= scalar
    with flag_gems.use_gems():
        inp1 &= scalar

    utils.gems_assert_equal(inp1, ref_inp1)
