import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.bitwise_xor_tensor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_bitwise_xor_tensor(shape, dtype):
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

    ref_out = torch.bitwise_xor(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.bitwise_xor(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_xor_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_bitwise_xor_tensor_(shape, dtype):
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

    ref_out = ref_inp1.bitwise_xor_(ref_inp2)
    with flag_gems.use_gems():
        res_out = inp1.bitwise_xor_(inp2)

    utils.gems_assert_equal(res_out, ref_out)
