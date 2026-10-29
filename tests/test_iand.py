import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.bitwise_and_tensor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES + utils.BOOL_TYPES)
def test_iand(shape, dtype):
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

    ref_out = ref_inp1.__iand__(ref_inp2)
    before = (
        inp1.device,
        inp1.dtype,
        tuple(inp1.shape),
        inp1.stride(),
        inp1.data_ptr(),
    )
    other_before = (
        inp2.device,
        inp2.dtype,
        tuple(inp2.shape),
        inp2.stride(),
        inp2.data_ptr(),
    )
    other_bytes = inp2.cpu().contiguous().reshape(-1).view(torch.uint8).clone()
    res_out = flag_gems.bitwise_and_tensor_(inp1, inp2)
    assert isinstance(res_out, torch.Tensor)
    assert res_out is inp1
    assert (
        res_out.device,
        res_out.dtype,
        tuple(res_out.shape),
        res_out.stride(),
        res_out.data_ptr(),
    ) == before
    assert (
        inp2.device,
        inp2.dtype,
        tuple(inp2.shape),
        inp2.stride(),
        inp2.data_ptr(),
    ) == other_before
    assert torch.equal(
        inp2.cpu().contiguous().reshape(-1).view(torch.uint8), other_bytes
    )

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.INT_DTYPES)
@pytest.mark.parametrize("scalar", [0, 0x00FF, -1])
def test_iand_scalar_int(shape, dtype, scalar):
    inp1 = torch.randint(
        low=-0x7FFF, high=0x7FFF, size=shape, dtype=dtype, device="cpu"
    ).to(flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())

    ref_out = ref_inp1.__iand__(scalar)
    before = (
        inp1.device,
        inp1.dtype,
        tuple(inp1.shape),
        inp1.stride(),
        inp1.data_ptr(),
    )
    res_out = flag_gems.bitwise_and_scalar_(inp1, scalar)
    assert isinstance(res_out, torch.Tensor)
    assert res_out is inp1
    assert (
        res_out.device,
        res_out.dtype,
        tuple(res_out.shape),
        res_out.stride(),
        res_out.data_ptr(),
    ) == before

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.bitwise_and_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", [False, True])
def test_iand_scalar_bool(shape, scalar):
    inp1 = torch.randint(0, 2, size=shape, dtype=torch.bool, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone())

    ref_out = ref_inp1.__iand__(scalar)
    before = (
        inp1.device,
        inp1.dtype,
        tuple(inp1.shape),
        inp1.stride(),
        inp1.data_ptr(),
    )
    res_out = flag_gems.bitwise_and_scalar_(inp1, scalar)
    assert isinstance(res_out, torch.Tensor)
    assert res_out is inp1
    assert (
        res_out.device,
        res_out.dtype,
        tuple(res_out.shape),
        res_out.stride(),
        res_out.data_ptr(),
    ) == before

    utils.gems_assert_equal(res_out, ref_out)
