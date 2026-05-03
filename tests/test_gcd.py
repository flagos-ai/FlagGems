import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def make_gcd_tensor(shape, dtype, values):
    info = torch.iinfo(dtype)
    tensor = torch.randint(info.min, info.max, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    flat = tensor.reshape(-1)
    if flat.numel() > 0:
        boundary = torch.tensor(values, dtype=dtype, device=flag_gems.device)
        flat[: min(flat.numel(), boundary.numel())] = boundary[: flat.numel()]
    return tensor


@pytest.mark.gcd
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd(shape, dtype):
    info = torch.iinfo(dtype)
    inp1 = make_gcd_tensor(shape, dtype, [0, -12, info.min, -27, 81])
    inp2 = make_gcd_tensor(shape, dtype, [6, 18, 0, -9, info.min])
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.gcd(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gcd(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_out(shape, dtype):
    info = torch.iinfo(dtype)
    inp1 = make_gcd_tensor(shape, dtype, [0, -12, info.min, -27, 81])
    inp2 = make_gcd_tensor(shape, dtype, [6, 18, 0, -9, info.min])
    out = torch.empty_like(inp1)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)
    ref_out = torch.empty_like(ref_inp1)

    torch.gcd(ref_inp1, ref_inp2, out=ref_out)
    with flag_gems.use_gems():
        torch.gcd(inp1, inp2, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_special_values(dtype):
    info = torch.iinfo(dtype)
    inp1 = torch.tensor(
        [0, 0, -12, -27, info.min, info.min, info.min, info.max],
        dtype=dtype,
        device=flag_gems.device,
    )
    inp2 = torch.tensor(
        [0, 6, 18, -9, 0, info.min, 2, info.min],
        dtype=dtype,
        device=flag_gems.device,
    )
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.gcd(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gcd(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_empty(dtype):
    inp1 = torch.empty((2, 0, 3), dtype=dtype, device=flag_gems.device)
    inp2 = torch.empty((2, 0, 3), dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    ref_out = torch.gcd(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.gcd(inp1, inp2)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.gcd
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_gcd_noncontiguous_broadcast(dtype):
    info = torch.iinfo(dtype)
    lhs = make_gcd_tensor((5, 7), dtype, [0, -12, info.min, -27, 81]).T
    rhs = make_gcd_tensor((1, 5), dtype, [6, 18, 0, -9, info.min])

    assert not lhs.is_contiguous()

    ref_lhs = utils.to_reference(lhs, False)
    ref_rhs = utils.to_reference(rhs, False)
    ref_out = torch.gcd(ref_lhs, ref_rhs)

    with flag_gems.use_gems():
        res_out = torch.gcd(lhs, rhs)

    utils.gems_assert_equal(res_out, ref_out)
