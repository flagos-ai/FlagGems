import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes that support inplace bitwise operations with broadcasting
INPLACE_BITWISE_SHAPES = [
    ((512, 1024), (512, 1024)),
    ((256, 512), (1, 512)),
    ((256, 512), (256, 1)),
    ((1024,), ()),
]


@pytest.mark.bitwise_right_shift_
@pytest.mark.parametrize("shapes", INPLACE_BITWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.uint8])
def test_irshift(shapes, dtype):
    shape_a, shape_b = shapes

    res_a = torch.randint(0, 100, shape_a, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    res_b = torch.randint(0, 8, shape_b, dtype=dtype, device="cpu").to(flag_gems.device)

    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    # PyTorch reference: modify ref_a in-place.
    ref_a.__irshift__(ref_b)

    data_ptr = res_a.data_ptr()

    before = (
        res_a.device,
        res_a.dtype,
        tuple(res_a.shape),
        res_a.stride(),
        res_a.data_ptr(),
    )
    other_before = (
        res_b.device,
        res_b.dtype,
        tuple(res_b.shape),
        res_b.stride(),
        res_b.data_ptr(),
    )
    other_bytes = res_b.cpu().contiguous().reshape(-1).view(torch.uint8).clone()
    ret = flag_gems.bitwise_right_shift_(res_a, res_b)
    assert isinstance(ret, torch.Tensor)
    assert ret is res_a
    assert (
        ret.device,
        ret.dtype,
        tuple(ret.shape),
        ret.stride(),
        ret.data_ptr(),
    ) == before
    assert (
        res_b.device,
        res_b.dtype,
        tuple(res_b.shape),
        res_b.stride(),
        res_b.data_ptr(),
    ) == other_before
    assert torch.equal(
        res_b.cpu().contiguous().reshape(-1).view(torch.uint8), other_bytes
    )

    assert ret is res_a
    assert res_a.data_ptr() == data_ptr
    utils.gems_assert_close(res_a, ref_a, dtype)


@pytest.mark.bitwise_right_shift_
@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_irshift_edge_cases(dtype):
    # Signed values cover negative-number right shift behavior.
    # Shift values cover zero shift and relatively large shifts.
    res_a = torch.tensor(
        [-128, -64, -8, -1, 0, 1, 8, 64],
        dtype=dtype,
        device=flag_gems.device,
    )
    res_b = torch.tensor(
        [0, 1, 2, 3, 4, 5, 6, 7],
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_a = utils.to_reference(res_a.clone())
    ref_b = utils.to_reference(res_b)

    ref_a.__irshift__(ref_b)

    data_ptr = res_a.data_ptr()

    before = (
        res_a.device,
        res_a.dtype,
        tuple(res_a.shape),
        res_a.stride(),
        res_a.data_ptr(),
    )
    other_before = (
        res_b.device,
        res_b.dtype,
        tuple(res_b.shape),
        res_b.stride(),
        res_b.data_ptr(),
    )
    other_bytes = res_b.cpu().contiguous().reshape(-1).view(torch.uint8).clone()
    ret = flag_gems.bitwise_right_shift_(res_a, res_b)
    assert isinstance(ret, torch.Tensor)
    assert ret is res_a
    assert (
        ret.device,
        ret.dtype,
        tuple(ret.shape),
        ret.stride(),
        ret.data_ptr(),
    ) == before
    assert (
        res_b.device,
        res_b.dtype,
        tuple(res_b.shape),
        res_b.stride(),
        res_b.data_ptr(),
    ) == other_before
    assert torch.equal(
        res_b.cpu().contiguous().reshape(-1).view(torch.uint8), other_bytes
    )

    assert ret is res_a
    assert res_a.data_ptr() == data_ptr
    utils.gems_assert_close(res_a, ref_a, dtype)
