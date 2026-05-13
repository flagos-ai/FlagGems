import pytest
import torch

import flag_gems
from flag_gems.ops.copy import _can_use_triton

from . import accuracy_utils as utils


@pytest.mark.copy_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype",
    utils.FLOAT_DTYPES + [torch.int32, torch.int64]
    if flag_gems.vendor_name == "cambricon"
    else utils.FLOAT_DTYPES,
)
def test_copy_inplace_same_dtype(shape, dtype):
    if flag_gems.vendor_name == "cambricon":
        if dtype in utils.FLOAT_DTYPES:
            src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        else:
            src = torch.randint(
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
                shape,
                dtype=dtype,
                device=flag_gems.device,
            )
    else:
        src = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_src = utils.to_reference(src)
    ref_dst = torch.zeros_like(ref_src)
    res_dst = torch.zeros_like(src)

    ref_dst.copy_(ref_src)
    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
def test_copy_inplace_broadcast():
    dst_shape = (2, 3)
    src = torch.arange(0, 3, dtype=torch.float32, device=flag_gems.device)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(
        torch.zeros(dst_shape, dtype=torch.float32, device=flag_gems.device)
    )
    res_dst = torch.zeros(dst_shape, dtype=torch.float32, device=flag_gems.device)

    ref_dst.copy_(ref_src)
    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
def test_copy_inplace_dtype_fallback():
    src = torch.arange(0, 8, dtype=torch.int32, device=flag_gems.device)
    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(
        torch.zeros(src.shape, dtype=torch.float32, device=flag_gems.device)
    )
    res_dst = torch.zeros(src.shape, dtype=torch.float32, device=flag_gems.device)

    ref_dst.copy_(ref_src)
    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="PyTorch does not support float8_e8m0fnu",
)
@pytest.mark.parametrize("shape", [(8,), (4, 4), (2, 3, 4)])
def test_copy_inplace_float8_e8m0fnu(shape):
    """Test that copy_ works correctly with float8_e8m0fnu (e8m0) dtype tensors.

    Triton does not recognize float8_e8m0fnu, so FlagGems should fallback to
    PyTorch's native copy_ implementation for this dtype.
    """
    device = flag_gems.device

    # e8m0 is an exponent-only format, create via view from uint8
    src_uint8 = torch.randint(0, 255, shape, dtype=torch.uint8, device=device)
    src = src_uint8.view(torch.float8_e8m0fnu)
    ref_src = utils.to_reference(src)

    ref_dst = utils.to_reference(
        torch.zeros(shape, dtype=torch.float8_e8m0fnu, device=device)
    )
    res_dst = torch.zeros(shape, dtype=torch.float8_e8m0fnu, device=device)
    ref_dst.copy_(ref_src)

    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.skipif(
    not hasattr(torch, "float8_e8m0fnu"),
    reason="PyTorch does not support float8_e8m0fnu",
)
def test_copy_inplace_float8_e8m0fnu_to_float32():
    """Test copy_ from float8_e8m0fnu to float32."""
    device = flag_gems.device
    shape = (8,)

    src_uint8 = torch.randint(1, 200, shape, dtype=torch.uint8, device=device)
    src = src_uint8.view(torch.float8_e8m0fnu)
    ref_src = utils.to_reference(src)

    ref_dst = utils.to_reference(torch.zeros(shape, dtype=torch.float32, device=device))
    res_dst = torch.zeros(shape, dtype=torch.float32, device=device)
    ref_dst.copy_(ref_src)

    with flag_gems.use_gems():
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)


@pytest.mark.copy_
@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (torch.float32, torch.int32),
        (torch.int16, torch.float32),
        (torch.bool, torch.float32),
    ],
)
def test_copy_inplace_mixed_dtype_triton(src_dtype, dst_dtype):
    device = flag_gems.device
    numel = 8

    if src_dtype is torch.bool:
        base = torch.tensor([True, False, True, True, False, True, False, True])
        src = base.to(device=device)
    else:
        if flag_gems.vendor_name == "mthreads":
            src = torch.arange(numel, device="cpu", dtype=src_dtype).to(device)
        else:
            src = torch.arange(numel, device=device, dtype=src_dtype)

    dst = torch.zeros(numel, dtype=dst_dtype, device=device)

    assert _can_use_triton(dst, src)

    ref_src = utils.to_reference(src)
    ref_dst = utils.to_reference(dst.clone())
    ref_dst.copy_(ref_src)

    with flag_gems.use_gems():
        res_dst = dst.clone()
        res_dst.copy_(src)

    utils.gems_assert_equal(res_dst, ref_dst)
