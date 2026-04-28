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
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.4"),
    reason="The copy operator implement required for torch >= 2.4",
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
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.4"),
    reason="The copy operator implement required for torch >= 2.4",
)
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
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.4"),
    reason="The copy operator implement required for torch >= 2.4",
)
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
    utils.SkipVersion("torch", "<2.4"),
    reason="The copy operator implement required for torch >= 2.4",
)
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
