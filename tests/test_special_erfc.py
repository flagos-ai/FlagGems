import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_erfc(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_erfc(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_erfc(ref_x)
    act_out = flag_gems.special_erfc(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.special_erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.SMALL_INT_DTYPES)
def test_special_erfc_small_int(shape, dtype):
    x = torch.randint(-10, 10, shape, dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    ref_x = utils.to_reference(x)
    ref_out = torch.ops.aten.special_erfc(ref_x)
    act_out = flag_gems.special_erfc(x)
    utils.gems_assert_close(act_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.special_erfc
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not utils.FP8_DTYPES,
    reason="float8 accuracy coverage requires CUDA and PyTorch float8 support",
)
@pytest.mark.parametrize("dtype", utils.FP8_DTYPES)
def test_special_erfc_float8(dtype):
    """ATen erfc is not implemented for float8; verify the uint8 byte pattern."""
    x = torch.randn(32, 17, dtype=torch.float32, device=flag_gems.device).to(dtype)

    res_out = flag_gems.special_erfc(x)

    assert res_out.dtype == dtype
    # Reference: compute in fp32, then re-encode as fp8.
    ref_out = torch.ops.aten.special_erfc(x.cpu().float()).to(dtype)
    utils.gems_assert_equal(res_out.view(torch.uint8), ref_out.view(torch.uint8))


@pytest.mark.erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.erfc(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.erfc(ref_x)
    act_out = flag_gems.erfc(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.erfc_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_erfc_(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        # Compute the reference in fp32 and cast back, mirroring the
        # non-inplace erfc tests. The inplace kernel writes through
        # out0=x and is bitwise-identical to the non-inplace one, so
        # low-precision dtypes are covered here too (issue #4076).
        ref_out = torch.ops.aten.erfc_(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.erfc_(ref_x)
    act_out = flag_gems.erfc_(x)
    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)
    utils.gems_assert_close(x, ref_out, dtype, equal_nan=True)


@pytest.mark.erfc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_erfc_byte(shape, dtype):
    # torch promotes integral erfc inputs to float32.
    x = torch.randint(1, 8, shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten.erfc(ref_x)
    act_out = flag_gems.erfc(x)
    utils.gems_assert_close(act_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.erfc_
def test_erfc_int_rejects_inplace():
    # torch.erfc_ rejects integral inputs -- "result type Float can't be
    # cast to the desired output type Int"; so does FlagGems.
    x = torch.tensor([1, 2, 3], dtype=torch.int32, device=flag_gems.device)
    ref_x = x.cpu()
    with pytest.raises(RuntimeError):
        torch.ops.aten.erfc_(ref_x)
    with pytest.raises(RuntimeError):
        flag_gems.erfc_(x)
