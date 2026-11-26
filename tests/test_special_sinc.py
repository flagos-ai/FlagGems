import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_sinc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_sinc(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.sinc(ref_inp)
    res_out = flag_gems.special_sinc(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_sinc
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_special_sinc_small_int(shape, dtype):
    inp = torch.randint(-10, 10, shape, dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.sinc(ref_inp)
    res_out = flag_gems.special_sinc(inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.special_sinc
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not utils.FP8_DTYPES,
    reason="float8 accuracy coverage requires CUDA and PyTorch float8 support",
)
@pytest.mark.parametrize("dtype", utils.FP8_DTYPES)
def test_special_sinc_float8(dtype):
    """ATen sinc is not implemented for float8; verify the uint8 byte pattern."""
    # Seed so the bitwise byte-pattern comparison is deterministic; the fp8
    # cast of the fp32 reference can land on a rounding boundary for a
    # handful of values when drawn at random.
    utils.init_seed(7)
    inp = torch.randn(32, 17, dtype=torch.float32, device=flag_gems.device).to(dtype)

    res_out = flag_gems.special_sinc(inp)

    assert res_out.dtype == dtype
    # Reference: compute in fp32, then re-encode as fp8.
    ref_out = torch.special.sinc(inp.cpu().float()).to(dtype)
    utils.gems_assert_equal(res_out.view(torch.uint8), ref_out.view(torch.uint8))
