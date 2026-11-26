import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = inp2.clamp(min=-1.0 + 1e-6)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.ops.aten.special_xlog1py(ref_inp1, ref_inp2)
    res_out = flag_gems.special_xlog1py(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_special_xlog1py_small_int(shape, dtype):
    inp1 = torch.randint(-10, 10, shape, dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    inp2 = torch.randint(1, 10, shape, dtype=torch.int32, device=flag_gems.device).to(
        dtype
    )
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = torch.ops.aten.special_xlog1py(ref_inp1, ref_inp2)
    res_out = flag_gems.special_xlog1py(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.special_xlog1py
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not utils.FP8_DTYPES,
    reason="float8 accuracy coverage requires CUDA and PyTorch float8 support",
)
@pytest.mark.parametrize("dtype", utils.FP8_DTYPES)
def test_special_xlog1py_float8(dtype):
    """ATen xlog1py is not implemented for float8; verify the uint8 byte pattern."""
    # Seed so the bitwise byte-pattern comparison is deterministic; the fp8
    # cast of the fp32 reference can land on a rounding boundary for a
    # handful of values when drawn at random.
    utils.init_seed(7)
    inp1 = torch.randn(32, 17, dtype=torch.float32, device=flag_gems.device).to(dtype)
    inp2 = torch.randn(32, 17, dtype=torch.float32, device=flag_gems.device)
    inp2 = inp2.clamp(min=-0.5).to(dtype)
    ref_inp1 = utils.to_reference(inp1, True)
    ref_inp2 = utils.to_reference(inp2, True)

    res_out = flag_gems.special_xlog1py(inp1, inp2)

    assert res_out.dtype == dtype
    # Reference: compute in fp32, then re-encode as fp8.
    ref_out = torch.ops.aten.special_xlog1py(
        ref_inp1.cpu().float(), ref_inp2.cpu().float()
    ).to(dtype)
    utils.gems_assert_equal(res_out.view(torch.uint8), ref_out.view(torch.uint8))


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py_x_zero(dtype):
    """x == 0 and y is not NaN => result is 0."""
    x = torch.zeros((10,), dtype=dtype, device=flag_gems.device)
    y = torch.linspace(-0.5, 2.0, 10, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.ops.aten.special_xlog1py(ref_x, ref_y)

    res_out = flag_gems.special_xlog1py(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_xlog1py
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_xlog1py_y_nan(dtype):
    """y is NaN => result is NaN."""
    x = torch.randn((10,), dtype=dtype, device=flag_gems.device)
    y = torch.full((10,), float("nan"), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.ops.aten.special_xlog1py(ref_x, ref_y)

    res_out = flag_gems.special_xlog1py(x, y)

    assert torch.isnan(res_out).all(), "Expected all-NaN output when y is NaN"
    assert torch.isnan(ref_out).all(), "Reference should also produce NaN"
