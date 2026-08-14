import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# COMPLEX_DTYPES are used here instead of FLOAT_DTYPES because fft_irfftn
# requires complex-valued input (half-Hermitian output from rfftn).
# Reference is torch.fft.irfftn computed on device (utils.to_reference not
# applicable since FFT comparison is GPU-native).

# fft_irfftn is not supported on the metax backend.
# complex32 (half-precision complex) FFT is a CUDA/cuFFT-only feature; other
# backends (CPU MKL, metax maca, etc.) do not support torch.fft on complex32.


# N-D shapes (1D, 2D, 3D). The underlying kernels use a direct O(N^2)
# DFT, so transformed dimensions are kept small.
FFT_IRFFTN_SHAPES = [
    (8,),
    (16,),
    (32,),
    (64,),
    (4, 64),
    (8, 128),
    (2, 4, 32),
    (4, 8, 64),
    (3, 17),
    (5, 7, 11),
    (10,),
]


@pytest.mark.fft_irfftn
@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="Not supported on metax")
@pytest.mark.parametrize("shape", FFT_IRFFTN_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.COMPLEX_DTYPES
)  # fft_irfftn requires complex input
def test_fft_irfftn(shape, dtype):
    """Test fft_irfftn accuracy by roundtrip with rfftn."""
    # Generate a real input
    inp_real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    # Get the half-Hermitian complex output from rfftn
    # torch.fft.rfftn serves as the golden reference
    inp_complex = torch.fft.rfftn(inp_real)

    # Reference output from PyTorch
    ref_out = utils.to_reference(torch.fft.irfftn(inp_complex, s=shape))

    # Our implementation
    with flag_gems.use_gems():
        res_out = torch.fft.irfftn(inp_complex, s=shape)

    # Output is always float32, compare with float32 tolerance
    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=1e-3)


# complex32 (fp16 complex) inputs are constructed directly as half-Hermitian
# tensors. The real (last) dimension is power-of-two in all shapes below:
# torch.fft.irfftn on complex32 is backed by cuFFT, which only supports
# power-of-2 dimensions in half precision, so no torch reference can be
# generated for odd N to compare against.
FFT_IRFFTN_COMPLEX32_SHAPES = [
    (8,),
    (16,),
    (32,),
    (64,),
    (4, 64),
    (8, 128),
    (2, 4, 32),
    (4, 8, 64),
]


@pytest.mark.fft_irfftn
@pytest.mark.skipif(
    flag_gems.vendor_name != "nvidia", reason="Not supported on non-nvidia backends"
)
@pytest.mark.parametrize("shape", FFT_IRFFTN_COMPLEX32_SHAPES)
def test_fft_irfftn_complex32(shape):
    """Test fft_irfftn with complex32 (fp16 complex) input produces float16 output."""
    half_shape = list(shape)
    half_shape[-1] = shape[-1] // 2 + 1
    inp_complex = torch.complex(
        torch.randn(half_shape, dtype=torch.float16, device=flag_gems.device),
        torch.randn(half_shape, dtype=torch.float16, device=flag_gems.device),
    )

    ref_out = utils.to_reference(torch.fft.irfftn(inp_complex, s=shape))

    with flag_gems.use_gems():
        res_out = torch.fft.irfftn(inp_complex, s=shape)

    assert res_out.dtype == torch.float16, f"expected float16, got {res_out.dtype}"
    utils.gems_assert_close(res_out, ref_out, torch.float16, atol=5e-2)


# complex128 (fp64 complex) inputs are constructed directly as half-Hermitian
# tensors. The kernels compute in fp64, output is float64.
FFT_IRFFTN_COMPLEX128_SHAPES = FFT_IRFFTN_SHAPES


@pytest.mark.fft_irfftn
@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="Not supported on metax")
@pytest.mark.parametrize("shape", FFT_IRFFTN_COMPLEX128_SHAPES)
def test_fft_irfftn_complex128(shape):
    """Test fft_irfftn with complex128 (fp64 complex) input produces float64 output."""
    half_shape = list(shape)
    half_shape[-1] = shape[-1] // 2 + 1
    inp_complex = torch.complex(
        torch.randn(half_shape, dtype=torch.float64, device=flag_gems.device),
        torch.randn(half_shape, dtype=torch.float64, device=flag_gems.device),
    )

    ref_out = utils.to_reference(torch.fft.irfftn(inp_complex, s=shape))

    with flag_gems.use_gems():
        res_out = torch.fft.irfftn(inp_complex, s=shape)

    assert res_out.dtype == torch.float64, f"expected float64, got {res_out.dtype}"
    utils.gems_assert_close(res_out, ref_out, torch.float64, atol=1e-9)


@pytest.mark.fft_irfftn
@pytest.mark.skipif(flag_gems.vendor_name == "metax", reason="Not supported on metax")
@pytest.mark.parametrize("shape", FFT_IRFFTN_SHAPES)
def test_fft_irfftn_default_s(shape):
    """Test fft_irfftn with default s and dim (all dims transformed)."""
    inp_real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp_complex = torch.fft.rfftn(inp_real)

    ref_out = utils.to_reference(torch.fft.irfftn(inp_complex))
    with flag_gems.use_gems():
        res_out = torch.fft.irfftn(inp_complex)

    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=1e-3)
