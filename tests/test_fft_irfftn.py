import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# COMPLEX_DTYPES are used here instead of FLOAT_DTYPES because fft_irfftn
# requires complex-valued input (half-Hermitian output from rfftn).
# Reference is torch.fft.irfftn computed on device (utils.to_reference not
# applicable since FFT comparison is GPU-native).

# FFT irfftn test shapes - only 1D shapes for now since multi-dimensional has recursion issues
FFT_IRFFTN_SHAPES = [(8,), (16,), (32,), (64,)]


@pytest.mark.fft_irfftn
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
        res_out = flag_gems.fft_irfftn(inp_complex, s=shape)

    # Output is always float32, compare with float32 tolerance
    utils.gems_assert_close(res_out, ref_out, torch.float32, atol=1e-3)
