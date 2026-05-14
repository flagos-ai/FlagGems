import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Define shapes for linalg_slogdet (square matrices)
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_accuracy_linalg_slogdet(shape, dtype):
    """Test linalg_slogdet accuracy against PyTorch reference."""
    # Ensure we have a square matrix
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    # Create input tensor
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_A = to_reference(A)

    # Compute reference
    ref_out = torch.linalg.slogdet(ref_A)

    # Compute with FlagGems
    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    # Compare sign
    gems_assert_close(res_out.sign, ref_out.sign, dtype)

    # Compare logabsdet (more tolerant for floating point)
    gems_assert_close(res_out.logabsdet, ref_out.logabsdet, dtype)
