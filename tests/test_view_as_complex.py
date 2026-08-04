import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.view_as_complex
@pytest.mark.parametrize(
    "shape",
    [(64, 2), (128, 64, 2), (4096, 4096, 2), (64, 512, 512, 2)],
)
# view_as_complex only supports float32 and float64 per PyTorch docs
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_view_as_complex(shape, dtype):
    """Test view_as_complex accuracy against PyTorch implementation."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.view_as_complex(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.view_as_complex(inp)

    # Compare via view_as_real since complex tensors need special handling
    res_out_ref = utils.to_reference(torch.view_as_real(res_out))
    ref_out_ref = utils.to_reference(torch.view_as_real(ref_out))
    utils.gems_assert_close(res_out_ref, ref_out_ref, dtype)


@pytest.mark.view_as_complex
def test_view_as_complex_invalid_last_dim():
    """Test that view_as_complex raises RuntimeError when last dim != 2."""
    inp = torch.randn((64, 3), dtype=torch.float32, device=flag_gems.device)

    with pytest.raises(
        RuntimeError, match="expects a tensor with last dimension of size 2"
    ):
        with flag_gems.use_gems():
            torch.view_as_complex(inp)
