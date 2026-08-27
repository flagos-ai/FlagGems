import pytest
import torch

import flag_gems
from tests import accuracy_utils as utils


@pytest.mark.imag
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_imag_complex(shape, dtype):
    """Test imag accuracy for complex tensors."""
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    ref_out = ref_inp.imag
    with flag_gems.use_gems():
        res_out = inp.imag

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.imag
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_imag_real(shape, dtype):
    """Test imag for real tensors returns zeros."""
    device = flag_gems.device
    inp = torch.randn(shape, dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)

    # CPU reference backend does not support .imag for real tensors.
    # Manually construct zero-filled reference output.
    ref_out = torch.zeros_like(ref_inp)
    with flag_gems.use_gems():
        res_out = inp.imag

    utils.gems_assert_equal(res_out, ref_out)
