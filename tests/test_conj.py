import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.conj
@pytest.mark.parametrize("shape", [(256,), (32, 64), (2, 3, 4)])
# _conj only operates on complex dtypes (FLOAT_DTYPES/INT_DTYPES not applicable)
@pytest.mark.parametrize("dtype", utils.COMPLEX_DTYPES)
def test_conj(shape, dtype):
    # Create complex input
    real = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    imag = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    inp = torch.complex(real, imag).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.conj(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.conj(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
