import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_eigvals
@pytest.mark.parametrize("batch", [1, 4, 16, 64])
# Triton kernel uses float32 arithmetic for eigenvalue computation; Half/BFloat16 not supported
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_eigvals(batch, dtype):
    inp = torch.randn((batch, 2, 2), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.linalg.eigvals(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.linalg.eigvals(inp)

    res_out_sorted = torch.view_as_real(res_out)
    ref_out_sorted = torch.view_as_real(ref_out)

    utils.gems_assert_close(res_out_sorted, ref_out_sorted, dtype, rtol=1e-4, atol=1e-4)
