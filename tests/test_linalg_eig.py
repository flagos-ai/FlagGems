import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_eig
@pytest.mark.parametrize("shape", [(2, 2), (4, 2, 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_linalg_eig(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_vals, ref_vecs = torch.linalg.eig(ref_inp)
    with flag_gems.use_gems():
        res_vals, res_vecs = torch.linalg.eig(inp)

    res_real = torch.real(res_vals)
    ref_real = torch.real(ref_vals)
    utils.gems_assert_close(res_real, ref_real, dtype)
