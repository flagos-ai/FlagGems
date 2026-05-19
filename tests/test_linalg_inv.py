import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_inv
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 2, 2), (2, 3, 3)])
# linalg.inv does not support low precision dtypes (Half/BFloat16)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_inv(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = inp + torch.eye(shape[-1], dtype=dtype, device=flag_gems.device) * 2.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.linalg.inv(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.linalg.inv(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
