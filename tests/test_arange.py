import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.arange
@pytest.mark.parametrize("end", [10, 100, 1000, 5.0])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.int64])
def test_arange(end, dtype):
    with flag_gems.use_gems():
        res_out = torch.arange(end, dtype=dtype, device=flag_gems.device)
    ref_out = torch.arange(end, dtype=dtype, device="cpu")

    utils.gems_assert_equal(res_out.cpu(), ref_out)
