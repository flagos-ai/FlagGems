import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    ARANGE_ENDS = [100]
    ARANGE_DTYPES = [torch.float32]
else:
    ARANGE_ENDS = [10, 100, 1000, 5.0]
    ARANGE_DTYPES = [torch.float32, torch.float16, torch.int64]


@pytest.mark.arange
@pytest.mark.parametrize("end", ARANGE_ENDS)
@pytest.mark.parametrize("dtype", ARANGE_DTYPES)
def test_arange(end, dtype):
    with flag_gems.use_gems():
        res_out = torch.arange(end, dtype=dtype, device=flag_gems.device)
    ref_out = torch.arange(end, dtype=dtype, device="cpu")

    utils.gems_assert_equal(res_out.cpu(), ref_out)
