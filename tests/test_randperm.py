import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.randperm
@pytest.mark.parametrize("n", [123, 12345, 123456])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_randperm(n, dtype):
    if n > torch.iinfo(torch.int16).max and dtype == torch.int16:
        return

    # Skip int16 for Moore Threads backend due to runtime crash
    if flag_gems.vendor_name == "mthreads" and dtype == torch.int16:
        pytest.skip("Moore Threads int16 randperm causes runtime crash")

    ref_out = torch.randperm(n, dtype=dtype, device="cpu" if cfg.TO_CPU else device)
    with flag_gems.use_gems():
        res_out = torch.randperm(n, dtype=dtype, device=flag_gems.device)

    sorted_ref, _ = torch.sort(ref_out)
    sorted_res, _ = torch.sort(res_out)
    utils.gems_assert_equal(sorted_res, sorted_ref)
