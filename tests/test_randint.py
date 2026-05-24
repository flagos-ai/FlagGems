import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.randint
@pytest.mark.parametrize("shape", utils.DISTRIBUTION_SHAPES)
@pytest.mark.parametrize("dtype", [torch.int64])
def test_randint(shape, dtype):
    high = 100

    with flag_gems.use_gems():
        res_out = torch.randint(high, shape, dtype=dtype, device=flag_gems.device)

    ref_out = utils.to_reference(res_out)

    assert (ref_out >= 0).all()
    assert (ref_out < high).all()

    # Statistical verification: sorted output should match sorted reference
    sorted_res, _ = torch.sort(res_out)
    sorted_ref, _ = torch.sort(ref_out)
    utils.gems_assert_equal(sorted_res, sorted_ref)
