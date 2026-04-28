import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark._functional_sym_constrain_range_for_size
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_functional_sym_constrain_range_for_size(shape, dtype):
    torch.manual_seed(0)
    dep_token = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_dep = utils.to_reference(dep_token)
    ref_out = torch.ops.aten._functional_sym_constrain_range_for_size(5, 1, 10, ref_dep)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._functional_sym_constrain_range_for_size(
            5, 1, 10, dep_token
        )
    utils.gems_assert_close(res_out, ref_out, dtype)
