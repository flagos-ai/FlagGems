import pytest
import torch

import flag_gems
from flag_gems.ops.special_log1p import special_log1p

from . import accuracy_utils as utils


@pytest.mark.special_log1p
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_log1p(shape, dtype):
    utils.init_seed(0)
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out = torch.special.log1p(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.special.log1p(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_log1p
@pytest.mark.parametrize("inp", [1.0, 5, -0.5])
def test_special_log1p_non_tensor(inp):
    ref_out = torch.special.log1p(torch.tensor(inp, dtype=torch.float32))
    res_out = special_log1p(inp)
    utils.gems_assert_close(ref_out, res_out, torch.float32)
