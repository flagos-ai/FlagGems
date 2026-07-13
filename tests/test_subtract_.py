import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import SCALARS


@pytest.mark.subtract_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("alpha", SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_subtract_(shape, alpha, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)

    ref_out = ref_inp1.subtract_(ref_inp2, alpha=alpha)
    with flag_gems.use_gems():
        res_out = inp1.subtract_(inp2, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp1, dtype)
