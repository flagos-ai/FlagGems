import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.mish_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_mish_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.ops.aten.mish_(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.mish_(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out.data_ptr() == inp.data_ptr(), "mish_ must operate in-place"
