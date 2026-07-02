import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unsqueeze_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_unsqueeze_copy(shape, dtype, dim):
    if len(shape) == 0:
        pytest.skip("skip scalar")

    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.ops.aten.unsqueeze_copy(ref_inp, dim)

    with flag_gems.use_gems():
        res_out = flag_gems.unsqueeze_copy(res_inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype)

    assert res_out.data_ptr() != res_inp.data_ptr()
