import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# fp64 is not supported on every platform (e.g. ascend, iluvatar, kunlunxin).
_IGAMMA_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    _IGAMMA_DTYPES.append(torch.float64)


@pytest.mark.igamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _IGAMMA_DTYPES)
def test_igamma_(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    other = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)

    ref_out = ref_inp.igamma_(ref_other)
    with flag_gems.use_gems():
        res_out = inp.igamma_(other)

    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)
