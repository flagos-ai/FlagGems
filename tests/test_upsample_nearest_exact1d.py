import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark._upsample_nearest_exact1d
@pytest.mark.parametrize("shape", [(2, 3, 16), (4, 8, 64), (8, 16, 256)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("factor", [2, 3])
def test_accuracy__upsample_nearest_exact1d(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_size = [shape[-1] * factor]

    ref_out = torch.ops.aten._upsample_nearest_exact1d(ref_x, out_size, None)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._upsample_nearest_exact1d(x, out_size, None)

    utils.gems_assert_close(res_out, ref_out, dtype)
