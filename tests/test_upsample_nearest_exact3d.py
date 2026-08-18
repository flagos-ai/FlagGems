import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.upsample_nearest_exact3d
@pytest.mark.parametrize("shape", [(2, 3, 8, 16, 16), (1, 1, 4, 32, 32)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("factor", [2, 3, 1.5, 2.5])
def test_upsample_nearest_exact3d(shape, dtype, factor):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    out_size = [int(shape[2] * factor), int(shape[3] * factor), int(shape[4] * factor)]
    ref_out = torch.ops.aten._upsample_nearest_exact3d(
        ref_x, out_size, None, None, None
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._upsample_nearest_exact3d(
            x, out_size, None, None, None
        )
    utils.gems_assert_close(res_out, ref_out, dtype)
