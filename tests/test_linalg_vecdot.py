import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.linalg_vecdot
@pytest.mark.parametrize(
    "shape",
    [
        (3,),
        (10,),
        (100,),
        (2, 3),
        (2, 10),
        (2, 100),
        (4, 3),
        (4, 10),
        (4, 100),
        (8, 3),
        (8, 10),
        (8, 100),
        (16, 3),
        (16, 10),
        (16, 100),
        (32, 10),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_linalg_vecdot(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)

    ref_out = torch.linalg.vecdot(ref_x, ref_y)
    with flag_gems.use_gems():
        res_out = flag_gems.linalg_vecdot(x, y)

    torch.testing.assert_close(res_out, ref_out, atol=1e-5, rtol=1e-5)
