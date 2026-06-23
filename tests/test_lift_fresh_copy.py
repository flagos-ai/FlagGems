import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    LIFT_FRESH_COPY_SHAPES = [(2, 3)]
else:
    LIFT_FRESH_COPY_SHAPES = [(2, 3), (128, 256), (512, 512)]


@pytest.mark.lift_fresh_copy
@pytest.mark.parametrize("shape", LIFT_FRESH_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_lift_fresh_copy(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten.lift_fresh_copy(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.lift_fresh_copy(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)
