import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.prelu
@pytest.mark.parametrize(
    "shape",
    [(2, 3), (128, 256), (512, 512), (4, 8, 16), (2, 32, 16, 16), (2, 128, 64, 64)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("weight_kind", ["scalar", "per_channel"])
def test_prelu(shape, dtype, weight_kind):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if weight_kind == "scalar":
        w = torch.randn((), dtype=dtype, device=flag_gems.device)
    else:
        c = shape[1]
        w = torch.randn((c,), dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_w = utils.to_reference(w)

    ref_out = torch.ops.aten.prelu(ref_x, ref_w)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.prelu(x, w)

    utils.gems_assert_close(res_out, ref_out, dtype)
