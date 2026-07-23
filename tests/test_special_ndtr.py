import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_ndtr
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_special_ndtr(shape, dtype):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)
    if dtype in (torch.float16, torch.bfloat16):
        ref_out = torch.ops.aten.special_ndtr(ref_x.float()).to(dtype)
    else:
        ref_out = torch.ops.aten.special_ndtr(ref_x)
    with flag_gems.use_gems():
        act_out = torch.ops.aten.special_ndtr(x)
    # ndtr uses a polynomial approximation which introduces numerical differences
    # from the reference implementation; tolerances are set per-dtype to account for
    # the limited precision of fp16 and bf16.
    atol = (
        5e-4 if dtype == torch.float16 else (2e-3 if dtype == torch.bfloat16 else 1e-4)
    )
    utils.gems_assert_close(act_out, ref_out, dtype, atol=atol)
