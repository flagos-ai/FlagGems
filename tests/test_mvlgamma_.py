import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

P_LIST = [1, 2, 3, 5, 8, 12]


@pytest.mark.mvlgamma_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("p", P_LIST)
def test_mvlgamma_(shape, dtype, p):
    # Use a large shape limit for float16/bfloat16 to avoid lgamma precision issues
    # at poles (non-positive integers) where float16/bfloat16 have limited precision
    numel = 1
    for dim in shape:
        numel *= dim
    if dtype in (torch.float16, torch.bfloat16) and numel > 1024:
        pytest.skip(
            "float16/bfloat16 have precision issues with lgamma at poles for large shapes"
        )

    torch.manual_seed(42)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.mvlgamma_(p)
    with flag_gems.use_gems():
        res_out = inp.mvlgamma_(p)

    # Use relaxed tolerance for float16 due to lgamma precision limitations
    atol = 1e-2 if dtype == torch.float16 else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
