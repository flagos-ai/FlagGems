import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Square matrix shapes for linalg_slogdet testing
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
# slogdet involves log and division, float16/bf16 lack precision
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_slogdet(shape, dtype):
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)

    ref_out = torch.linalg.slogdet(ref_A)

    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    utils.gems_assert_close(res_out.sign, ref_out.sign, dtype)
    utils.gems_assert_close(res_out.logabsdet, ref_out.logabsdet, dtype)
