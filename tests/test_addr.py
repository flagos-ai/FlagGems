import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    MN_SHAPES = [
        (1, 32),
    ]
    FLOAT_DTYPES = [torch.float32]
else:
    MN_SHAPES = [
        (1, 32),
        (160, 1024),
        (5333, 497),
    ]
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.addr
@pytest.mark.parametrize("M, N", MN_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_addr(M, N, dtype):
    input_tensor = torch.randn((M, N), dtype=dtype, device=flag_gems.device)
    vec1 = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    vec2 = torch.randn((N,), dtype=dtype, device=flag_gems.device)
    alpha = torch.randn((), dtype=dtype, device=flag_gems.device)
    beta = torch.randn((), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(input_tensor, True)
    ref_vec1 = utils.to_reference(vec1, True)
    ref_vec2 = utils.to_reference(vec2, True)

    ref_out = torch.addr(ref_inp, ref_vec1, ref_vec2, alpha=alpha, beta=beta)
    with flag_gems.use_gems():
        res_out = torch.addr(input_tensor, vec1, vec2, alpha=alpha, beta=beta)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)
