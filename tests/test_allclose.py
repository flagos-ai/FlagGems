import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES


@pytest.mark.allclose
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("equal_nan", [False, True])
@pytest.mark.parametrize("gen_nan", [0, 1, 2, 3, 4])
def test_allclose(shape, dtype, equal_nan, gen_nan):
    # [gen_nan] 1: nan, 2: inf, 3: -inf, 4: inf vs -inf
    rtol = torch.rand(1, dtype=torch.float32, device=flag_gems.device).item() * (
        0.0001 if dtype in [torch.bfloat16, torch.float16] else 0.01
    )
    if dtype in utils.ALL_FLOAT_DTYPES:
        atol = (
            torch.finfo(dtype).tiny
            * torch.randint(0, 4, (1,), device=flag_gems.device).item()
        )
        inp1 = torch.full(shape, 1.234, dtype=dtype, device=flag_gems.device)
        inp2 = torch.full(shape, 1.234, dtype=dtype, device=flag_gems.device)
        if gen_nan:
            nan_num = torch.full(
                (1,),
                float("nan" if gen_nan == 1 else "inf"),
                dtype=dtype,
                device=flag_gems.device,
            )
            # FIXME: Neg doesn't support double on torch_musa, so workaround temporarily.
            inp1.view(-1)[0] = (
                (-nan_num.cpu()).to(flag_gems.device) if gen_nan == 3 else nan_num
            )
            inp2.view(-1)[0] = (
                (-nan_num.cpu()).to(flag_gems.device) if gen_nan >= 3 else nan_num
            )
    else:
        atol = (
            torch.finfo(torch.float16).eps
            * torch.randint(0, 10, (1,), device=flag_gems.device).item()
        )
        inp1 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
        inp2 = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)

    ref_inp1 = utils.to_reference(inp1, False)
    ref_inp2 = utils.to_reference(inp2, False)

    with flag_gems.use_gems():
        res_out = torch.allclose(inp1, inp2, rtol, atol, equal_nan=equal_nan)

    ref_out = torch.allclose(ref_inp1, ref_inp2, rtol, atol, equal_nan=equal_nan)

    assert res_out == ref_out
