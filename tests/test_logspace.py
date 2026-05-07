import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg


@pytest.mark.logspace
@pytest.mark.parametrize("start", [0, 2, 4])
@pytest.mark.parametrize("end", [32, 40])
@pytest.mark.parametrize("steps", [0, 1, 8, 17])
@pytest.mark.parametrize("base", [1.2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.ALL_INT_DTYPES + [None])
@pytest.mark.parametrize("pin_memory", [False])
def test_logspace(start, end, steps, base, dtype, pin_memory):
    if (
        flag_gems.vendor_name == "kunlunxin"
        and dtype is torch.half
        and torch.__version__ < "2.5"
    ):
        pytest.skip("#2828: Implementation for lerp cpu half is not ready")

    temp = torch.logspace(
        start,
        end,
        steps,
        base,
        dtype=dtype,
        layout=None,
        device="cpu",
        pin_memory=pin_memory,
    )

    ref_out = temp.to("cpu" if cfg.TO_CPU else flag_gems.device)

    # compute on cpu and move back to device
    with flag_gems.use_gems():
        res_out = torch.logspace(
            start,
            end,
            steps,
            base,
            dtype=dtype,
            layout=None,
            device=flag_gems.device,
            pin_memory=pin_memory,
        )

    if dtype in [torch.float16, torch.bfloat16, torch.float32, None]:
        utils.gems_assert_close(res_out, ref_out, dtype=dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
