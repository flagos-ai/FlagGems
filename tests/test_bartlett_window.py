import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    WINDOW_LENGTHS = [100]
else:
    WINDOW_LENGTHS = [0, 1, 2, 3, 10, 100, 1000, 4096, 10000]


@pytest.mark.bartlett_window
@pytest.mark.parametrize("window_length", WINDOW_LENGTHS)
@pytest.mark.parametrize("periodic", [True, False])
# bartlett_window only supports float32 (torch.bartlett_window default dtype)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_bartlett_window(window_length, periodic, dtype):
    ref_out = torch.bartlett_window(
        window_length, periodic=periodic, dtype=dtype, device="cpu"
    )
    with flag_gems.use_gems():
        res_out = torch.bartlett_window(
            window_length, periodic=periodic, dtype=dtype, device=flag_gems.device
        )
    if window_length > 1:
        utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        utils.gems_assert_equal(res_out, ref_out)
