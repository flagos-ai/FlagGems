import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

vendor_name = flag_gems.vendor_name


@pytest.mark.linalg_vecdot
@pytest.mark.parametrize("shape", utils.UT_SHAPES_2D + utils.UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0])
def test_linalg_vecdot(shape, dtype, dim):
    if vendor_name in ["kunlunxin", "tsingmicro"]:
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    if vendor_name in ["mthreads", "tsingmicro"]:
        x = torch.randn(shape, dtype=dtype, device="cpu")
        y = torch.randn(shape, dtype=dtype, device="cpu")
    else:
        x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)

    ref_out = torch.linalg.vecdot(ref_x, ref_y, dim=dim)
    res_out = flag_gems.linalg_vecdot(x, y, dim=dim)

    if dtype == torch.float16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-1)
    elif dtype == torch.bfloat16:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=5e-1)
    elif dtype == torch.float32 and len(shape) >= 2:
        utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-2)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype)
