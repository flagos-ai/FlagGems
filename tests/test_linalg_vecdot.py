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
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)

    ref_out = torch.linalg.vecdot(ref_x, ref_y, dim=dim)
    res_out = flag_gems.linalg_vecdot(x, y, dim=dim)

    if dim < 0:
        dim = dim % len(shape)
    vec_dim = shape[dim]

    if dtype in (torch.float16, torch.bfloat16):
        utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=2048)
    else:
        utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=vec_dim)


@pytest.mark.linalg_vecdot_out
@pytest.mark.parametrize("shape", utils.UT_SHAPES_2D + utils.UT_SHAPES_1D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0])
def test_linalg_vecdot_out(shape, dtype, dim):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    y = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x)
    ref_y = utils.to_reference(y)

    ref_out = torch.linalg.vecdot(ref_x, ref_y, dim=dim)
    out = flag_gems.linalg_vecdot_out(x, y, dim=dim)

    if dim < 0:
        dim = dim % len(shape)
    vec_dim = shape[dim]

    if dtype in (torch.float16, torch.bfloat16):
        utils.gems_assert_close(out, ref_out, dtype, reduce_dim=2048)
    else:
        utils.gems_assert_close(out, ref_out, dtype, reduce_dim=vec_dim)
