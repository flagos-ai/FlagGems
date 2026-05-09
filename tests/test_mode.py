import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


@pytest.mark.mode
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_mode_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = inp.cpu()

    ref_out_value, ref_out_index = torch.mode(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.mode(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_value.cpu(), ref_out_value)
    # Verify the returned index actually points to the mode value
    gather_idx = res_out_index.cpu().reshape(
        list(ref_inp.shape[:dim]) + [1] + list(ref_inp.shape[dim + 1 :])
    )
    values_at_index = ref_inp.gather(dim, gather_idx).reshape(res_out_index.shape)
    utils.gems_assert_equal(values_at_index, ref_out_value)
