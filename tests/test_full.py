import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.full
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
@pytest.mark.parametrize("fill_value", [3.1415926, 2, False])
def test_full(shape, dtype, fill_value):
    # without dtype
    ref_out = torch.full(shape, fill_value, device="cpu" if cfg.TO_CPU else device)

    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, device=flag_gems.device)

    utils.gems_assert_equal(res_out, ref_out)

    # with dtype
    ref_out = torch.full(
        shape, fill_value, dtype=dtype, device="cpu" if cfg.TO_CPU else device
    )

    with flag_gems.use_gems():
        res_out = torch.full(shape, fill_value, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(res_out, ref_out)
