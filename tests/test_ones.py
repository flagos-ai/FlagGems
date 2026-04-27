import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.ones
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
def test_ones(shape, dtype):
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.ones(shape, device="cpu" if cfg.TO_CPU else device)
    )

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.ones(shape, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.ones(shape, dtype=dtype, device="cpu" if cfg.TO_CPU else device)
    )
