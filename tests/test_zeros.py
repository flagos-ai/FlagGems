import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.zeros
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.ALL_INT_DTYPES + utils.ALL_FLOAT_DTYPES
)
def test_zeros(shape, dtype):
    expected_dev = "cpu" if cfg.TO_CPU else device
    # without dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, device=flag_gems.device)

    utils.gems_assert_equal(res_out, torch.zeros(shape, device=expected_dev))

    # with dtype
    with flag_gems.use_gems():
        res_out = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(
        res_out, torch.zeros(shape, dtype=dtype, device=expected_dev)
    )
