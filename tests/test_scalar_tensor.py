import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device


@pytest.mark.scalar_tensor
@pytest.mark.parametrize(
    "dtype", utils.BOOL_TYPES + utils.INT_DTYPES + utils.FLOAT_DTYPES
)
@pytest.mark.parametrize("fill_value", [0.01, 2, 0, -1, True, False])
def test_scalar_tensor(dtype, fill_value):
    ref_out = torch.scalar_tensor(
        fill_value, dtype=dtype, device="cpu" if cfg.TO_CPU else device
    )

    with flag_gems.use_gems():
        res_out = torch.scalar_tensor(fill_value, dtype=dtype, device=flag_gems.device)

    utils.gems_assert_equal(res_out, ref_out)
