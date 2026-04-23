import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

ARGSORT_HIDDENSIZE = (
    [1, 256, 2048, 9333]
    if QUICK_MODE
    else [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
)


@pytest.mark.argsort
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", ARGSORT_HIDDENSIZE)
@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_accuracy_argsort(batch_size, hiddensize, descending, dtype, dim):
    if dtype in utils.BOOL_TYPES:
        y = torch.randint(
            0, 2, (batch_size, hiddensize), dtype=dtype, device=flag_gems.device
        )
    elif dtype in utils.ALL_INT_DTYPES:
        min_v, max_v = torch.iinfo(dtype).min, torch.iinfo(dtype).max
        y = torch.randint(
            min_v, max_v, (batch_size, hiddensize), dtype=dtype, device="cpu"
        ).to(flag_gems.device)
    else:
        y = torch.randn((batch_size, hiddensize), dtype=dtype, device=flag_gems.device)

    ref_y = utils.to_reference(y)
    ref_index = torch.argsort(ref_y, dim=dim, stable=True, descending=descending)

    with flag_gems.use_gems():
        res_index = torch.argsort(y, dim=dim, stable=True, descending=descending)

    utils.gems_assert_equal(res_index, ref_index)
