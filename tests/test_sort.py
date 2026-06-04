import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    SORT_BATCH_SIZES = [4]
    SORT_HIDDENSIZES = [256, 2048]
    SORT_DESCENDING = [False]
    SORT_DIMS = [-1]
else:
    SORT_BATCH_SIZES = [4, 8]
    SORT_HIDDENSIZES = [1, 256, 2048, 9333, 65536, 32768, 128 * 1024, 256 * 1024]
    SORT_DESCENDING = [True, False]
    SORT_DIMS = [0, -1]


@pytest.mark.sort
@pytest.mark.parametrize("batch_size", SORT_BATCH_SIZES)
@pytest.mark.parametrize("hiddensize", SORT_HIDDENSIZES)
@pytest.mark.parametrize("descending", SORT_DESCENDING)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", SORT_DIMS)
def test_sort(batch_size, hiddensize, descending, dtype, dim):
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
    # we only implement stable sort, non-stable sort is undefined
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(
            y, dim=dim, stable=True, descending=descending
        )

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.sort_stable
@pytest.mark.parametrize("batch_size", SORT_BATCH_SIZES)
@pytest.mark.parametrize("hiddensize", SORT_HIDDENSIZES)
@pytest.mark.parametrize("descending", SORT_DESCENDING)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.INT_DTYPES)
@pytest.mark.parametrize("dim", SORT_DIMS)
def test_sort_stable(batch_size, hiddensize, descending, dtype, dim):
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
    ref_value, ref_index = torch.sort(
        ref_y, dim=dim, stable=True, descending=descending
    )

    with flag_gems.use_gems():
        res_value, res_index = torch.sort(
            y, dim=dim, stable=True, descending=descending
        )

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)
