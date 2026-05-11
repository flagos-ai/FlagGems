import random
import time

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


@pytest.mark.topk
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", [128, 256])
@pytest.mark.parametrize("topk", [0, 5])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_topk(batch_size, hiddensize, topk, largest, dtype):
    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]
    ref_x = utils.to_reference(x)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        ref_x = ref_x.cuda()

    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        if cfg.TO_CPU:
            ref_value = ref_value.cpu()
            ref_index = ref_index.cpu()

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.topk
@pytest.mark.parametrize(
    "shape, topk",
    [
        ((16, 1024, 256), 256),
        ((8, 512, 32), 32),
        ((4, 128, 64), 64),
        ((2, 33, 128), 128),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_topk_3d_lastdim(shape, topk, dtype):
    batch_size = int(np.prod(shape[:-1]))
    hiddensize = shape[-1]

    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(shape)
    x_2d = x.reshape(batch_size, hiddensize)

    for bsz in range(batch_size):
        col_indices = torch.randperm(hiddensize)
        x_2d[bsz, :] = x_2d[bsz, col_indices]

    ref_x = utils.to_reference(x)
    ref_value, ref_index = torch.topk(ref_x, topk, dim=-1, largest=True, sorted=True)

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, dim=-1, largest=True, sorted=True)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)
