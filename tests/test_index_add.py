import random
import time

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    DIM_LIST = [1]
else:
    DIM_LIST = [0, 1]

random.seed(time.time() // 100)


@pytest.mark.index_add
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_out = torch.index_add(ref_inp, dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        res_out = torch.index_add(inp, dim, index, src, alpha=alpha)

    utils.gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_add_
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_add_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    src_shape = list(inp.shape)
    index_max = src_shape[dim]
    index_len = index_max
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    alpha = 2

    ref_inp = utils.to_reference(inp)
    ref_src = utils.to_reference(src)
    ref_index = utils.to_reference(index)
    ref_inp.index_add_(dim, ref_index, ref_src, alpha=alpha)
    with flag_gems.use_gems():
        inp.index_add_(dim, index, src, alpha=alpha)

    utils.gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)
