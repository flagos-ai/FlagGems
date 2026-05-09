import pytest
import torch

import flag_gems

from .accuracy_utils import REDUCTION_SHAPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

DIM_LIST = [1] if QUICK_MODE else [0, 1]


@pytest.mark.index_copy
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_copy(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src_shape = list(inp.shape)
    index_len = src_shape[dim]
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_out = torch.index_copy(ref_inp, dim, ref_index, ref_src)
    with flag_gems.use_gems():
        res_out = torch.index_copy(inp, dim, index, src)
    gems_assert_close(res_out, ref_out, dtype=dtype, reduce_dim=dim)


@pytest.mark.index_copy_
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_index_copy_(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    src_shape = list(inp.shape)
    index_len = src_shape[dim]
    index = torch.randperm(index_len, device=flag_gems.device)
    src_shape[dim] = index_len
    src = torch.randn(src_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)
    ref_src = to_reference(src)
    ref_index = to_reference(index)
    ref_inp.index_copy_(dim, ref_index, ref_src)
    with flag_gems.use_gems():
        inp.index_copy_(dim, index, src)
    gems_assert_close(inp, ref_inp, dtype=dtype, reduce_dim=dim)
