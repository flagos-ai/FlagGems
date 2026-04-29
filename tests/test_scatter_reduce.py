import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

SCATTER_REDUCE_SHAPES = [
    (4, 8),
    (16, 32),
    (64, 64),
    (128, 256),
]

REDUCE_OPS = ["sum", "prod", "amax", "amin"]


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("shape", SCATTER_REDUCE_SHAPES)
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_dim0(shape, reduce, dtype):
    rows, cols = shape
    src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, rows, shape, device=flag_gems.device)
    base = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    ref_base = utils.to_reference(base.clone(), False)
    ref_src = utils.to_reference(src, False)
    ref_idx = utils.to_reference(idx, False)
    ref_out = ref_base.scatter_reduce_(0, ref_idx, ref_src, reduce=reduce, include_self=True)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(base, 0, idx, src, reduce=reduce, include_self=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("shape", SCATTER_REDUCE_SHAPES)
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_dim1(shape, reduce, dtype):
    rows, cols = shape
    src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, cols, shape, device=flag_gems.device)
    base = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    ref_base = utils.to_reference(base.clone(), False)
    ref_src = utils.to_reference(src, False)
    ref_idx = utils.to_reference(idx, False)
    ref_out = ref_base.scatter_reduce_(1, ref_idx, ref_src, reduce=reduce, include_self=True)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(base, 1, idx, src, reduce=reduce, include_self=True)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_exclude_self(reduce, dtype):
    shape = (8, 16)
    rows, cols = shape
    src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, rows, shape, device=flag_gems.device)
    base = torch.ones(shape, dtype=dtype, device=flag_gems.device)

    ref_base = utils.to_reference(base.clone(), False)
    ref_src = utils.to_reference(src, False)
    ref_idx = utils.to_reference(idx, False)
    ref_out = ref_base.scatter_reduce_(0, ref_idx, ref_src, reduce=reduce, include_self=False)

    with flag_gems.use_gems():
        res_out = torch.scatter_reduce(base, 0, idx, src, reduce=reduce, include_self=False)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.scatter_reduce_
@pytest.mark.parametrize("reduce", REDUCE_OPS)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_scatter_reduce_inplace(reduce, dtype):
    shape = (8, 16)
    rows, cols = shape
    src = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    idx = torch.randint(0, rows, shape, device=flag_gems.device)
    base = torch.zeros(shape, dtype=dtype, device=flag_gems.device)

    ref_base = utils.to_reference(base.clone(), False)
    ref_src = utils.to_reference(src, False)
    ref_idx = utils.to_reference(idx, False)
    ref_base.scatter_reduce_(0, ref_idx, ref_src, reduce=reduce)

    with flag_gems.use_gems():
        base.scatter_reduce_(0, idx, src, reduce=reduce)

    utils.gems_assert_close(base, ref_base, dtype)
