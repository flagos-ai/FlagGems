import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# QUICK_MODE limits to a single float32 dtype to avoid CI timeout
FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES
DIM_LIST = [1] if QUICK_MODE else [0, 1]
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS_SHAPE = (
    [(True, DIMS_LIST[0], utils.REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(
        zip([True, False] * 2, DIMS_LIST, utils.REDUCTION_SHAPES + [(7, 4, 11, 1)])
    )
)


@pytest.mark.amin
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_amin(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amin(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amin_
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_amin_(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amin(ref_inp, dim=dim, keepdim=True)
    ref_out = ref_out.expand_as(inp)
    with flag_gems.use_gems():
        res_out = torch.amin(inp, dim=dim, keepdim=True)
        res_out = res_out.expand_as(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.amin
@pytest.mark.parametrize(
    "shape, dim",
    [
        ((4, 8, 4096), 1),  # non-inner: K = 4096 spans multiple K tiles
        ((4, 4096, 8), 1),  # non-inner: N = 4096 exercises the reduction loop
        ((8, 4096), 1),  # inner: N = 4096 exercises the reduction loop
        ((4096, 8), 0),  # non-inner via the outer dim
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_amin_dim_multi_tile(shape, dim, keepdim):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.amin(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)
