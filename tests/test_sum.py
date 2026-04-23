import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]


@pytest.mark.sum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.sum(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


INCLUDE_0_SHAPES = [(1, 0, 128, 512), (4096, 1, 256, 0), (200, 10, 0, 3)]


@pytest.mark.sum_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + INCLUDE_0_SHAPES)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize(
    "dim",
    [
        0,
        1,
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.sum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.sum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.sum_dim_out
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_sum_dim_out(shape, dim, keepdim, dtype):
    # Regression test: sum_dim_out must resize external out tensor and skip squeeze.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_result = torch.sum(ref_inp, dim=dim, keepdim=keepdim)

    # Pre-allocate out tensor with wrong shape to test resize logic
    out = torch.empty((1,), dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_result = torch.sum(inp, dim=dim, keepdim=keepdim, out=out)

    if isinstance(dim, int):
        dim = [dim]
    dim = [d % inp.ndim for d in dim]
    _dim = 1
    for d in dim:
        _dim *= shape[d]
    if dim == []:
        _dim = inp.numel()
    utils.gems_assert_close(res_result, ref_result, dtype, reduce_dim=_dim)
    utils.gems_assert_close(out, ref_result, dtype, reduce_dim=_dim)
