import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIMS = [1]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIMS = [0, 1, (0, 1)]

INCLUDE_0_SHAPES = [(1, 0, 128), (4096, 1, 0), (200, 0, 3)]


def _nan_input(shape, dtype, device, nan_ratio=0.3):
    """Create tensor with ~nan_ratio fraction of NaN values."""
    x = torch.randn(shape, dtype=dtype, device=device) * 10
    mask = torch.rand(shape, device=device) < nan_ratio
    x[mask] = float("nan")
    return x


@pytest.mark.nansum
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + INCLUDE_0_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nansum(shape, dtype):
    inp = _nan_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nansum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nansum(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=inp.numel())


@pytest.mark.nansum_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dim", DIMS)
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_nansum_dim(shape, dim, keepdim, dtype):
    if isinstance(dim, int) and dim >= len(shape):
        return

    inp = _nan_input(shape, dtype, flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.nansum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.nansum(inp, dim=dim, keepdim=keepdim)

    if isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)
    dims = [d % inp.ndim for d in dims]
    _dim = 1
    for d in dims:
        _dim *= shape[d]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=_dim)


@pytest.mark.nansum_edge
def test_nansum_edge():
    device = flag_gems.device

    # all NaN elements should produce 0
    x = torch.full((5, 5), float("nan"), device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nansum(ref_inp, dim=1)
    with flag_gems.use_gems():
        res = torch.nansum(x, dim=1)
    utils.gems_assert_close(res, ref_out, torch.float32, reduce_dim=5)

    # integer input with type promotion and conversion
    x = torch.tensor([1, 2, 3, 4], device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nansum(ref_inp)
    with flag_gems.use_gems():
        res = torch.nansum(x)
    utils.gems_assert_close(res, ref_out, torch.int64, reduce_dim=4)

    # empty tensor should return 0
    x = torch.empty(0, device=device)
    ref_inp = utils.to_reference(x, True)
    ref_out = torch.nansum(ref_inp)
    with flag_gems.use_gems():
        res = torch.nansum(x)
    utils.gems_assert_close(res, ref_out, torch.float32, reduce_dim=0)
