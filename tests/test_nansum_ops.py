import math

import pytest
import torch

import flag_gems

from .accuracy_utils import (
    ALL_FLOAT_DTYPES,
    ALL_INT_DTYPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    REDUCTION_SHAPES,
    REDUCTION_SMALL_SHAPES,
    SkipVersion,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

FLOAT_DTYPES = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIM_LIST = [1] if QUICK_MODE else [0, 1]
DIMS_LIST = [1] if QUICK_MODE else [0, 1, [0, 1], [1, 0]]
KIND_KEEPDIM_DIMS_SHAPE = (
    [("normal", True, DIMS_LIST[0], REDUCTION_SHAPES[0])]
    if QUICK_MODE
    else list(
        zip(
            ["normal", "allTrue"] * 2,
            [True, False] * 2,
            DIMS_LIST,
            REDUCTION_SHAPES + [(7, 4, 11, 1)],
        )
    )
)
KEEPDIM_DIMS = (
    [(True, DIMS_LIST[0])] if QUICK_MODE else list(zip([True, False] * 2, DIMS_LIST))
)
KEEPDIM_DIM = (
    [(True, DIM_LIST[0])] if QUICK_MODE else list(zip([True, False], DIM_LIST))
)


def _inject_nans_(t: torch.Tensor, ratio: float = 0.2) -> torch.Tensor:
    if not t.is_floating_point():
        return t
    if t.numel() == 0:
        return t
    mask = torch.rand_like(t) < ratio
    t[mask] = float("nan")
    return t


@pytest.mark.nansum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_nansum_without_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _inject_nans_(inp)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nansum(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nansum(inp)

    non_nan = (~torch.isnan(ref_inp)).sum().item()
    gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(1, int(non_nan)))


@pytest.mark.nansum
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIM + [(False, []), (True, [])])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_nansum_dim(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    _inject_nans_(inp)
    ref_inp = to_reference(inp, True)

    ref_out = torch.nansum(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.nansum(inp, dim=dim, keepdim=keepdim)

    mask = ~torch.isnan(ref_inp)
    if isinstance(dim, int):
        dims = (dim,)
    elif dim == []:
        dims = None
    else:
        dims = tuple(dim)

    if dims is None:
        reduce_dim = int(mask.sum().item())
    else:
        dims = tuple(d % inp.ndim for d in dims)
        cnt = mask.sum(dim=dims, keepdim=keepdim)
        reduce_dim = int(cnt.max().item()) if cnt.numel() else 0

    gems_assert_close(res_out, ref_out, dtype, reduce_dim=max(1, reduce_dim))