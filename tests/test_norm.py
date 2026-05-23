import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    gems_assert_close,
    to_reference,
)

DIMS_LIST = [0, 1, [0, 1], [1, 0]]
KEEPDIM_DIMS = list(zip([True, False] * 2, DIMS_LIST))


@pytest.mark.norm
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_norm(shape, ord, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.norm(ref_inp, ord)
    with flag_gems.use_gems():
        res_out = torch.norm(inp, ord)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.norm_scalar
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_norm_scalar(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.norm(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.norm(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.norm_scalaropt_dim
@pytest.mark.parametrize("shape", REDUCTION_SHAPES)
@pytest.mark.parametrize("ord", [2, float("inf"), -float("inf"), 0, 1])
@pytest.mark.parametrize("keepdim, dim", KEEPDIM_DIMS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_norm_scalaropt_dim(shape, ord, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp, True)

    ref_out = torch.norm(ref_inp, ord, dim, keepdim)
    with flag_gems.use_gems():
        res_out = torch.norm(inp, ord, dim, keepdim)

    gems_assert_close(res_out, ref_out, dtype)
