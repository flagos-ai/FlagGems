import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, INT_DTYPES, gems_assert_close, to_reference
from .conftest import QUICK_MODE

DIFF_SHAPES = [(1024,), (100, 200), (10, 20, 30), (16, 128, 64, 60)]
DIFF_3D_SHAPES = [(8, 16, 32), (32, 64, 100)]

if QUICK_MODE:
    DIFF_SHAPES = DIFF_SHAPES[:2]
    DIFF_3D_SHAPES = DIFF_3D_SHAPES[:1]


def _make_input(shape, dtype, device):
    if dtype in INT_DTYPES:
        return torch.randint(-100, 100, shape, dtype=dtype, device=device)
    return torch.randn(shape, dtype=dtype, device=device)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("n", [1, 2, 5])
def test_diff(shape, dtype, n):
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp, n=n)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, n=n)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_3D_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test_diff_3d(shape, dim, dtype):
    inp = _make_input(shape, dtype, flag_gems.device)
    ref_inp = to_reference(inp)
    ref_out = torch.diff(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, dim=dim)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + INT_DTYPES)
def test_diff_prepend_append(shape, dtype):
    inp = _make_input(shape, dtype, flag_gems.device)
    prepend = _make_input(
        shape[:-1] + (3,) if len(shape) > 1 else (3,),
        dtype,
        flag_gems.device,
    )
    append = _make_input(
        shape[:-1] + (2,) if len(shape) > 1 else (2,),
        dtype,
        flag_gems.device,
    )
    ref_inp = to_reference(inp)
    ref_prepend = to_reference(prepend)
    ref_append = to_reference(append)

    ref_out = torch.diff(ref_inp, prepend=ref_prepend, append=ref_append)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, prepend=prepend, append=append)

    gems_assert_close(res_out, ref_out, dtype)
