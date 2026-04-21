import pytest
import torch

import flag_gems

from .accuracy_utils import FLOAT_DTYPES, gems_assert_close, to_reference

DIFF_SHAPES = [(1024,), (100, 200), (10, 20, 30), (16, 128, 64, 60)]


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_diff(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.diff(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.diff(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", DIFF_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test_accuracy_diff_dim(shape, dtype, dim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.diff(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, dim=dim)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", [(1024,), (100, 200)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("n", [1, 2, 3])
def test_accuracy_diff_n(shape, dtype, n):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.diff(ref_inp, n=n)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, n=n)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.diff
@pytest.mark.parametrize("shape", [(100,), (50, 60)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_diff_prepend_append(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    prepend = torch.randn(
        shape[:-1] + (3,) if len(shape) > 1 else (3,),
        dtype=dtype,
        device=flag_gems.device,
    )
    append = torch.randn(
        shape[:-1] + (2,) if len(shape) > 1 else (2,),
        dtype=dtype,
        device=flag_gems.device,
    )

    ref_inp = to_reference(inp)
    ref_prepend = to_reference(prepend)
    ref_append = to_reference(append)

    ref_out = torch.diff(ref_inp, prepend=ref_prepend)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, prepend=prepend)
    gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.diff(ref_inp, append=ref_append)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, append=append)
    gems_assert_close(res_out, ref_out, dtype)

    ref_out = torch.diff(ref_inp, prepend=ref_prepend, append=ref_append)
    with flag_gems.use_gems():
        res_out = torch.diff(inp, prepend=prepend, append=append)
    gems_assert_close(res_out, ref_out, dtype)
