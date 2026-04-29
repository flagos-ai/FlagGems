import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

MEDIAN_SHAPES_DIMS = [
    ((8,), 0),
    ((8, 8), 0),
    ((8, 8), 1),
    ((4, 8, 16), 0),
    ((4, 8, 16), 1),
    ((4, 8, 16), 2),
    ((2, 3, 4, 5), 2),
]


@pytest.mark.median
@pytest.mark.parametrize("shape, dim", MEDIAN_SHAPES_DIMS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=dim)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("shape, dim", [((8, 16), 1), ((4, 8, 16), 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_keepdim(shape, dim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=dim, keepdim=True)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=dim, keepdim=True)

    assert res_out.values.shape == ref_out.values.shape
    utils.gems_assert_close(res_out.values, ref_out.values, dtype)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("n", [7, 8, 15, 16])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_odd_even_length(n, dtype):
    inp = torch.randn((n,), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=0)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_large(dtype):
    inp = torch.randn((64, 1024), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=1)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)


@pytest.mark.median
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_median_noncontiguous(dtype):
    base = torch.randn((16, 32), dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.median(ref_inp, dim=0)
    with flag_gems.use_gems():
        res_out = torch.median(inp, dim=0)

    utils.gems_assert_close(res_out.values, ref_out.values, dtype)
    utils.gems_assert_equal(res_out.indices, ref_out.indices)
