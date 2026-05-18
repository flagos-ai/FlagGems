import pytest
import torch

import flag_gems
from flag_gems.ops import median, median_dim

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    INT_DTYPES = [torch.int32]
    SHAPES_1D = [7, 64]
    SHAPES_DIM = [((4, 7), 1), ((3, 4, 5), 2)]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    INT_DTYPES = [torch.int32, torch.int64]
    SHAPES_1D = [1, 3, 7, 8, 64, 255, 256, 1023, 1024]
    SHAPES_DIM = [
        ((7,), 0),
        ((64,), 0),
        ((4, 7), 0),
        ((4, 7), 1),
        ((8, 64), 0),
        ((8, 64), 1),
        ((64, 64), 0),
        ((64, 64), 1),
        ((256, 256), 0),
        ((256, 256), 1),
        ((3, 4, 5), 0),
        ((3, 4, 5), 1),
        ((3, 4, 5), 2),
        ((8, 16, 32), 2),
        ((2, 3, 4, 5), 0),
        ((2, 3, 4, 5), 1),
        ((2, 3, 4, 5), 3),
        ((4, 7), -1),
        ((3, 4, 5), -1),
        ((3, 4, 5), -2),
    ]

ALL_DTYPES = FLOAT_DTYPES + INT_DTYPES


def make_tensor(shape, dtype):
    if dtype in INT_DTYPES:
        return torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
    return torch.randn(shape, dtype=dtype, device=flag_gems.device)


@pytest.mark.median
@pytest.mark.parametrize("n", SHAPES_1D)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_scalar(dtype, n):
    inp = make_tensor((n,), dtype)
    ref = torch.median(utils.to_reference(inp))
    res = median(inp)
    utils.gems_assert_equal(res, ref)


@pytest.mark.median
@pytest.mark.parametrize("shape", [(4, 64), (8, 8), (3, 4, 5), (2, 3, 4, 5)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_scalar_multidim(dtype, shape):
    inp = make_tensor(shape, dtype)
    ref = torch.median(utils.to_reference(inp))
    res = median(inp)
    utils.gems_assert_equal(res, ref)


@pytest.mark.median_dim
@pytest.mark.parametrize("shape,dim", SHAPES_DIM)
@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_median_dim(dtype, shape, dim):
    inp = make_tensor(shape, dtype)
    ref = torch.median(utils.to_reference(inp), dim)
    res = median_dim(inp, dim)
    if dtype in INT_DTYPES:
        utils.gems_assert_equal(res.values, ref.values)
    else:
        utils.gems_assert_close(res.values, ref.values, dtype)


@pytest.mark.median_dim
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((4, 7), 1),
        ((3, 4, 5), 2),
        ((2, 3, 4, 5), -1),
    ],
)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_keepdim(dtype, shape, dim):
    inp = make_tensor(shape, dtype)
    ref = torch.median(utils.to_reference(inp), dim, keepdim=True)
    res = median_dim(inp, dim, keepdim=True)
    utils.gems_assert_equal(res.values, ref.values)
    assert res.values.shape == ref.values.shape


@pytest.mark.median_dim
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_median_dim_special(dtype):
    # all same values
    inp = torch.ones(3, 7, dtype=dtype, device=flag_gems.device)
    ref = torch.median(utils.to_reference(inp), dim=1)
    res = median_dim(inp, dim=1)
    utils.gems_assert_equal(res.values, ref.values)

    # size-1 reduction dim
    inp = make_tensor((4, 1, 5), dtype)
    ref = torch.median(utils.to_reference(inp), dim=1)
    res = median_dim(inp, dim=1)
    utils.gems_assert_equal(res.values, ref.values)
    utils.gems_assert_equal(res.indices, ref.indices)


@pytest.mark.median
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1024, 1024), 1),
        ((4096, 128), 1),
        ((128, 4096), 1),
    ],
)
def test_median_large(shape, dim):
    inp = make_tensor(shape, torch.float32)
    ref = torch.median(utils.to_reference(inp), dim)
    res = median_dim(inp, dim)
    utils.gems_assert_equal(res.values, ref.values)
