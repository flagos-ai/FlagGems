import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

FLIP_DIMS = [(0,), (-2,), (2,), (0, 2), (2, 1), (0, -1, 1)]


def get_max_ndim(shape, dims):
    max_ndim = max(len(shape), len(dims))
    for dim in dims:
        dim = dim + 1 if dim >= 0 else -dim
        if dim > max_ndim:
            max_ndim = dim

    return max_ndim


@pytest.mark.flip
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dims", FLIP_DIMS)
def test_flip(shape, dtype, dims):
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)
    max_ndim = get_max_ndim(shape, dims)
    inp = utils.unsqueeze_tensor(inp, max_ndim)
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.flip
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES)
@pytest.mark.parametrize("dims", FLIP_DIMS)
def test_flip_with_non_dense_input(shape, dtype, dims):
    max_ndim = get_max_ndim(shape, dims)
    shape = utils.unsqueeze_tuple(shape, max(max_ndim, 2))

    shape_dialted = tuple(item * 2 for item in shape)
    if dtype in utils.ALL_FLOAT_DTYPES:
        inp = torch.randn(shape_dialted, dtype=dtype, device=flag_gems.device)[::2, ::2]
    else:
        inp = torch.randint(-1000, 1000, shape_dialted, device=flag_gems.device).to(
            dtype
        )[::2, ::2]
    ref_inp = utils.to_reference(inp, False)

    with flag_gems.use_gems():
        res_out = torch.flip(inp, dims)
    ref_out = torch.flip(ref_inp, dims)
    utils.gems_assert_equal(res_out, ref_out)
