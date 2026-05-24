import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .accuracy_utils import gems_assert_close, to_reference

FLOAT_DTYPES = utils.FLOAT_DTYPES

# Test for permute_copy
PERMUTE_COPY_SHAPES = [(2, 3, 4), (4, 2, 3), (3, 4, 2), (16, 128, 64), (2, 19, 7)]
PERMUTE_COPY_PERMS = [
    (0, 1, 2),  # identity
    (2, 0, 1),  # full permutation
    (1, 2, 0),  # another permutation
    (0, 2, 1),  # partial permutation
    (2, 1, 0),  # reverse
]


@pytest.mark.permute_copy
@pytest.mark.parametrize("shape", PERMUTE_COPY_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dims", PERMUTE_COPY_PERMS)
def test_permute_copy(shape, dtype, dims):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.permute_copy(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.permute_copy(inp, dims)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.permute_copy
@pytest.mark.parametrize("shape", [(2, 3, 4), (4, 2, 3)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_permute_copy_1d(shape, dtype):
    # Test 1D case
    shape_1d = (shape[0] * shape[1] * shape[2],)
    inp = torch.randn(shape_1d, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.permute_copy(ref_inp, (0,))
    with flag_gems.use_gems():
        res_out = torch.permute_copy(inp, (0,))

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.permute_copy
@pytest.mark.parametrize("shape", [(2, 3), (3, 4), (16, 128)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("dims", [(0, 1), (1, 0)])
def test_permute_copy_2d(shape, dtype, dims):
    # Test 2D case
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.permute_copy(ref_inp, dims)
    with flag_gems.use_gems():
        res_out = torch.permute_copy(inp, dims)

    gems_assert_close(res_out, ref_out, dtype)
