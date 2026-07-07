import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# unsqueeze tests
UNSQUEEZE_DIMS = [0, 1, 2, -1, -2]


@pytest.mark.unsqueeze
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", UNSQUEEZE_DIMS)
def test_unsqueeze(shape, dtype, dim):
    # Adjust dim to be valid for the shape
    ndim = len(shape)
    valid_dim = dim if dim >= 0 else ndim + dim + 1
    if valid_dim < 0 or valid_dim > ndim:
        pytest.skip("Invalid dim for shape")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unsqueeze(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.unsqueeze(inp, dim)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.unsqueeze_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", UNSQUEEZE_DIMS)
def test_unsqueeze_(shape, dtype, dim):
    # Adjust dim to be valid for the shape
    ndim = len(shape)
    valid_dim = dim if dim >= 0 else ndim + dim + 1
    if valid_dim < 0 or valid_dim > ndim:
        pytest.skip("Invalid dim for shape")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.unsqueeze_(dim)
    with flag_gems.use_gems():
        res_out = inp.unsqueeze_(dim)

    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, ref_inp)
