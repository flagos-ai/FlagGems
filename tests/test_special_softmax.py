import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.special_softmax
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
def test_special_softmax(shape, dtype, dim):
    # Test special_softmax with different dims
    if len(shape) == 1:
        dim = -1  # 1D tensors only support dim=-1

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.softmax(ref_inp, dim)
    with flag_gems.use_gems():
        res_out = torch.special.softmax(inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.special_softmax
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1])
@pytest.mark.parametrize(
    "dtype_arg", [None, torch.float32, torch.float16, torch.bfloat16]
)
def test_special_softmax_with_dtype(shape, dtype, dim, dtype_arg):
    # Test special_softmax with dtype parameter
    if len(shape) == 1:
        dim = -1  # 1D tensors only support dim=-1

    # Create input in the base dtype
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.special.softmax(ref_inp, dim, dtype=dtype_arg)
    with flag_gems.use_gems():
        res_out = torch.special.softmax(inp, dim, dtype=dtype_arg)

    # Determine expected output dtype
    expected_dtype = dtype_arg if dtype_arg is not None else dtype
    utils.gems_assert_close(res_out, ref_out, expected_dtype)
