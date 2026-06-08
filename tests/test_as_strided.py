import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.as_strided
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided(shape, dtype):
    """Test as_strided with various shapes and strides."""
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test case 1: basic as_strided with smaller size
    if len(shape) >= 2:
        size = (shape[0], min(shape[1] // 2 + 1, 1))
        stride = (shape[1], 1)
    elif len(shape) == 1:
        size = (max(shape[0] // 2 + 1, 1),)
        stride = (1,)
    else:
        size = (1,)
        stride = (1,)

    ref_out = torch.as_strided(ref_inp, size, stride)
    with flag_gems.use_gems():
        res_out = torch.as_strided(inp, size, stride)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.as_strided
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided_with_offset(shape, dtype):
    """Test as_strided with storage_offset."""
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test case: as_strided with storage_offset
    if len(shape) >= 2:
        size = (shape[0], min(shape[1] // 2 + 1, 1))
        stride = (shape[1], 1)
        storage_offset = shape[1] // 2 if shape[1] > 1 else 0
    elif len(shape) == 1:
        size = (max(shape[0] // 2 + 1, 1),)
        stride = (1,)
        storage_offset = shape[0] // 2 if shape[0] > 1 else 0
    else:
        size = (1,)
        stride = (1,)
        storage_offset = 0

    ref_out = torch.as_strided(ref_inp, size, stride, storage_offset)
    with flag_gems.use_gems():
        res_out = torch.as_strided(inp, size, stride, storage_offset)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.as_strided_
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_as_strided_(shape, dtype):
    """Test as_strided_ in-place operation."""
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    # Test case: as_strided_ with smaller size
    if len(shape) >= 2:
        size = (shape[0], min(shape[1] // 2 + 1, 1))
        stride = (shape[1], 1)
    elif len(shape) == 1:
        size = (max(shape[0] // 2 + 1, 1),)
        stride = (1,)
    else:
        size = (1,)
        stride = (1,)

    ref_out = ref_inp.as_strided_(size, stride)
    with flag_gems.use_gems():
        res_out = inp.as_strided_(size, stride)

    utils.gems_assert_equal(res_out, ref_out)
