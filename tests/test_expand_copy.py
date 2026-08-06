import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.expand_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_copy(shape, dtype):
    # Test expand_copy which broadcasts input to a larger shape
    # expand_copy takes a tensor and expands it to a target shape using broadcasting
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Determine target shape based on input shape dimensionality
    # For scalar (), expand to (2, 3)
    # For 1D (n,), expand to (2, n) or (n*2,) if n > 1
    # For 2D+, expand any dimension of size 1 to size 2
    ndim = len(shape)
    if ndim == 0:
        # Scalar tensor - expand to 2D
        target_shape = (2, 3)
    elif ndim == 1:
        if shape[0] == 1:
            # (1,) -> (2, 3)
            target_shape = (2, 3)
        elif shape[0] > 1:
            # (n,) -> (2, n)
            target_shape = (2, shape[0])
        else:
            target_shape = shape
    else:
        # For multi-dimensional, expand first dimension of size 1, or last dim if > 1
        target_shape = list(shape)
        # Find a dimension we can expand (size 1 or > 1 with singleton neighbors)
        if target_shape[0] == 1:
            target_shape[0] = 2
        elif target_shape[-1] == 1:
            target_shape[-1] = 2
        elif target_shape[0] > 1:
            target_shape = [2] + target_shape
        else:
            target_shape[-1] = target_shape[-1] * 2 if target_shape[-1] > 1 else 4
        target_shape = tuple(target_shape)

    ref_out = torch.ops.aten.expand_copy(ref_inp, target_shape)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.expand_copy(inp, target_shape)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.expand_copy
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_expand_copy_same_shape(shape, dtype):
    # Test expand_copy when target shape equals input shape
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.expand_copy(ref_inp, shape)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.expand_copy(inp, shape)

    utils.gems_assert_equal(res_out, ref_out)
