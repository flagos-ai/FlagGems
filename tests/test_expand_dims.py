import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes covering 1D to 4D tensors, including edge cases like all-ones and singleton dims.
EXPAND_DIMS_SHAPES = [
    (3,),
    (3, 4),
    (3, 4, 5),
    (1, 2, 3, 4),
    (1, 1, 1),
]
EXPAND_DIMS_DIMS = [0, 1, 2, 3, -1, -2]


@pytest.mark.expand_dims
@pytest.mark.parametrize("shape", EXPAND_DIMS_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", EXPAND_DIMS_DIMS)
def test_expand_dims(shape, dtype, dim):
    """Test expand_dims (unsqueeze) accuracy against PyTorch reference."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Normalize dim for reference
    ndim = len(shape)
    dim_normalized = dim if dim >= 0 else dim + ndim + 1
    if dim_normalized < 0 or dim_normalized > ndim:
        # Skip invalid dim
        return

    ref_out = torch.unsqueeze(ref_inp, dim_normalized)
    with flag_gems.use_gems():
        res_out = torch.unsqueeze(inp, dim_normalized)

    utils.gems_assert_equal(res_out, ref_out)
