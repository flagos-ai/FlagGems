import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.unflatten
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [-1, 0, 1, 2])
def test_unflatten(shape, dtype, dim):
    """Test unflatten accuracy against PyTorch implementation."""
    # Normalize negative dim
    ndim = len(shape)
    if dim < 0:
        dim = dim % ndim

    # Skip if dim is out of range
    if dim >= ndim:
        pytest.skip(f"dim {dim} out of range for shape {shape}")

    dim_size = shape[dim]

    # Find a valid factorization for sizes
    # Try to split into 2 factors
    factor = None
    for f in range(2, dim_size + 1):
        if dim_size % f == 0:
            factor = f
            break

    if factor is None or dim_size < 2:
        pytest.skip("Cannot find valid unflatten sizes for this shape")

    sizes = (factor, dim_size // factor)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.unflatten(ref_inp, dim, sizes)
    with flag_gems.use_gems():
        res_out = torch.unflatten(inp, dim, sizes)

    res_out_ref = utils.to_reference(res_out)
    ref_out_ref = utils.to_reference(ref_out)
    utils.gems_assert_close(res_out_ref, ref_out_ref, dtype)
