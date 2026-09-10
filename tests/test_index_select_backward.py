import random

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# index_select_backward tests
INDEX_SELECT_BACKWARD_SHAPES = [
    (3, 4),
    (5, 3),
    (2, 3, 4),
    (4,),
    (8, 16),
    (2, 8, 16),
]


@pytest.mark.index_select_backward
@pytest.mark.parametrize("shape", INDEX_SELECT_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_select_backward(shape, dtype):
    # Generate valid test case: need index_len < dim_size_out
    # Pick random dim (0 to ndim-1)
    dim = random.randint(0, len(shape) - 1)
    dim_size_out = shape[dim] + 2  # Make output larger than input
    index_len = shape[dim]

    # Generate random index
    index = torch.randint(0, dim_size_out, (index_len,), device=flag_gems.device)

    # Create grad tensor with the shape
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Build self_sizes
    self_sizes = list(shape)
    self_sizes[dim] = dim_size_out
    self_sizes = tuple(self_sizes)

    # Reference calculation using float32 accumulation for numerical stability
    ref_grad = utils.to_reference(grad).to(torch.float32)
    ref_index = utils.to_reference(index)
    ref_out = torch.zeros(self_sizes, dtype=torch.float32, device=ref_grad.device)

    # Manually compute scatter_add since the API requires matching dimensions
    if dim == 0:
        for i, idx in enumerate(ref_index.tolist()):
            ref_out[idx] += ref_grad[i]
    elif dim == 1:
        for i, idx in enumerate(ref_index.tolist()):
            ref_out[:, idx] += ref_grad[:, i]
    else:
        for i, idx in enumerate(ref_index.tolist()):
            ref_out[:, :, idx] += ref_grad[:, :, i]

    with flag_gems.use_gems():
        res_out = flag_gems.index_select_backward(grad, self_sizes, dim, index)

    utils.gems_assert_close(res_out, ref_out.to(dtype), dtype)


@pytest.mark.index_select_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_select_backward_1d(dtype):
    """Test 1D case specifically"""
    grad = torch.randn(4, dtype=dtype, device=flag_gems.device)
    self_sizes = (6,)
    dim = 0
    index = torch.tensor([0, 2, 4, 5], device=flag_gems.device)

    # Reference calculation using float32 accumulation for numerical stability
    ref_grad = utils.to_reference(grad).to(torch.float32)
    ref_index = utils.to_reference(index)
    ref_out = torch.zeros(self_sizes, dtype=torch.float32, device=ref_grad.device)

    for i, idx in enumerate(ref_index.tolist()):
        ref_out[idx] += ref_grad[i]

    with flag_gems.use_gems():
        res_out = flag_gems.index_select_backward(grad, self_sizes, dim, index)

    utils.gems_assert_close(res_out, ref_out.to(dtype), dtype)
