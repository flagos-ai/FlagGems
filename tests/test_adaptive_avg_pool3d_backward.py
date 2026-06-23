import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES = [
    (4, 4, 4),
    (8, 8, 8),
    (3, 3, 3),
    (5, 5, 5),
]


# Define shapes for 3D adaptive average pooling
ADAPTIVE_AVG_POOL3D_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 7, 7, 7),
    (1, 2, 10, 10, 10),
]


@pytest.mark.adaptive_avg_pool3d_backward
@pytest.mark.parametrize("shape", ADAPTIVE_AVG_POOL3D_SHAPES)
@pytest.mark.parametrize("output_size", ADAPTIVE_AVG_POOL3D_OUTPUT_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_avg_pool3d_backward(shape, output_size, dtype):
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute forward pass to get output shape
    ref_out = torch.nn.functional.adaptive_avg_pool3d(ref_inp, output_size)
    torch.nn.functional.adaptive_avg_pool3d(inp, output_size)

    # Compute backward with gradient of ones
    grad_output = torch.ones_like(ref_out)

    # Reference implementation
    ref_grad = torch.ops.aten._adaptive_avg_pool3d_backward.default(
        grad_output, ref_inp
    )

    # GEMS implementation
    with flag_gems.use_gems():
        gems_grad = torch.ops.aten._adaptive_avg_pool3d_backward.default(
            grad_output.to(flag_gems.device), inp
        )

    utils.gems_assert_close(gems_grad, ref_grad, dtype)
