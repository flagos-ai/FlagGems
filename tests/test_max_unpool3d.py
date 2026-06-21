import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

# QUICK_MODE uses float32 only for faster CI validation
FLOAT_DTYPES = [torch.float32] if QUICK_MODE else utils.FLOAT_DTYPES

# 3D shapes (N, C, D, H, W) with all spatial dims even,
# needed by max_unpool3d with kernel_size=2, stride=2
UPSAMPLE_SHAPES_3D = [
    (4, 8, 32, 32, 32),
    (3, 5, 16, 18, 22),
    (2, 16, 8, 64, 64),
    (12, 24, 16, 16, 16),
    (1, 2, 62, 64, 66),
]


@pytest.mark.max_unpool3d
@pytest.mark.parametrize("shape", UPSAMPLE_SHAPES_3D)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_unpool3d(shape, dtype):
    # shape is (N, C, D, H, W) for the original input before pooling
    # After max_pool3d with kernel_size=2, stride=2, pooled shape is (N, C, D//2, H//2, W//2)
    N, C, D, H, W = shape

    # First do max pooling to get pooled input and indices
    input_orig = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    pooled, indices = pool(input_orig)

    # Reference: use PyTorch's max_unpool3d
    ref_input = utils.to_reference(pooled)
    ref_indices = utils.to_reference(indices)
    ref_out = torch.nn.functional.max_unpool3d(
        ref_input, ref_indices, kernel_size=2, stride=2
    )

    # GEMS implementation
    with flag_gems.use_gems():
        res_out = flag_gems.max_unpool3d(pooled, indices)

    utils.gems_assert_close(res_out, ref_out, dtype)
