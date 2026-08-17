import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Test shapes for max_unpool3d (similar to pool3d output shapes)
MAX_UNPOOL3D_SHAPES = [
    (1, 1, 4, 4, 4),
    (2, 3, 8, 8, 8),
    (1, 1, 8, 8, 8),
    (1, 2, 16, 16, 16),
    (4, 4, 8, 8, 8),
]


@pytest.mark.max_unpool3d
@pytest.mark.parametrize("shape", MAX_UNPOOL3D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_unpool3d(shape, dtype):
    # Generate input for max_pool3d first, then unpool
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    output, indices = pool(input)

    ref_input = utils.to_reference(input)
    ref_pool = torch.nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    ref_output, ref_indices = ref_pool(ref_input)

    # Use flag_gems.max_unpool3d directly with output_size
    ref_out = torch.nn.functional.max_unpool3d(
        ref_output, ref_indices, kernel_size=2, stride=2, output_size=shape
    )
    with flag_gems.use_gems():
        res_out = flag_gems.max_unpool3d(
            output,
            indices,
            kernel_size=2,
            stride=2,
            output_size=shape[2:],  # Only D, H, W
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
