import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Shapes for max_unpool2d: (N, C, H, W) input sizes that are valid for 2x2 pooling
MAX_UNPOOL2D_SHAPES = [
    (1, 1, 4, 4),
    (1, 1, 8, 8),
    (2, 3, 8, 8),
    (1, 16, 16, 16),
    (4, 8, 16, 16),
]


@pytest.mark.max_unpool2d
@pytest.mark.parametrize("shape", MAX_UNPOOL2D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_unpool2d(shape, dtype):
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Apply max_pool2d to get pooled output and indices
    pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
    ref_pooled, ref_indices = pool(ref_inp.float().contiguous())
    pooled, indices = pool(inp.contiguous())

    # Get output_size for unpooling
    output_size = [inp.shape[2], inp.shape[3]]

    # Reference unpool via aten - indices must be int64
    ref_out = torch.ops.aten.max_unpool2d(
        ref_pooled, ref_indices.to(torch.int64), output_size
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.max_unpool2d(
            pooled, indices.to(torch.int64), output_size
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
