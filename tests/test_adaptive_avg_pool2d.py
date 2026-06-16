import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

ADAPTIVE_AVGPOOL2D_CONFIGS = [
    # Test various combinations of input and output sizes
    # Cases where output size is smaller than input
    ((4, 3, 32, 32), (1, 1)),  # Downsize to 1x1
    ((4, 3, 32, 32), (2, 2)),  # Downsize to 2x2
    ((4, 3, 32, 32), (8, 8)),  # Downsize to 8x8
    ((4, 3, 32, 32), (16, 16)),  # Downsize to 16x16
    ((2, 16, 56, 56), (7, 7)),  # ResNet-like case
    # Test non-square inputs and outputs
    ((8, 16, 28, 40), (14, 10)),  # Non-square input to non-square output
    ((4, 8, 60, 80), (15, 20)),  # Non-square input to smaller non-square output
    # Test 1D output size
    ((4, 3, 32, 32), 8),  # Same output size for both dimensions
    # Large case
    ((1, 64, 224, 224), (7, 7)),  # Typical image classification case
    # Edge cases
    ((2, 4, 10, 10), (1, 5)),  # Different scaling for different dimensions
    ((4, 2, 50, 100), (25, 25)),  # 2x down one dimension, 4x down other
]


@pytest.mark.adaptive_avg_pool2d
@pytest.mark.parametrize("shape, output_size", ADAPTIVE_AVGPOOL2D_CONFIGS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_adaptive_avg_pool2d_forward(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    ref_out = torch.ops.aten._adaptive_avg_pool2d(ref_inp, output_size)
    res_out = flag_gems.adaptive_avg_pool2d(inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)
