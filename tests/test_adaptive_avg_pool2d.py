# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_accuracy_adaptive_avg_pool2d_forward(shape, output_size, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    if isinstance(output_size, int):
        output_size = [output_size, output_size]

    ref_out = torch.ops.aten._adaptive_avg_pool2d(ref_inp, output_size)
    res_out = flag_gems.adaptive_avg_pool2d(inp, output_size)

    utils.gems_assert_close(res_out, ref_out, dtype)
