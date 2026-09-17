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


# Test for GatherBlockQuantized
# Since this operator doesn't exist in PyTorch, we test against a manual reference
# gather_block_quantized dequantizes int8 quantized data with float32 scales.
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.gather_block_quantized
@pytest.mark.parametrize("shape", [(1024,), (2048,), (4096,)])
@pytest.mark.parametrize("block_size", [128, 256])
def test_gather_block_quantized(shape, block_size, dtype):
    """Test gather_block_quantized operator."""
    # Create quantized data (int8) and scale factors
    n_elements = shape[0]
    n_blocks = (n_elements + block_size - 1) // block_size

    # Generate quantized data (stored as int8, values in range [-128, 127])
    quantized_data = torch.randint(
        -100, 100, shape, dtype=torch.int8, device=flag_gems.device
    )

    # Generate scale factors per block
    scales = (
        torch.rand(n_blocks, dtype=dtype, device=flag_gems.device) * 2 + 0.5
    )  # scales in [0.5, 2.5]

    # Test without indices (dequantize all)
    with flag_gems.use_gems():
        res_out = flag_gems.ops.gather_block_quantized(
            quantized_data, scales, None, block_size
        )

    # Reference implementation
    ref_quantized_data = utils.to_reference(quantized_data)
    ref_scales = utils.to_reference(scales)
    ref_out = torch.empty(
        n_elements, dtype=torch.float32, device=ref_quantized_data.device
    )
    for i in range(n_elements):
        block_idx = i // block_size
        if block_idx < n_blocks:
            ref_out[i] = ref_quantized_data[i].float() * ref_scales[block_idx].float()
        else:
            ref_out[i] = 0

    # Compare
    utils.gems_assert_close(res_out, ref_out, dtype, atol=1e-3)
