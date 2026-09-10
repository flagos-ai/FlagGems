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

# Define shapes for linalg_slogdet (square matrices)
SLOGDET_SHAPES = [(2, 3, 3), (4, 4), (8, 8), (16, 16), (32, 32)]


@pytest.mark.linalg_slogdet
@pytest.mark.parametrize("shape", SLOGDET_SHAPES)
# linalg.slogdet generated kernel only supports float32 on CUDA.
@pytest.mark.parametrize("dtype", [torch.float32])
def test_linalg_slogdet(shape, dtype):
    """Test linalg_slogdet accuracy against PyTorch reference."""
    # Ensure we have a square matrix
    assert len(shape) >= 2 and shape[-1] == shape[-2], "Input must be square matrix"

    # Create input tensor
    A = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_A = utils.to_reference(A)

    # Compute reference
    ref_out = torch.linalg.slogdet(ref_A)

    # Compute with FlagGems
    with flag_gems.use_gems():
        res_out = torch.linalg.slogdet(A)

    # Compare sign
    utils.gems_assert_close(res_out.sign, ref_out.sign, dtype)

    # Compare logabsdet (more tolerant for floating point)
    utils.gems_assert_close(
        res_out.logabsdet, ref_out.logabsdet, dtype, reduce_dim=shape[-1]
    )
