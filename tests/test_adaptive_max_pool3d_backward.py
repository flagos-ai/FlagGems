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

ADAPTIVE_MAX_POOL3D_OUTPUT_SIZES = [
    (1, 1, 1),
    (3, 3, 3),
    (4, 4, 4),
    (8, 8, 8),
]


# Adaptive max pool 3d backward test
ADAPTIVE_MAX_POOL3D_SHAPES = [
    (1, 1, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 32, 32, 32),
    (2, 8, 8, 8, 8),
    # Non-evenly-divisible shapes to catch race conditions in backward pass.
    (1, 1, 7, 7, 7),
    (1, 1, 10, 10, 10),
]


@pytest.mark.adaptive_max_pool3d_backward
@pytest.mark.parametrize("shape", ADAPTIVE_MAX_POOL3D_SHAPES)
@pytest.mark.parametrize("output_size", ADAPTIVE_MAX_POOL3D_OUTPUT_SIZES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_adaptive_max_pool3d_backward(shape, output_size, dtype):
    # Skip invalid combinations where output_size exceeds input spatial dims.
    input_spatial = shape[2:]
    if any(out > inp for out, inp in zip(output_size, input_spatial)):
        pytest.skip("output size larger than input spatial dims")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Compute forward pass to get indices
    ref_output, ref_indices = torch.nn.functional.adaptive_max_pool3d(
        ref_inp, output_size=output_size, return_indices=True
    )
    res_output, res_indices = torch.nn.functional.adaptive_max_pool3d(
        inp, output_size=output_size, return_indices=True
    )

    # Compute backward with gradient of ones
    ref_grad_output = torch.ones_like(ref_output)
    grad_output = torch.ones_like(res_output)

    ref_out = torch.ops.aten.adaptive_max_pool3d_backward(
        ref_grad_output, ref_inp, ref_indices
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.adaptive_max_pool3d_backward(
            grad_output, inp, res_indices
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
