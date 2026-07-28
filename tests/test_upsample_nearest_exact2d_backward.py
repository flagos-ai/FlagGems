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


@pytest.mark.upsample_nearest_exact2d_backward
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact2d_backward(shape, dtype):
    # Create input tensor and upsample it to get output size
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    # Forward pass to get output size
    out_h = shape[2] * 2
    out_w = shape[3] * 2
    output_size = (out_h, out_w)

    # Compute reference forward output using torch.ops.aten
    ref_out = torch.ops.aten._upsample_nearest_exact2d(
        ref_x, [out_h, out_w], None, None
    )
    # Create grad_output from forward output (using ones for gradient).
    # Keep on CPU for reference impl, move to GPU for GEMS kernel.
    grad_output = torch.ones_like(ref_out)

    # Compute backward
    input_size = tuple(x.shape)  # (N, C, H, W)

    ref_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward.default(
        grad_output, output_size, input_size
    )

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward.default(
            grad_output.to(flag_gems.device), output_size, input_size
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)


@pytest.mark.upsample_nearest_exact2d_backward
@pytest.mark.parametrize("shape", utils.UPSAMPLE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact2d_backward_with_scales(shape, dtype):
    # Test with explicit scales
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    # Use scales instead of output_size
    scale_h = 2.0
    scale_w = 2.0
    out_h = int(shape[2] * scale_h)
    out_w = int(shape[3] * scale_w)
    output_size = (out_h, out_w)

    # Compute reference forward output using torch.ops.aten
    ref_out = torch.ops.aten._upsample_nearest_exact2d(ref_x, None, [scale_h, scale_w])
    # Create grad_output from forward output (using ones for gradient).
    # Keep on CPU for reference impl, move to GPU for GEMS kernel.
    grad_output = torch.ones_like(ref_out)

    input_size = tuple(x.shape)

    ref_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward.default(
        grad_output, output_size, input_size, scale_h, scale_w
    )

    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten._upsample_nearest_exact2d_backward.default(
            grad_output.to(flag_gems.device), output_size, input_size, scale_h, scale_w
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
