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

# 3D volumes of varying sizes for testing reflection padding backward
REFLECTION_PAD3D_SHAPES = [
    (1, 1, 4, 4, 4),
    (2, 3, 8, 8, 8),
    (1, 1, 16, 16, 16),
    (2, 4, 8, 16, 32),
]

# Padding values must be strictly less than corresponding dimension size
REFLECTION_PAD3D_PADDINGS = [
    (1, 1, 1, 1, 1, 1),
    (2, 2, 2, 2, 2, 2),
    (1, 2, 1, 2, 1, 2),
    (0, 0, 0, 0, 0, 0),
]


@pytest.mark.reflection_pad3d_backward
@pytest.mark.parametrize("shape", REFLECTION_PAD3D_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("padding", REFLECTION_PAD3D_PADDINGS)
def test_reflection_pad3d_backward(shape, dtype, padding):
    # Create input tensor on GPU
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Compute padded shape: (N, C, D + pad_d0 + pad_d1, H + pad_h0 + pad_h1, W + pad_w0 + pad_w1)
    # padding is (pad_d0, pad_d1, pad_h0, pad_h1, pad_w0, pad_w1)
    # shape is (N, C, D, H, W) where N=0, C=1, D=2, H=3, W=4
    padded_shape = (
        shape[0],  # N
        shape[1],  # C
        shape[2] + padding[0] + padding[1],  # D
        shape[3] + padding[2] + padding[3],  # H
        shape[4] + padding[4] + padding[5],  # W
    )

    # Create gradient tensor with ones (simulating gradient of loss w.r.t. padded output)
    grad_output = torch.ones(padded_shape, dtype=dtype, device=flag_gems.device)

    # Reference backward
    ref_x = utils.to_reference(x)
    ref_grad = utils.to_reference(grad_output)
    ref_out = torch.ops.aten.reflection_pad3d_backward(ref_grad, ref_x, padding)

    # Gems backward on GPU
    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad3d_backward(grad_output, x, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)
