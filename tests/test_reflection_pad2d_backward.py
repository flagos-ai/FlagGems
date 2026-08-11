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

# 4D shapes covering small to medium spatial sizes; pad2d operates on last two dims
if cfg.QUICK_MODE:
    # Quick-mode subset: one minimal shape + one small shape to smoke-test the kernel.
    REFLECTION_PAD2D_BACKWARD_SHAPES = [(1, 1, 3, 3), (2, 4, 16, 16)]
    # Two paddings covering symmetric and asymmetric cases in quick mode.
    REFLECTION_PAD2D_BACKWARD_PADDINGS = [[1, 1, 1, 1], [1, 2, 3, 4]]
else:
    # Full set: covers 1×1 spatial to 128×128 spatial sizes, batch sizes 1-4,
    # and channel counts 1-32, matching the shapes used by reflection_pad2d.
    REFLECTION_PAD2D_BACKWARD_SHAPES = [
        (1, 1, 3, 3),
        (1, 3, 8, 8),
        (2, 4, 16, 16),
        (4, 8, 32, 32),
        (2, 16, 64, 64),
        (1, 32, 128, 128),
    ]
    # Paddings chosen to cover: all-ones, symmetric, asymmetric, large uniform,
    # and asymmetric-with-large-bottom cases; all < min(H, W) of the smallest shape.
    REFLECTION_PAD2D_BACKWARD_PADDINGS = [
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [1, 2, 3, 4],
        [5, 5, 5, 5],
        [3, 4, 2, 5],
    ]


@pytest.mark.reflection_pad2d_backward
@pytest.mark.parametrize("shape", REFLECTION_PAD2D_BACKWARD_SHAPES)
@pytest.mark.parametrize("padding", REFLECTION_PAD2D_BACKWARD_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad2d_backward(shape, padding, dtype):
    # Ensure padding doesn't exceed input dimensions
    _, _, h, w = shape
    pad_left, pad_right, pad_top, pad_bottom = padding
    if pad_left >= w or pad_right >= w or pad_top >= h or pad_bottom >= h:
        pytest.skip("Padding exceeds input dimensions")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    # Create padded output and gradient
    padded = torch.nn.functional.pad(inp, padding, mode="reflect")
    grad_output = torch.randn_like(padded)
    ref_grad_output = utils.to_reference(grad_output, upcast=True)

    # Reference backward (computed in float64 for higher precision)
    ref_out = torch.ops.aten.reflection_pad2d_backward(
        ref_grad_output, ref_inp, padding
    )

    # FlagGems backward
    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad2d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.reflection_pad2d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_reflection_pad2d_backward_3d(dtype):
    """Test with 3D input (C, H, W) instead of 4D (N, C, H, W)"""
    # 3D input shape (C, H, W) - no batch dimension; padding is moderate
    shape = (3, 8, 8)
    padding = [2, 2, 2, 2]

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, upcast=True)

    padded = torch.nn.functional.pad(inp, padding, mode="reflect")
    grad_output = torch.randn_like(padded)
    ref_grad_output = utils.to_reference(grad_output, upcast=True)

    # Reference backward (computed in float64 for higher precision)
    ref_out = torch.ops.aten.reflection_pad2d_backward(
        ref_grad_output, ref_inp, padding
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.reflection_pad2d_backward(grad_output, inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)
