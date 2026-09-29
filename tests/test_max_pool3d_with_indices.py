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

# Representative 5-D pooling configs: (shape, kernel_size, stride, padding,
# dilation, ceil_mode). Covers cubic/non-cubic kernels, strides,
# symmetric/asymmetric padding, dilation, ceil_mode and a batch > 1 shape.
MAXPOOL3D_WITH_INDICES_CONFIGS = [
    # Classic 3x3x3 kernel, stride 2, padding 1
    ((4, 3, 16, 16, 16), 3, 2, 1, 1, False),
    # Non-cubic kernel and stride
    ((8, 16, 12, 14, 14), (2, 3, 3), (1, 2, 2), (0, 1, 1), 1, False),
    # ceil_mode
    ((2, 4, 15, 15, 15), 3, 2, 1, 1, True),
    # dilation
    ((1, 1, 9, 9, 9), 2, 1, 0, 2, False),
    # Typical 3D CNN shape
    ((1, 64, 8, 28, 28), 3, 2, 1, 1, False),
    # No padding
    ((2, 8, 8, 16, 16), 2, 2, 0, 1, False),
    # Non-symmetric padding
    ((2, 8, 10, 16, 20), 2, 2, (0, 1, 0), 1, False),
    # Small input
    ((1, 1, 5, 5, 5), 2, 1, 0, 1, False),
]


@pytest.mark.max_pool3d_with_indices
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation, ceil_mode",
    MAXPOOL3D_WITH_INDICES_CONFIGS,
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d_with_indices(
    shape, kernel_size, stride, padding, dilation, ceil_mode, dtype
):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_indices = torch.ops.aten.max_pool3d_with_indices(
        ref_inp,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )

    res_out, res_indices = flag_gems.max_pool3d_with_indices(
        inp,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
    )

    utils.gems_assert_close(res_out, ref_out, dtype)
    # Indices are flat offsets into the input (D, H, W) volume, so an exact
    # comparison is required: a different index would select a different value.
    assert res_indices.shape == ref_indices.shape
    assert torch.equal(
        res_indices.to(torch.int64).cpu(), ref_indices.to(torch.int64).cpu()
    )


@pytest.mark.max_pool3d_with_indices
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_pool3d_with_indices_ties(dtype):
    """Repeated maxima must resolve to the same index as the reference."""
    inp = torch.ones((2, 3, 6, 6, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out, ref_indices = torch.ops.aten.max_pool3d_with_indices(ref_inp, 2, 2, 0, 1)
    res_out, res_indices = flag_gems.max_pool3d_with_indices(inp, 2, 2, 0, 1)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert torch.equal(
        res_indices.to(torch.int64).cpu(), ref_indices.to(torch.int64).cpu()
    )
