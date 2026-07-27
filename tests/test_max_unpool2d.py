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

# Shapes for max_unpool2d: (N, C, H, W) input sizes covering even/odd dims
MAX_UNPOOL2D_SHAPES = [
    (1, 1, 4, 4),
    (1, 1, 8, 8),
    (2, 3, 8, 8),
    (1, 16, 16, 16),
    (4, 8, 16, 16),
    # Odd-dimension shapes: tests coefficient for ceil_mode=False MaxPool
    # where indices may map outside the region divisible by kernel_size
    (1, 1, 5, 5),
    (2, 3, 7, 9),
    # Single-element spatial dims
    (1, 1, 2, 2),
    # Non-square shapes
    (1, 1, 6, 10),
    (2, 4, 10, 6),
]

# Pooling configurations to test: (kernel_size, stride, padding)
POOL_CONFIGS = [
    (2, 2, 0),
    (3, 2, 0),
    (3, 3, 0),
]


@pytest.mark.max_unpool2d
@pytest.mark.parametrize("shape", MAX_UNPOOL2D_SHAPES)
@pytest.mark.parametrize("pool_cfg", POOL_CONFIGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_unpool2d(shape, pool_cfg, dtype):
    kernel_size, stride, padding = pool_cfg
    H, W = shape[2], shape[3]
    out_h = (H + 2 * padding - kernel_size) // stride + 1
    out_w = (W + 2 * padding - kernel_size) // stride + 1
    if out_h <= 0 or out_w <= 0:
        pytest.skip(
            f"Output size ({out_h},{out_w}) too small for shape {shape} + cfg {pool_cfg}"
        )
    # Create input tensor
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Apply max_pool2d to get pooled output and indices
    pool = torch.nn.MaxPool2d(
        kernel_size, stride=stride, padding=padding, return_indices=True
    )
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


@pytest.mark.max_unpool2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_max_unpool2d_non_contiguous(dtype):
    """Test that non-contiguous inputs are handled correctly."""
    inp = torch.randn(2, 3, 8, 8, dtype=dtype, device=flag_gems.device)
    # Transpose to create non-contiguous tensor layout
    inp_noncontig = inp.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    assert not inp_noncontig.is_contiguous()

    pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True)
    pooled_noncontig, indices_noncontig = pool(inp_noncontig)

    output_size = [8, 8]
    ref_pooled = utils.to_reference(pooled_noncontig)
    ref_indices = utils.to_reference(indices_noncontig)
    ref_out = torch.ops.aten.max_unpool2d(
        ref_pooled, ref_indices.to(torch.int64), output_size
    )

    with flag_gems.use_gems():
        res_out = torch.ops.aten.max_unpool2d(
            pooled_noncontig, indices_noncontig.to(torch.int64), output_size
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
