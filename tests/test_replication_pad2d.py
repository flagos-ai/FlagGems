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

# 4D shapes covering small (8x8) to medium (512x512) spatial sizes;
# pad2d operates on the last two dims. Includes a 3D (C, H, W) case via
# the dedicated `test_replication_pad2d_3d_input` below.
if cfg.QUICK_MODE:
    # Quick-mode: one minimal shape + one moderate padding for smoke testing.
    REPLICATION_PAD2D_SHAPES = [(2, 3, 8, 8)]
    REPLICATION_PAD2D_PADDINGS = [(1, 1, 2, 2)]
else:
    # Shapes: 8x8 (minimal), 128x256 (mid), 512x512 (large) — same coverage
    # as the experimental_ops test in PR #1384.
    REPLICATION_PAD2D_SHAPES = [(2, 3, 8, 8), (4, 8, 128, 256), (2, 4, 512, 512)]
    # Paddings: zero (identity), uniform, and asymmetric left/top-only.
    REPLICATION_PAD2D_PADDINGS = [(0, 0, 0, 0), (1, 1, 2, 2), (3, 0, 0, 3)]


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("shape", REPLICATION_PAD2D_SHAPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD2D_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad2d(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.replication_pad2d(ref_inp, padding)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad2d(inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.replication_pad2d_out
@pytest.mark.parametrize("shape", REPLICATION_PAD2D_SHAPES)
@pytest.mark.parametrize("padding", REPLICATION_PAD2D_PADDINGS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad2d_out(shape, padding, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad_left, pad_right, pad_top, pad_bottom = padding
    H_out = shape[-2] + pad_top + pad_bottom
    W_out = shape[-1] + pad_left + pad_right
    out_shape = (*shape[:-2], H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = utils.to_reference(out)

    torch.ops.aten.replication_pad2d.out(ref_inp, padding, out=ref_out)

    with flag_gems.use_gems():
        torch.ops.aten.replication_pad2d.out(inp, padding, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.replication_pad2d
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_replication_pad2d_3d_input(dtype):
    """Test with 3D input (C, H, W) — no batch dimension."""
    # 3D input shape (C, H, W); moderate padding to exercise all four edges.
    shape = (3, 8, 8)
    padding = (1, 2, 3, 4)

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.replication_pad2d(ref_inp, padding)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.replication_pad2d(inp, padding)

    utils.gems_assert_close(res_out, ref_out, dtype)
