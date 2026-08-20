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
    PAD2D_SHAPES = [(2, 4, 32, 64)]
    PAD2D_DTYPES = [torch.float32]
    PAD2D_PADDINGS = [(1, 1, 1, 1)]
    PAD2D_LIST_PADDINGS = [[1, 1, 1, 1]]
    PAD2D_3D_PADDINGS = [(1, 1, 1, 1)]
    PAD2D_OUT_SHAPES = [(2, 4, 32, 64)]
    PAD2D_OUT_DTYPES = [torch.float32]
    PAD2D_OUT_PADDINGS = [(1, 1, 1, 1)]
else:
    PAD2D_SHAPES = [(3, 33, 33), (2, 4, 32, 64), (8, 16, 64, 64), (32, 64, 128, 256)]
    PAD2D_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
    PAD2D_PADDINGS = [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
        (4, 0, 4, 0),
    ]
    PAD2D_LIST_PADDINGS = [[1, 1, 1, 1], [2, 3, 4, 5]]
    PAD2D_3D_PADDINGS = [(1, 1, 1, 1), (2, 3, 4, 5)]
    PAD2D_OUT_SHAPES = [
        (3, 33, 33),
        (2, 4, 32, 64),
        (8, 16, 64, 64),
        (32, 64, 128, 256),
    ]
    PAD2D_OUT_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
    PAD2D_OUT_PADDINGS = [
        (1, 1, 1, 1),
        (2, 3, 2, 3),
        (3, 5, 3, 5),
        (0, 4, 0, 4),
        (4, 0, 4, 0),
    ]


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("shape", PAD2D_SHAPES)
@pytest.mark.parametrize("dtype", PAD2D_DTYPES)
@pytest.mark.parametrize(
    "padding",
    PAD2D_PADDINGS,
)
def test_reflection_pad2d(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("padding", PAD2D_LIST_PADDINGS)
def test_reflection_pad2d_list_padding(padding):
    # Test with list format: [pad_left, pad_right, pad_top, pad_bottom]
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
def test_reflection_pad2d_empty_padding():
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    padding = (0, 0, 0, 0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d
@pytest.mark.parametrize("padding", PAD2D_3D_PADDINGS)
def test_reflection_pad2d_3d_input(padding):
    # Test with 3D input (C, H, W) - no batch dimension
    shape = (3, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad2d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d_out
@pytest.mark.parametrize("shape", PAD2D_OUT_SHAPES)
@pytest.mark.parametrize("dtype", PAD2D_OUT_DTYPES)
@pytest.mark.parametrize(
    "padding",
    PAD2D_OUT_PADDINGS,
)
def test_reflection_pad2d_out(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad_left, pad_right, pad_top, pad_bottom = padding
    H_out = x.shape[-2] + pad_top + pad_bottom
    W_out = x.shape[-1] + pad_left + pad_right
    out_shape = (*x.shape[:-2], H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        torch.ops.aten.reflection_pad2d.out(x, padding, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d_out
@pytest.mark.parametrize("padding", PAD2D_3D_PADDINGS)
def test_reflection_pad2d_out_3d_input(padding):
    shape = (3, 32, 64)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    pad_left, pad_right, pad_top, pad_bottom = padding
    H_out = x.shape[-2] + pad_top + pad_bottom
    W_out = x.shape[-1] + pad_left + pad_right
    out_shape = (*x.shape[:-2], H_out, W_out)
    out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        torch.ops.aten.reflection_pad2d.out(x, padding, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad2d_out
def test_reflection_pad2d_out_empty_padding():
    shape = (2, 4, 32, 64)
    dtype = torch.float32
    padding = (0, 0, 0, 0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad2d(ref_x, padding)

    with flag_gems.use_gems():
        torch.ops.aten.reflection_pad2d.out(x, padding, out=out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
