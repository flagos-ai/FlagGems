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


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    # 5D shapes: (N, C, D, H, W) covering small/medium volumes
    "shape",
    [
        (2, 4, 8, 16, 16),
        (1, 3, 16, 32, 32),
        (2, 4, 16, 32, 64),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 2, 3, 1, 1),
        (1, 1, 2, 2, 2, 2),
        (0, 4, 0, 4, 0, 4),
        (4, 0, 4, 0, 4, 0),
    ],
)
def test_reflection_pad3d(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 4, 5, 2, 1),
    ],
)
def test_reflection_pad3d_list_padding(padding):
    # Fixed shape for padding-focused test
    shape = (2, 4, 8, 16, 16)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d
def test_reflection_pad3d_empty_padding():
    # Fixed shape for zero-padding edge case
    shape = (2, 4, 8, 16, 16)
    dtype = torch.float32
    padding = (0, 0, 0, 0, 0, 0)
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 4, 5, 2, 1),
    ],
)
def test_reflection_pad3d_4d_input(padding):
    # 4D input (C, D, H, W) without batch dimension
    shape = (3, 8, 16, 16)
    dtype = torch.float32
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_x = utils.to_reference(x.clone())
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    with flag_gems.use_gems():
        act_out = flag_gems.reflection_pad3d(x, padding)

    utils.gems_assert_close(act_out, ref_out, dtype, equal_nan=True)


@pytest.mark.reflection_pad3d_out
@pytest.mark.parametrize(
    # 5D shapes for out= variant
    "shape",
    [
        (2, 4, 8, 16, 16),
        (1, 3, 16, 32, 32),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "padding",
    [
        (1, 1, 1, 1, 1, 1),
        (2, 3, 2, 3, 1, 1),
    ],
)
def test_reflection_pad3d_out(shape, dtype, padding):
    x = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.ops.aten.reflection_pad3d(ref_x, padding)

    out = torch.empty(ref_out.shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        flag_gems.reflection_pad3d_out(x, padding, out)

    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)
