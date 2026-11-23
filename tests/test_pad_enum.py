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

# aten _pad_enum integer mode enum: 0=reflect, 1=replicate, 2=circular, 3=constant.
# Only mode 3 (constant) consumes `value`; the others require value=None.


@pytest.mark.pad_enum
@pytest.mark.parametrize(
    "shape",
    # Representative N-D shapes; 3D/4D cover the common F.pad use cases.
    [(1024, 1024), (16, 128, 64, 60), (20, 320, 15)],
)
@pytest.mark.parametrize("value", [0.0, 1.5, -2.0])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__pad_enum_constant(shape, value, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    rank = len(shape)
    pad = [torch.randint(0, 8, (1,)).item() for _ in range(rank * 2)]

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten._pad_enum(ref_inp, pad, 3, value)
    res_out = flag_gems.ops._pad_enum(inp, pad, 3, value)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.pad_enum
@pytest.mark.parametrize(
    "shape_pad",
    # reflect: pad must be strictly smaller than the padded dim size.
    [((2, 3, 16, 16), [2, 2, 2, 2]), ((4, 8, 50), [7, 7])],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__pad_enum_reflect(shape_pad, dtype):
    shape, pad = shape_pad
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten._pad_enum(ref_inp, pad, 0, None)
    res_out = flag_gems.ops._pad_enum(inp, pad, 0, None)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.pad_enum
@pytest.mark.parametrize(
    "shape_pad",
    [((2, 3, 16, 16), [3, 4, 3, 4]), ((2, 3, 8, 8, 8), [1, 1, 1, 1, 1, 1])],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__pad_enum_replicate(shape_pad, dtype):
    shape, pad = shape_pad
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten._pad_enum(ref_inp, pad, 1, None)
    res_out = flag_gems.ops._pad_enum(inp, pad, 1, None)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.pad_enum
@pytest.mark.parametrize(
    "shape_pad",
    # circular: pad must be <= the padded dim size.
    [((2, 3, 24, 24), [5, 5, 5, 5]), ((4, 8, 40), [10, 10])],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__pad_enum_circular(shape_pad, dtype):
    shape, pad = shape_pad
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)
    ref_out = torch.ops.aten._pad_enum(ref_inp, pad, 2, None)
    res_out = flag_gems.ops._pad_enum(inp, pad, 2, None)

    utils.gems_assert_equal(res_out, ref_out)
