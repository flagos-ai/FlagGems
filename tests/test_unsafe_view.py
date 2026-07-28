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


@pytest.mark.unsafe_view
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_view(shape, dtype):
    # Test various reshapes that maintain the same number of elements
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Test reshape to 1D
    new_shape = (inp.numel(),)
    ref_out = torch.ops.aten._unsafe_view.default(ref_inp, new_shape)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_view.default(inp, new_shape)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.unsafe_view
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_view_2d(shape, dtype):
    # Test reshape to 2D
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # Calculate a valid 2D shape
    numel = inp.numel()
    # Use factors that divide evenly
    if numel % 4 == 0:
        new_shape = (numel // 4, 4)
    elif numel % 2 == 0:
        new_shape = (numel // 2, 2)
    else:
        new_shape = (numel, 1)

    ref_out = torch.ops.aten._unsafe_view.default(ref_inp, new_shape)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_view.default(inp, new_shape)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.unsafe_view
@pytest.mark.parametrize("shape", utils.SPECIAL_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_unsafe_view_infer_dim(shape, dtype):
    # Test reshape with -1 dimension inference
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    numel = inp.numel()
    # Use a valid shape with -1
    if numel % 3 == 0:
        new_shape = (3, -1)
    elif numel % 2 == 0:
        new_shape = (2, -1)
    else:
        new_shape = (-1, 1)

    ref_out = torch.ops.aten._unsafe_view.default(ref_inp, new_shape)
    with flag_gems.use_gems():
        res_out = torch.ops.aten._unsafe_view.default(inp, new_shape)

    utils.gems_assert_equal(res_out, ref_out)
