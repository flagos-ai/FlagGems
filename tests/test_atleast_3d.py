# Copyright 2026, The FlagOS Contributors.
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


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_3d(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.dim() >= 3
    # atleast_3d is a view operator: the result must alias the input storage
    # rather than materialise a copy.
    assert res_out.data_ptr() == inp.data_ptr()


@pytest.mark.atleast_3d
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_atleast_3d_nan_inf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    flat = inp.flatten()
    # Place nan/inf/-inf in the leading elements so every shape (including the
    # 0-dim and single-element cases) carries at least one special value.
    for i, value in enumerate([float("nan"), float("inf"), float("-inf")]):
        if i < flat.numel():
            flat[i] = value
    inp = flat.reshape(shape)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.atleast_3d(ref_inp)
    res_out = flag_gems.atleast_3d(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    assert res_out.dim() >= 3
