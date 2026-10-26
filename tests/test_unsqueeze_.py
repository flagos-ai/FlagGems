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

UNSQUEEZE_DIMS = [0, 1, 2, -1, -2]


@pytest.mark.unsqueeze_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", UNSQUEEZE_DIMS)
def test_unsqueeze_(shape, dtype, dim):
    ndim = len(shape)
    valid_dim = dim if dim >= 0 else ndim + dim + 1
    if valid_dim < 0 or valid_dim > ndim:
        pytest.skip("Invalid dim for shape")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())
    ref_out = ref_inp.unsqueeze_(dim)
    inp_ptr = inp.data_ptr()
    res_out = flag_gems.unsqueeze_(inp, dim)
    assert res_out is inp
    assert inp.data_ptr() == inp_ptr
    utils.gems_assert_equal(res_out, ref_out)
    utils.gems_assert_equal(inp, ref_inp)
