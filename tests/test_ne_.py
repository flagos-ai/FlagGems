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


@pytest.mark.ne_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ne_(shape, dtype):
    inp1 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)
    ref_out = ref_inp1.ne_(ref_inp2)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.ne_.Tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp1, ref_inp1, dtype)
    assert res_out is inp1


@pytest.mark.ne_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_ne_scalar_(shape, dtype):
    inp = torch.randint(0, 10, shape, dtype=dtype, device=flag_gems.device)
    scalar = 0

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.ne_(scalar)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.ne_.Scalar(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)
    utils.gems_assert_close(inp, ref_inp, dtype)
    assert res_out is inp
