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


@pytest.mark.lt_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp1 = utils.to_reference(inp1.clone(), True)
    ref_inp2 = utils.to_reference(inp2, True)
    ref_out = ref_inp1.lt_(ref_inp2)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.lt_.Tensor(inp1, inp2)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out is inp1


@pytest.mark.lt_scalar_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_lt_scalar_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    scalar = 0

    ref_inp = utils.to_reference(inp.clone(), True)
    ref_out = ref_inp.lt_(scalar)

    with flag_gems.use_gems():
        res_out = torch.ops.aten.lt_.Scalar(inp, scalar)

    utils.gems_assert_close(res_out, ref_out, dtype)
    assert res_out is inp
