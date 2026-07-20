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


@pytest.mark.hardsigmoid
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardsigmoid(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(res_inp, True)

    ref_out = torch.nn.functional.hardsigmoid(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.nn.functional.hardsigmoid(res_inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardsigmoid_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_hardsigmoid_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.hardsigmoid.out(ref_inp, out=ref_out)

    out = torch.empty_like(inp)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardsigmoid.out(inp, out=out)

    assert res_out is out
    utils.gems_assert_close(out, ref_out, dtype)
