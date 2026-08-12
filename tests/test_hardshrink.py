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


@pytest.mark.hardshrink
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("lambd", [0.0, 0.5, 1.0])
def test_hardshrink(shape, dtype, lambd):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten.hardshrink(ref_inp, lambd)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.hardshrink(inp, lambd)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.hardshrink_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("lambd", [0.0, 0.75, 1.5])
def test_hardshrink_out(shape, dtype, lambd):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.ops.aten.hardshrink.out(ref_inp, lambd, out=ref_out)
    with flag_gems.use_gems():
        res_out = torch.empty_like(inp)
        torch.ops.aten.hardshrink.out(inp, lambd, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)
