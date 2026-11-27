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


@pytest.mark.isnan
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_isnan(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp = torch.masked_fill(inp, inp > 1.0, float("nan"))
    ref_inp = utils.to_reference(inp)

    ref_out = torch.isnan(ref_inp)
    res_out = flag_gems.isnan(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.isnan_backward
def test_isnan_backward_non_differentiable():
    """``isnan`` outputs a bool tensor, so it has no gradient.

    The backward form is undefined; this pins that the output matches the
    native op and is non-differentiable, rather than fabricating a gradient
    for a boolean result.
    """
    inp = torch.randn(1024, dtype=torch.float32, device=flag_gems.device)
    inp[::3] = float("nan")
    ref_inp = utils.to_reference(inp, False)

    res_out = flag_gems.isnan(inp)
    ref_out = torch.isnan(ref_inp)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.dtype == torch.bool
    assert not res_out.requires_grad
    assert res_out.grad_fn is None
