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


@pytest.mark.clamp_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("bounds", [(-0.5, 0.5), (-1.0, 1.0), (0.0, 0.0), (0.5, -0.5)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_backward(shape, bounds, dtype):
    min_value, max_value = bounds
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp, True)
    ref_grad = utils.to_reference(out_grad, True)

    ref_out = torch.clamp(ref_inp, min=min_value, max=max_value)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)

    res_in_grad = flag_gems.clamp_backward(out_grad, inp, min_value, max_value)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
