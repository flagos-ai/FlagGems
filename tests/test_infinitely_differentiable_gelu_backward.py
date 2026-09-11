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


@pytest.mark.infinitely_differentiable_gelu_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_infinitely_differentiable_gelu_backward(shape, dtype):
    grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    self_input = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_grad = utils.to_reference(grad, True)
    ref_self = utils.to_reference(self_input, True)

    ref_out = torch.ops.aten.infinitely_differentiable_gelu_backward(ref_grad, ref_self)
    res_out = torch.ops.aten.infinitely_differentiable_gelu_backward(grad, self_input)

    # Use higher tolerance for low precision dtypes due to accumulated errors in exp/erf operations
    atol = 1e-2 if dtype in [torch.float16, torch.bfloat16] else 1e-4
    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
