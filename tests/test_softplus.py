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


@pytest.mark.softplus
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    beta = torch.rand(1).item()
    threshold = torch.rand(1).item() * 40.0
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.nn.functional.softplus(ref_inp, beta=beta, threshold=threshold)

    with flag_gems.use_gems():
        res_out = torch.nn.functional.softplus(inp, beta=beta, threshold=threshold)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.softplus_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_softplus_backward(shape, dtype):
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    res_grad_output = torch.randn_like(res_inp)
    beta = torch.rand(1).item() + 0.5  # Ensure beta > 0.5 for stability
    threshold = torch.rand(1).item() * 40.0

    ref_inp = utils.to_reference(res_inp, True)
    ref_grad_output = utils.to_reference(res_grad_output, True)

    ref_grad_input = torch.ops.aten.softplus_backward(
        ref_grad_output, ref_inp, beta=beta, threshold=threshold
    )
    with flag_gems.use_gems():
        res_grad_input = torch.ops.aten.softplus_backward(
            res_grad_output, res_inp, beta=beta, threshold=threshold
        )

    utils.gems_assert_close(res_grad_input, ref_grad_input, dtype)
