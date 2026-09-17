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
from flag_gems.ops._masked_softmax_backward import _masked_softmax_backward

from . import accuracy_utils as utils
from . import conftest as cfg

if cfg.QUICK_MODE:
    # (shape, dim): one small 2-D case exercised in quick mode.
    MASKED_SOFTMAX_BACKWARD_SHAPES = [((128, 256), 1)]
else:
    # Representative shapes covering small and reduction-heavy 2-D/3-D cases
    # along both an inner (dim=-1) and a non-inner reduction axis.
    MASKED_SOFTMAX_BACKWARD_SHAPES = [
        ((1, 8), 1),
        ((128, 256), 1),
        ((512, 512), 1),
        ((8, 16, 32), 2),
        ((32, 64, 64), 1),
    ]


def _reference_masked_softmax_backward(grad_output, output, mask, dim):
    out = torch.where(mask, torch.zeros_like(output), output)
    tmp = torch.where(mask, torch.zeros_like(out), out * grad_output)
    scale = tmp.sum(dim=dim, keepdim=True)
    grad_input = out * (grad_output - scale)
    return torch.where(mask, torch.zeros_like(grad_input), grad_input)


@pytest.mark.masked_softmax_backward
@pytest.mark.parametrize("shape_dim", MASKED_SOFTMAX_BACKWARD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_masked_softmax_backward(shape_dim, dtype):
    shape, dim = shape_dim
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # ``mask`` marks masked-out positions; build a consistent forward output so
    # masked entries are exactly zero, matching aten's contract.
    mask = torch.rand(shape, device=flag_gems.device) < 0.3
    output = torch.ops.aten._masked_softmax(inp, mask, dim, 2)
    output = torch.where(mask, torch.zeros_like(output), output)
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    # Reference runs in fp64 (on CPU under --ref=cpu); move every operand,
    # including the boolean mask, onto the reference device to stay consistent.
    ref_grad_output = utils.to_reference(grad_output, True)
    ref_output = utils.to_reference(output, True)
    ref_mask = utils.to_reference(mask)
    ref_grad_input = _reference_masked_softmax_backward(
        ref_grad_output, ref_output, ref_mask, dim
    )

    res_grad_input = _masked_softmax_backward(grad_output, output, mask, dim)

    utils.gems_assert_close(
        res_grad_input, ref_grad_input, dtype, reduce_dim=shape[dim]
    )
