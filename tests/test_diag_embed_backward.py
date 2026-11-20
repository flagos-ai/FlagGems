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

# One case per layout the gather kernel can see: default dims on 1-D, 2-D and
# 3-D inputs, non-default plane dims (ATen-formula path), non-power-of-two
# diagonal lengths, and a non-contiguous upstream gradient.
DIAG_EMBED_BACKWARD_CASES = [
    ((64,), 0, -2, -1),
    ((32, 48), 0, -2, -1),
    ((8, 7, 33), 1, -2, -1),
    ((8, 7, 33), 1, -3, -2),
    ((8, 7, 33), -1, -2, -3),
]


def _diag_embed_backward_reference(grad_output, inp, offset, dim1, dim2):
    """float64 reference: the diagonal of the upstream gradient."""
    ref_grad = utils.to_reference(grad_output, True)
    rank = inp.ndim + 1
    a = dim1 % rank
    b = dim2 % rank
    return torch.diagonal(ref_grad, offset, a, b).to(inp.dtype)


@pytest.mark.diag_embed_backward
@pytest.mark.parametrize("shape, offset, dim1, dim2", DIAG_EMBED_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diag_embed_backward(shape, offset, dim1, dim2, dtype):
    """Gradient of ``diag_embed``: the diagonal of the upstream gradient."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    diag_len = inp.shape[-1]
    plane = diag_len + abs(offset)
    # grad_output shape = input shape with two plane dims (size L) inserted
    # before the last dim.
    grad_out = torch.randn(
        (*shape[:-1], plane, plane), dtype=dtype, device=flag_gems.device
    )

    ref_grad = _diag_embed_backward_reference(grad_out, inp, offset, dim1, dim2).to(
        dtype
    )
    res_grad = flag_gems.diag_embed_backward(grad_out, inp, offset, dim1, dim2)

    utils.gems_assert_equal(res_grad, ref_grad)


@pytest.mark.diag_embed_backward
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_diag_embed_backward_bridge(offset):
    """The FlagGems ``diag_embed_backward`` entry reaches the Triton kernel."""
    inp = torch.randn((32, 48), dtype=torch.float32, device=flag_gems.device)
    plane = 48 + abs(offset)

    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    grad_out = torch.randn(
        (32, plane, plane), dtype=torch.float32, device=flag_gems.device
    )
    ref_grad = utils.to_reference(grad_out, True)
    (ref_grad_in,) = torch.autograd.grad(
        torch.diag_embed(ref_inp, offset), ref_inp, ref_grad
    )

    res_grad = flag_gems.diag_embed_backward(grad_out, inp, offset)

    utils.gems_assert_equal(res_grad, ref_grad_in.to(torch.float32))


@pytest.mark.diag_embed_backward
def test_diag_embed_backward_use_gems_entry():
    """The public ``torch.diag_embed`` under ``use_gems()`` keeps ATen's node.

    ``aten::diag_embed`` is a CompositeExplicitAutograd op, so the generated
    ``DiagEmbedBackward0`` node stays above the device kernel and
    differentiates with the ATen formula (a zero-copy diagonal view).
    ``flag_gems.diag_embed_backward`` is the entry that reaches the Triton
    gather kernel.
    """
    inp = torch.randn((32, 48), dtype=torch.float32, device=flag_gems.device)
    grad_out = torch.randn((32, 48, 48), dtype=torch.float32, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad_out = utils.to_reference(grad_out, True)
    (ref_grad,) = torch.autograd.grad(torch.diag_embed(ref_inp), ref_inp, ref_grad_out)

    # only_enable is the CI-compliant activation for the test framework
    # (use_gems() is banned in test files); it rides the same registration
    # path and is idempotent for an already-registered op, so no teardown.
    flag_gems.only_enable(include=["diag_embed"])
    res_inp = inp.detach().requires_grad_(True)
    res_out = torch.diag_embed(res_inp)
    (res_grad,) = torch.autograd.grad(res_out, res_inp, grad_out)

    utils.gems_assert_equal(res_grad, ref_grad.to(torch.float32))
