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

# One case per layout the kernels see: trailing and leading plane dims,
# swapped dim1/dim2, non-square planes, positive/negative offsets and a rank-4
# input with dims strictly inside the shape.
DIAGONAL_SCATTER_BACKWARD_CASES = [
    ((32, 48), 0, 0, 1),
    ((32, 48), 1, 0, 1),
    ((32, 48), -2, 0, 1),
    ((16, 8, 7), 0, 1, 2),
    ((16, 8, 7), 2, 1, 2),
    ((16, 8, 7), 0, 2, 1),
    ((4, 3, 5, 6), -1, 1, 3),
    ((4, 3, 5, 6), 0, 3, 0),
]


def _diagonal_scatter_backward_reference(grad_output, src, offset, dim1, dim2):
    """float64 reference following ATen's ``DiagonalScatterBackward0`` formula."""
    ref_grad = utils.to_reference(grad_output, True)
    ref_src = utils.to_reference(src, True)
    zeros = torch.zeros_like(ref_src)
    grad_input = torch.diagonal_scatter(ref_grad, zeros, offset, dim1, dim2)
    grad_src = torch.diagonal(ref_grad, offset, dim1, dim2).contiguous()
    return grad_input.to(src.dtype), grad_src.to(src.dtype)


@pytest.mark.diagonal_scatter_backward
@pytest.mark.parametrize("shape, offset, dim1, dim2", DIAGONAL_SCATTER_BACKWARD_CASES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_diagonal_scatter_backward(shape, offset, dim1, dim2, dtype):
    """Gradients of ``diagonal_scatter`` w.r.t. ``self`` and ``src``."""
    grad_output = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    diag = torch.diagonal(grad_output, offset, dim1, dim2)
    src = torch.randn(diag.shape, dtype=dtype, device=flag_gems.device)

    ref_input, ref_src = _diagonal_scatter_backward_reference(
        grad_output, src, offset, dim1, dim2
    )
    res_input, res_src = flag_gems.diagonal_scatter_backward(
        grad_output, src, offset, dim1, dim2
    )

    utils.gems_assert_equal(res_input, ref_input)
    utils.gems_assert_equal(res_src, ref_src)


@pytest.mark.diagonal_scatter_backward
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_diagonal_scatter_backward_autograd(offset):
    """The FlagGems entry matches the gradients produced by ``torch.autograd``."""
    inp = torch.randn((16, 32, 48), dtype=torch.float32, device=flag_gems.device)
    diag = torch.diagonal(inp, offset, 1, 2)
    src = torch.randn(diag.shape, dtype=torch.float32, device=flag_gems.device)
    grad_output = torch.randn_like(inp)

    ref_inp = utils.to_reference(inp.clone(), True).requires_grad_(True)
    ref_src = utils.to_reference(src.clone(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_output, True)
    ref_out = torch.diagonal_scatter(ref_inp, ref_src, offset, 1, 2)
    ref_grad_in, ref_grad_src = torch.autograd.grad(
        ref_out, (ref_inp, ref_src), ref_grad
    )

    res_grad_in, res_grad_src = flag_gems.diagonal_scatter_backward(
        grad_output, src, offset, 1, 2
    )

    utils.gems_assert_equal(res_grad_in, ref_grad_in.to(torch.float32))
    utils.gems_assert_equal(res_grad_src, ref_grad_src.to(torch.float32))


@pytest.mark.diagonal_scatter_backward
def test_diagonal_scatter_backward_empty_diagonal():
    """An offset past the plane yields empty gradients on both sides."""
    grad_output = torch.randn((8, 4, 6), dtype=torch.float32, device=flag_gems.device)
    diag = torch.diagonal(grad_output, 6, 1, 2)
    src = torch.empty(diag.shape, dtype=torch.float32, device=flag_gems.device)

    res_input, res_src = flag_gems.diagonal_scatter_backward(grad_output, src, 6, 1, 2)

    # ``grad_output`` is the expected ``grad_input`` here (the diagonal is empty,
    # so the copy kernel returns it unchanged).  Move both operands to the CPU
    # explicitly: ``gems_assert_equal`` only relocates ``res`` under ``--ref=cpu``,
    # which would leave ``ref`` on the device and trip a device mismatch.
    utils.gems_assert_equal(res_input.cpu(), grad_output.cpu())
    assert res_src.shape == src.shape


@pytest.mark.diagonal_scatter_backward
def test_diagonal_scatter_backward_use_gems_entry():
    """``flag_gems.diagonal_scatter_backward`` is exposed via its explicit entry.

    ``aten::diagonal_scatter`` is CompositeExplicitAutograd and there is no
    ``aten::diagonal_scatter_backward`` schema, so the operator is called
    through the explicit ``flag_gems.diagonal_scatter_backward`` entry point.
    """
    grad_output = torch.randn((8, 32, 32), dtype=torch.float32, device=flag_gems.device)
    src = torch.randn((8, 32), dtype=torch.float32, device=flag_gems.device)

    ref_input, ref_src = _diagonal_scatter_backward_reference(grad_output, src, 0, 1, 2)
    res_input, res_src = flag_gems.diagonal_scatter_backward(grad_output, src, 0, 1, 2)

    utils.gems_assert_equal(res_input, ref_input)
    utils.gems_assert_equal(res_src, ref_src)
