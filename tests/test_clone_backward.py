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
"""Accuracy tests for the gradient of ``clone``.

ATen has no ``aten::clone_backward`` schema: ``clone`` is a pure copy whose VJP
is the identity, implemented by the C++ ``CopyBackwards`` node. FlagGems keeps
the same contract in ``_CloneFunction.backward`` (``flag_gems.ops.clone``), so
these tests drive the full autograd chain -- forward through ``flag_gems.clone``,
then ``torch.autograd.grad`` -- instead of calling a standalone backward helper.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _grad_pair(inp, out_grad, memory_format=None):
    """Run the reference and FlagGems clone backwards on the same upstream grad."""
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad = utils.to_reference(out_grad)

    kwargs = {} if memory_format is None else {"memory_format": memory_format}
    res_out = flag_gems.clone(inp, **kwargs)
    ref_out = torch.ops.aten.clone(ref_inp, **kwargs)

    (res_in_grad,) = torch.autograd.grad(res_out, inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(ref_out, ref_inp, ref_grad)
    return res_in_grad, ref_in_grad


@pytest.mark.clone_backward
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_backward(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    # A non-uniform upstream gradient distinguishes a real pass-through from a
    # gradient of ones.
    out_grad = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    res_in_grad, ref_in_grad = _grad_pair(inp, out_grad)

    # clone is a pure copy, so its gradient is the upstream gradient verbatim;
    # that makes the comparison exact rather than tolerance based.
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
    # Compare against the reference-side grad_output so the expectation lives on
    # the same device under --ref=cpu, where gems_assert_equal moves only res.
    utils.gems_assert_equal(res_in_grad, utils.to_reference(out_grad))


@pytest.mark.clone_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "case",
    ["expanded_grad", "transposed_grad", "stepped_grad", "offset_grad"],
)
def test_clone_backward_noncontiguous_grad(dtype, case):
    """A non-contiguous upstream gradient must not be read as a flat buffer.

    ``CopyBackwards`` forwards ``grad_output`` untouched, so the input gradient
    inherits its layout. A backward that assumed contiguity would silently
    return transposed or broadcast values here.
    """
    torch.manual_seed(0)
    base = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    if case == "expanded_grad":
        out_grad = base[:, :1].expand(4, 8)
    elif case == "transposed_grad":
        out_grad = base.t()
    elif case == "stepped_grad":
        out_grad = base[:, ::2]
    else:
        out_grad = torch.randn((4, 16), dtype=dtype, device=flag_gems.device)[:, ::2]

    inp = torch.randn(out_grad.shape, dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)

    res_in_grad, ref_in_grad = _grad_pair(inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.clone_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("transpose", [False, True])
def test_clone_backward_noncontiguous_input(dtype, transpose):
    """The gradient must match the input's shape even when the input is a view."""
    torch.manual_seed(0)
    base = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    inp = (base.t() if transpose else base[:, ::2]).requires_grad_(True)
    out_grad = torch.randn(inp.shape, dtype=dtype, device=flag_gems.device)

    res_in_grad, ref_in_grad = _grad_pair(inp, out_grad)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.clone_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "memory_format", [torch.preserve_format, torch.contiguous_format]
)
def test_clone_backward_memory_format(dtype, memory_format):
    """``memory_format`` is a layout hint only; it must not change the gradient."""
    torch.manual_seed(0)
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device).t()
    inp.requires_grad_(True)
    out_grad = torch.randn(inp.shape, dtype=dtype, device=flag_gems.device)

    res_in_grad, ref_in_grad = _grad_pair(inp, out_grad, memory_format)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.clone_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_backward_dtype_mismatched_grad(dtype):
    """``CopyBackwards`` converts a dtype-mismatched grad back to the input dtype.

    This is the one branch where the identity VJP is *not* a straight
    pass-through, so it is the case a "just return grad_output" backward would
    get wrong.
    """
    inp = torch.randn((8, 16), dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    # A wider grad dtype forces the conversion branch on both sides.
    grad_dtype = torch.float32 if dtype != torch.float32 else torch.float64
    out_grad = torch.randn(inp.shape, dtype=grad_dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad = utils.to_reference(out_grad)

    (res_in_grad,) = torch.autograd.grad(flag_gems.clone(inp), inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(
        torch.ops.aten.clone(ref_inp), ref_inp, ref_grad
    )

    # The gradient comes back in the input dtype, not the upstream grad's.
    assert res_in_grad.dtype == inp.dtype
    assert ref_in_grad.dtype == ref_inp.dtype
    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)


@pytest.mark.clone_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_backward_backward_accumulates_into_grad(dtype):
    """``.backward()`` must accumulate a copy, so later grad mutation is isolated."""
    torch.manual_seed(0)
    inp = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    out_grad = torch.randn(inp.shape, dtype=dtype, device=flag_gems.device)

    # Reference first, on an unmutated copy of the same upstream gradient.
    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_grad = utils.to_reference(out_grad)
    torch.ops.aten.clone(ref_inp).backward(ref_grad)

    flag_gems.clone(inp).backward(out_grad)
    grad_after_first = inp.grad.clone()
    # Mutating the incoming grad must not retroactively change ``inp.grad``.
    out_grad.add_(1)
    assert not torch.equal(inp.grad, out_grad)

    utils.gems_assert_close(grad_after_first, ref_inp.grad, dtype)


@pytest.mark.clone_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clone_backward_empty(dtype):
    """An empty tensor has an empty gradient and must not launch a kernel."""
    inp = torch.randn((0, 3), dtype=dtype, device=flag_gems.device)
    inp.requires_grad_(True)
    out_grad = torch.randn((0, 3), dtype=dtype, device=flag_gems.device)

    (res_in_grad,) = torch.autograd.grad(flag_gems.clone(inp), inp, out_grad)

    # Shape and emptiness are the whole contract here; there is nothing to
    # compare element-wise. ``gems_assert_equal`` would trip on the reference's
    # upcast dtype, so compare against the same-dtype upstream grad.
    assert res_in_grad.shape == inp.shape
    assert res_in_grad.numel() == 0
    utils.gems_assert_equal(res_in_grad, utils.to_reference(out_grad))


@pytest.mark.clone_backward
@pytest.mark.skipif(utils.TO_CPU, reason="aliasing is only observable on the device")
def test_clone_backward_is_an_identity_pass_through():
    """The VJP reuses the upstream gradient's storage, as ``CopyBackwards`` does.

    Pinning this down keeps someone from turning the backward into a second
    device copy: on a 1 GiB tensor that would add a full read+write pass to a
    path that costs nothing today.
    """
    inp = torch.randn((1024, 1024), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)
    out_grad = torch.randn(inp.shape, dtype=torch.float32, device=flag_gems.device)

    (res_in_grad,) = torch.autograd.grad(flag_gems.clone(inp), inp, out_grad)
    (ref_in_grad,) = torch.autograd.grad(torch.ops.aten.clone(inp), inp, out_grad)

    assert res_in_grad.untyped_storage().data_ptr() == (
        out_grad.untyped_storage().data_ptr()
    )
    assert ref_in_grad.untyped_storage().data_ptr() == (
        out_grad.untyped_storage().data_ptr()
    )
