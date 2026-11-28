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

# torch_gcu cannot build a legacy (torch.strided) NestedTensor from a flat value
# buffer: `_nested_view_from_buffer` is not implemented there, so the cases that
# compare nested metadata stay device-path only.  The mask validation case below
# never reaches that constructor, so it still runs on the enflame backend.
_NESTED_VIEW_FROM_BUFFER = flag_gems.vendor_name != "enflame"


# (N, L, D) shapes for the padded dense tensor `t` and the (N, L) mask.
NESTED_FROM_MASK_SHAPES = [
    (1, 4, 3),
    (2, 8, 4),
    (8, 16, 8),
    (16, 64, 32),
    (64, 128, 16),
    (4, 256, 64),
]


def _left_aligned_mask(N, L, device):
    """Generate a random left-aligned boolean mask of shape (N, L)."""
    lengths = torch.randint(0, L + 1, (N,), device=device)
    idx = torch.arange(L, device=device).unsqueeze(0)
    return idx < lengths.unsqueeze(1)


def _check_nested_metadata(res, ref):
    """Assert the result is a legacy NestedTensor with the same structural metadata.

    The reference aten implementation returns a legacy (``torch.strided`` layout)
    NestedTensor, so FlagGems must match it rather than returning a jagged
    NestedTensor -- otherwise downstream ops relying on strided layout would
    silently take a different code path.
    """
    assert res.layout == ref.layout
    assert res.is_nested == ref.is_nested
    assert torch.equal(res._nested_tensor_size(), ref._nested_tensor_size())
    assert torch.equal(res._nested_tensor_strides(), ref._nested_tensor_strides())
    assert torch.equal(
        res._nested_tensor_storage_offsets(), ref._nested_tensor_storage_offsets()
    )


def _assert_nested_equal(res, ref, dtype):
    _check_nested_metadata(res, ref)
    res_comps = torch.unbind(res)
    ref_comps = torch.unbind(ref)
    assert len(res_comps) == len(ref_comps)
    for res_t, ref_t in zip(res_comps, ref_comps):
        assert res_t.shape == ref_t.shape
        utils.gems_assert_close(res_t, ref_t, dtype)


@pytest.mark.nested_tensor_from_mask
@pytest.mark.skipif(
    not _NESTED_VIEW_FROM_BUFFER,
    reason="torch_gcu cannot build a legacy nested view from a flat buffer",
)
@pytest.mark.parametrize("shape", NESTED_FROM_MASK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask(shape, dtype):
    N, L, D = shape
    t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    mask = _left_aligned_mask(N, L, flag_gems.device)

    ref_t = utils.to_reference(t)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten._nested_tensor_from_mask(ref_t, ref_mask, mask_check=True)
    res_out = flag_gems._nested_tensor_from_mask(t, mask, mask_check=True)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_tensor_from_mask
@pytest.mark.skipif(
    not _NESTED_VIEW_FROM_BUFFER,
    reason="torch_gcu cannot build a legacy nested view from a flat buffer",
)
@pytest.mark.parametrize("shape", NESTED_FROM_MASK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_no_mask_check(shape, dtype):
    N, L, D = shape
    t = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # Arbitrary (possibly non-left-aligned) mask.
    mask = torch.randint(0, 2, (N, L), dtype=torch.bool, device=flag_gems.device)

    ref_t = utils.to_reference(t)
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten._nested_tensor_from_mask(ref_t, ref_mask, mask_check=False)
    res_out = flag_gems._nested_tensor_from_mask(t, mask, mask_check=False)

    _assert_nested_equal(res_out, ref_out, dtype)


@pytest.mark.nested_tensor_from_mask
def test_nested_tensor_from_mask_mask_check_raises():
    N, L, D = 4, 8, 3
    t = torch.randn(N, L, D, device=flag_gems.device)
    # A mask with a gap (False followed by True) is not left-aligned.
    mask = torch.tensor(
        [
            [True, False, True, False, False, False, False, False],
            [True, True, False, False, False, False, False, False],
            [False, False, False, False, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ],
        dtype=torch.bool,
        device=flag_gems.device,
    )
    with pytest.raises(RuntimeError):
        flag_gems._nested_tensor_from_mask(t, mask, mask_check=True)


def _grad_case(shape, dtype, device):
    """Build a left-aligned random mask and a dense source leaf tensor.

    At least one component keeps a non-empty prefix so that the native backward
    (``grad_output.to_padded_tensor(0)``) is defined; the entirely-empty nested
    case is covered separately by
    ``test_nested_tensor_from_mask_backward_all_empty``.
    """
    N, L, D = shape
    lengths = torch.randint(0, L + 1, (N,), device=device)
    lengths[torch.randint(0, N, (1,), device=device)] = torch.randint(
        1, L + 1, (1,), device=device
    )
    idx = torch.arange(L, device=device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)
    t = torch.randn(shape, dtype=dtype, device=device)
    return t, mask, lengths


def _grad_values(nested):
    """Draw the upstream gradient values shared by both graphs."""
    return torch.randn(int(nested.values().numel()), dtype=nested.values().dtype)


def _matching_grad_out(nested, values, device):
    """Build an upstream gradient nested tensor with ``nested``'s metadata.

    ``values`` is shared by the PyTorch reference and the FlagGems candidate;
    only the device differs, so ``--ref cpu`` can compare the two graphs
    without mixing devices.
    """
    sizes = nested._nested_tensor_size()
    strides = nested._nested_tensor_strides()
    offsets = nested._nested_tensor_storage_offsets()
    return torch.ops.aten._nested_view_from_buffer(
        values.to(device), sizes, strides, offsets
    )


@pytest.mark.nested_tensor_from_mask_backward
@pytest.mark.parametrize("shape", NESTED_FROM_MASK_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_backward(shape, dtype):
    N, L, D = shape
    t, mask, _ = _grad_case(shape, dtype, flag_gems.device)

    ref_t = utils.to_reference(t.detach()).requires_grad_()
    res_t = t.detach().clone().requires_grad_()
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten._nested_tensor_from_mask(ref_t, ref_mask, mask_check=True)
    res_out = flag_gems._nested_tensor_from_mask(res_t, mask, mask_check=True)

    _assert_nested_equal(res_out, ref_out, dtype)

    # The upstream gradient is the nested tensor's own value layout, matching
    # the repository benchmark's ``torch.randn_like(out)``.
    values = _grad_values(ref_out)
    ref_grad_out = _matching_grad_out(ref_out, values, ref_out.device)
    res_grad_out = _matching_grad_out(ref_out, values, res_out.device)
    (ref_grad,) = torch.autograd.grad(
        ref_out, (ref_t,), ref_grad_out, retain_graph=True
    )
    (res_grad,) = torch.autograd.grad(
        res_out, (res_t,), res_grad_out, retain_graph=True
    )

    assert res_grad.shape == ref_grad.shape
    assert res_grad.dtype == ref_grad.dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    # Padding rows are exactly zero, and only the valid prefix is populated.
    lengths = mask.sum(dim=1)
    padding = torch.arange(L, device=flag_gems.device).unsqueeze(
        0
    ) >= lengths.unsqueeze(1)
    assert torch.count_nonzero(res_grad[padding]) == 0


@pytest.mark.nested_tensor_from_mask_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_backward_no_mask_check(dtype):
    N, L, D = 4, 16, 8
    t = torch.randn((N, L, D), dtype=dtype, device=flag_gems.device)
    mask = torch.randint(0, 2, (N, L), dtype=torch.bool, device=flag_gems.device)

    ref_t = utils.to_reference(t.detach()).requires_grad_()
    res_t = t.detach().clone().requires_grad_()
    ref_mask = utils.to_reference(mask)

    ref_out = torch.ops.aten._nested_tensor_from_mask(ref_t, ref_mask, mask_check=False)
    res_out = flag_gems._nested_tensor_from_mask(res_t, mask, mask_check=False)
    _assert_nested_equal(res_out, ref_out, dtype)

    values = _grad_values(ref_out)
    ref_grad_out = _matching_grad_out(ref_out, values, ref_out.device)
    res_grad_out = _matching_grad_out(ref_out, values, res_out.device)
    (ref_grad,) = torch.autograd.grad(
        ref_out, (ref_t,), ref_grad_out, retain_graph=True
    )
    (res_grad,) = torch.autograd.grad(
        res_out, (res_t,), res_grad_out, retain_graph=True
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.nested_tensor_from_mask_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_backward_entry(dtype):
    N, L, D = 8, 32, 16
    t, mask, _ = _grad_case((N, L, D), dtype, flag_gems.device)
    ref_t = utils.to_reference(t.detach()).requires_grad_()
    res_t = t.detach().clone().requires_grad_()

    ref_out = torch.ops.aten._nested_tensor_from_mask(
        ref_t, utils.to_reference(mask), mask_check=True
    )
    res_out = flag_gems._nested_tensor_from_mask(res_t, mask, mask_check=True)

    # Attribute the gradient to the FlagGems autograd Function rather than to
    # the native NestedTensorFromMaskBackward0 node.
    assert type(res_out.grad_fn).__name__ == "NestedViewFromBufferBackward0"
    inner = res_out.grad_fn.next_functions[0][0]
    assert inner.name() == "NestedTensorFromMaskFunctionBackward"

    values = _grad_values(ref_out)
    ref_grad_out = _matching_grad_out(ref_out, values, ref_out.device)
    res_grad_out = _matching_grad_out(ref_out, values, res_out.device)
    (ref_grad,) = torch.autograd.grad(
        ref_out, (ref_t,), ref_grad_out, retain_graph=True
    )
    (res_grad,) = torch.autograd.grad(
        res_out, (res_t,), res_grad_out, retain_graph=True
    )
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.nested_tensor_from_mask_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_backward_non_contiguous(dtype):
    N, L, D = 6, 24, 8
    base = torch.randn((N, D, L), dtype=dtype, device=flag_gems.device)
    t = base.transpose(1, 2)
    assert not t.is_contiguous()
    lengths = torch.randint(1, L + 1, (N,), device=flag_gems.device)
    idx = torch.arange(L, device=flag_gems.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)

    ref_t = utils.to_reference(t.detach()).requires_grad_()
    res_t = t.detach().clone().requires_grad_()
    ref_out = torch.ops.aten._nested_tensor_from_mask(
        ref_t, utils.to_reference(mask), mask_check=True
    )
    res_out = flag_gems._nested_tensor_from_mask(res_t, mask, mask_check=True)
    _assert_nested_equal(res_out, ref_out, dtype)

    values = _grad_values(ref_out)
    ref_grad_out = _matching_grad_out(ref_out, values, ref_out.device)
    res_grad_out = _matching_grad_out(ref_out, values, res_out.device)
    (ref_grad,) = torch.autograd.grad(
        ref_out, (ref_t,), ref_grad_out, retain_graph=True
    )
    (res_grad,) = torch.autograd.grad(
        res_out, (res_t,), res_grad_out, retain_graph=True
    )
    assert res_grad.stride() == ref_grad.stride()
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.nested_tensor_from_mask_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_backward_padding(dtype):
    N, L, D = 4, 12, 5
    t, mask, lengths = _grad_case((N, L, D), dtype, flag_gems.device)
    # A component with zero valid rows must contribute an all-zero gradient.
    lengths[0] = 0
    idx = torch.arange(L, device=flag_gems.device).unsqueeze(0)
    mask = idx < lengths.unsqueeze(1)

    ref_t = utils.to_reference(t.detach()).requires_grad_()
    res_t = t.detach().clone().requires_grad_()
    ref_out = torch.ops.aten._nested_tensor_from_mask(
        ref_t, utils.to_reference(mask), mask_check=True
    )
    res_out = flag_gems._nested_tensor_from_mask(res_t, mask, mask_check=True)
    _assert_nested_equal(res_out, ref_out, dtype)

    values = _grad_values(ref_out)
    ref_grad_out = _matching_grad_out(ref_out, values, ref_out.device)
    res_grad_out = _matching_grad_out(ref_out, values, res_out.device)
    (ref_grad,) = torch.autograd.grad(
        ref_out, (ref_t,), ref_grad_out, retain_graph=True
    )
    (res_grad,) = torch.autograd.grad(
        res_out, (res_t,), res_grad_out, retain_graph=True
    )
    assert torch.count_nonzero(res_grad[0]) == 0
    utils.gems_assert_close(res_grad, ref_grad, dtype)


@pytest.mark.nested_tensor_from_mask_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_tensor_from_mask_backward_all_empty(dtype):
    """An all-False mask selects no rows.

    Native PyTorch cannot backpropagate this degenerate graph: its generated
    ``NestedTensorFromMaskBackward0`` calls ``to_padded_tensor`` on an empty
    nested tensor, which raises ``at least one constituent tensor should have
    non-zero numel``. The FlagGems backward returns the mathematically correct
    all-zero gradient, so the two behaviours are recorded separately instead of
    pretending the native call succeeds.
    """
    N, L, D = 4, 8, 3
    t = torch.randn((N, L, D), dtype=dtype, device=flag_gems.device)
    mask = torch.zeros((N, L), dtype=torch.bool, device=flag_gems.device)

    ref_t = utils.to_reference(t.detach()).requires_grad_()
    res_t = t.detach().clone().requires_grad_()
    ref_out = torch.ops.aten._nested_tensor_from_mask(
        ref_t, utils.to_reference(mask), mask_check=True
    )
    res_out = flag_gems._nested_tensor_from_mask(res_t, mask, mask_check=True)
    assert res_out._nested_tensor_size().tolist() == [[0, D]] * N

    values = _grad_values(ref_out)
    with pytest.raises(RuntimeError, match="non-zero numel"):
        torch.autograd.grad(
            ref_out, (ref_t,), _matching_grad_out(ref_out, values, ref_out.device)
        )
    (res_grad,) = torch.autograd.grad(
        res_out,
        (res_t,),
        _matching_grad_out(ref_out, values, res_out.device),
    )
    assert res_grad.shape == (N, L, D)
    assert torch.count_nonzero(res_grad) == 0
