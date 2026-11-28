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


def _reference(grad_output, indices, values, accumulate):
    """ATen formula of the ``_index_put_impl_`` backward (derivatives.yaml)."""
    if utils.TO_CPU:
        grad_output = grad_output.cpu()
        indices = [i.cpu() if i is not None else None for i in indices]
        values = values.cpu()
    if accumulate:
        grad_self = grad_output
    else:
        grad_self = torch.ops.aten.index_put(
            grad_output, indices, torch.zeros_like(values), False
        )
    # at::index ignores ``None`` slots (they only align dims); Python advanced
    # indexing would turn them into newaxis, so strip them here.
    tensor_indices = [i for i in indices if i is not None]
    grad_values = (
        grad_output[tensor_indices]
        if len(tensor_indices) > 1
        else grad_output[tensor_indices[0]]
    )
    return grad_self, grad_values


INDEX_PUT_IMPL_BWD_SHAPES = [
    # (input_shape, idx_len, accumulate)
    ((64, 64), 8, False),
    ((128, 32), 16, False),
    ((256, 64), 32, True),
    ((512, 128), 64, False),
    ((1,), 1, False),
    ((16, 32, 64), 24, False),
]


@pytest.mark.index_put_impl_backward_
@pytest.mark.parametrize("input_shape, idx_len, accumulate", INDEX_PUT_IMPL_BWD_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_put_impl_backward_functional(input_shape, idx_len, accumulate, dtype):
    """The explicit ``index_put_impl_backward_`` entry point."""
    utils.init_seed(0)
    grad_output = torch.randn(
        input_shape, dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    values = torch.randn(
        (idx_len,) + tuple(input_shape[1:]),
        dtype=dtype,
        device=flag_gems.device,
        requires_grad=False,
    )
    indices = [torch.randint(0, input_shape[0], (idx_len,), device=flag_gems.device)]
    if not accumulate:
        # exercise duplicate indices: the zero-write is idempotent
        if idx_len > 2:
            indices[0][2] = indices[0][0]

    grad_self, grad_values = flag_gems.index_put_impl_backward_(
        grad_output, None, indices, values, accumulate
    )
    ref_grad_self, ref_grad_values = _reference(
        grad_output, indices, values, accumulate
    )

    utils.gems_assert_close(grad_self, ref_grad_self, dtype)
    utils.gems_assert_close(grad_values, ref_grad_values, dtype)


@pytest.mark.index_put_impl_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_put_impl_backward_multi_index(dtype):
    """Multiple broadcast-aligned index tensors (advanced-indexing pairs)."""
    utils.init_seed(0)
    grad_output = torch.randn(
        (64, 3, 5, 7), dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    idx0 = torch.randint(0, 64, (2, 3), device=flag_gems.device)
    idx1 = torch.randint(0, 3, (2, 3), device=flag_gems.device)
    idx2 = torch.randint(0, 5, (2, 3), device=flag_gems.device)
    values = torch.randn(
        (2, 3, 7), dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    grad_self, grad_values = flag_gems.index_put_impl_backward_(
        grad_output, None, [idx0, idx1, idx2], values, False
    )
    ref_grad_self, ref_grad_values = _reference(
        grad_output, [idx0, idx1, idx2], values, False
    )

    utils.gems_assert_close(grad_self, ref_grad_self, dtype)
    utils.gems_assert_close(grad_values, ref_grad_values, dtype)


@pytest.mark.index_put_impl_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_put_impl_backward_none_slot(dtype):
    """``None`` slots in the index list (basic + advanced indexing mix)."""
    utils.init_seed(0)
    grad_output = torch.randn(
        (5, 7, 6), dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    idx = torch.tensor([1, 3], device=flag_gems.device)
    values = torch.randn(
        (2, 7, 6), dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    grad_self, grad_values = flag_gems.index_put_impl_backward_(
        grad_output, None, [idx, None, None], values, False
    )
    ref_grad_self, ref_grad_values = _reference(
        grad_output, [idx, None, None], values, False
    )

    utils.gems_assert_close(grad_self, ref_grad_self, dtype)
    utils.gems_assert_close(grad_values, ref_grad_values, dtype)


@pytest.mark.index_put_impl_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_put_impl_backward_bool_mask(dtype):
    """A boolean mask index expands to ``nonzero`` tuples like the forward."""
    utils.init_seed(0)
    grad_output = torch.randn(
        (5, 6), dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    mask = torch.tensor([True, False, True, False, True], device=flag_gems.device)
    values = torch.randn(
        (3, 6), dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    grad_self, grad_values = flag_gems.index_put_impl_backward_(
        grad_output, None, [mask], values, False
    )
    ref_grad_self, ref_grad_values = _reference(grad_output, [mask], values, False)

    utils.gems_assert_close(grad_self, ref_grad_self, dtype)
    utils.gems_assert_close(grad_values, ref_grad_values, dtype)


@pytest.mark.index_put_impl_backward_
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_put_impl_backward_negative_index(dtype):
    """Negative indices normalize like ATen (grad[-1] is the last row)."""
    utils.init_seed(0)
    grad_output = torch.randn(
        (5, 6), dtype=dtype, device=flag_gems.device, requires_grad=False
    )
    idx = torch.tensor([2, -1, 4], device=flag_gems.device)
    values = torch.randn(
        (3, 6), dtype=dtype, device=flag_gems.device, requires_grad=False
    )

    grad_self, grad_values = flag_gems.index_put_impl_backward_(
        grad_output, None, [idx], values, False
    )
    ref_grad_self, ref_grad_values = _reference(grad_output, [idx], values, False)

    utils.gems_assert_close(grad_self, ref_grad_self, dtype)
    utils.gems_assert_close(grad_values, ref_grad_values, dtype)


@pytest.mark.index_put_impl_backward_
@pytest.mark.skipif(
    utils.TO_CPU,
    reason="native autograd reference requires the CUDA index_put backward",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_index_put_impl_backward_matches_autograd_grad(dtype):
    """The FlagGems forward bridge produces the same gradients as ATen."""
    utils.init_seed(0)
    inp = torch.randn((8, 6), dtype=dtype, device=flag_gems.device, requires_grad=True)
    values = torch.randn(
        (3, 6), dtype=dtype, device=flag_gems.device, requires_grad=True
    )
    indices = [torch.tensor([2, 5, 2], device=flag_gems.device)]
    grad_out = torch.randn((8, 6), dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp.detach(), True).requires_grad_(True)
    ref_values = utils.to_reference(values.detach(), True).requires_grad_(True)
    ref_grad = utils.to_reference(grad_out, True)
    ref_out = torch.index_put(ref_inp, [utils.to_reference(indices[0])], ref_values)
    ref_grad_self, ref_grad_values = torch.autograd.grad(
        ref_out, [ref_inp, ref_values], ref_grad
    )

    res_inp = inp.detach().requires_grad_(True)
    res_values = values.detach().requires_grad_(True)
    res_out = flag_gems.index_put(res_inp, indices, res_values)
    assert res_out.grad_fn is not None
    assert "IndexPutImplBackwardFunction" in type(res_out.grad_fn).__name__
    res_grad_self, res_grad_values = torch.autograd.grad(
        res_out, [res_inp, res_values], grad_out
    )

    utils.gems_assert_close(res_grad_self, ref_grad_self, dtype)
    utils.gems_assert_close(res_grad_values, ref_grad_values, dtype)


@pytest.mark.index_put_impl_backward_
def test_index_put_impl_backward_no_index_identity():
    """Without any tensor index the forward is an identity copy."""
    utils.init_seed(0)
    grad_output = torch.randn((3, 4), device=flag_gems.device)
    values = torch.randn((), device=flag_gems.device)

    grad_self, grad_values = flag_gems.index_put_impl_backward_(
        grad_output, None, [], values, False
    )
    assert torch.equal(grad_self, grad_output)
    assert grad_values.numel() == 1
