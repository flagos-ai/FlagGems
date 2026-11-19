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

import numpy as np
import pytest
import torch
from torch.nested._internal.nested_tensor import _nt_view_dummy

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.skipif(
    cfg.TO_CPU, reason="CUDA-only op; no CPU reference implementation"
)


def _make_inputs(B, T, trailing_dims, ragged_idx, dtype, device):
    """Create a padded tensor and the offsets describing the ragged dimension."""
    # Padded shape: [B, T, *trailing_dims] with the ragged dim placed at ragged_idx.
    shape = [B] + [1] * (ragged_idx - 1) + [T] + list(trailing_dims)
    padded = torch.randn(shape, dtype=dtype, device=device)

    # Random per-batch sequence lengths in [1, T].
    np.random.seed(42)
    lengths = np.random.randint(1, T + 1, size=B)
    offsets = torch.tensor(
        [0] + np.cumsum(lengths).tolist(), dtype=torch.int32, device=device
    )
    return padded, offsets


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B,T,D", [(2, 8, 4), (8, 32, 16), (3, 5, 4)])
def test_nested_from_padded_tensor_3d(dtype, B, T, D):
    padded, offsets = _make_inputs(B, T, (D,), 1, dtype, flag_gems.device)
    dummy = _nt_view_dummy()

    ref_padded = utils.to_reference(padded)
    ref_offsets = utils.to_reference(offsets)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, ref_offsets, dummy, 1, None, None, None
    )

    res_out = flag_gems._nested_from_padded_tensor(
        padded, offsets, dummy, 1, None, None, None
    )

    assert res_out.is_nested
    assert ref_out.is_nested
    assert res_out.size() == ref_out.size()
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    utils.gems_assert_equal(res_out.offsets(), ref_out.offsets())


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B,T", [(2, 8), (8, 32), (3, 5)])
def test_nested_from_padded_tensor_2d(dtype, B, T):
    padded, offsets = _make_inputs(B, T, (), 1, dtype, flag_gems.device)
    dummy = _nt_view_dummy()

    ref_padded = utils.to_reference(padded)
    ref_offsets = utils.to_reference(offsets)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, ref_offsets, dummy, 1, None, None, None
    )

    res_out = flag_gems._nested_from_padded_tensor(
        padded, offsets, dummy, 1, None, None, None
    )

    assert res_out.is_nested
    assert ref_out.is_nested
    assert res_out.size() == ref_out.size()
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    utils.gems_assert_equal(res_out.offsets(), ref_out.offsets())


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B,T,D", [(2, 8, 4), (8, 32, 16)])
def test_nested_from_padded_tensor_sum_s(dtype, B, T, D):
    """The caller-provided ``sum_S`` path must match the reference exactly."""
    padded, offsets = _make_inputs(B, T, (D,), 1, dtype, flag_gems.device)
    dummy = _nt_view_dummy()
    sum_S = int(offsets[-1].item())

    ref_padded = utils.to_reference(padded)
    ref_offsets = utils.to_reference(offsets)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, ref_offsets, dummy, 1, None, None, sum_S
    )

    res_out = flag_gems._nested_from_padded_tensor(
        padded, offsets, dummy, 1, None, None, sum_S
    )

    assert res_out.size() == ref_out.size()
    assert res_out.values().numel() == ref_out.values().numel() == sum_S * D
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    utils.gems_assert_equal(res_out.offsets(), ref_out.offsets())


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_cache_invalidation(dtype):
    """The offsets->sum_S cache must invalidate when offsets is mutated in place.

    A first call populates the cache with the original total length. We then
    mutate ``offsets`` in place (which bumps its version counter) so that the
    last batch is shortened, and call again. The cached value must be
    discarded and the new output sized against the updated offsets;
    correctness is checked against the ATen reference.
    """
    from flag_gems.ops._nested_from_padded_tensor import _SUM_S_CACHE

    B, T, D = 4, 16, 3
    padded, offsets = _make_inputs(B, T, (D,), 1, dtype, flag_gems.device)
    dummy = _nt_view_dummy()

    # First call: populates the cache for this offsets tensor. The packed
    # values buffer is flat with shape [S, D], so S is its leading dim.
    res1 = flag_gems._nested_from_padded_tensor(padded, offsets, dummy, 1, None, None)
    assert offsets in _SUM_S_CACHE
    old_sum_s = res1.values().shape[0]

    # Mutate offsets in place so the total packed length shrinks. This bumps
    # offsets._version, which must invalidate the cached sum_S. (We shrink the
    # last batch so every batch still stays within the padded dim T.)
    new_lengths = offsets[1:].clone()
    new_lengths[-1] -= 4
    offsets[1:] = new_lengths
    assert offsets._version != _SUM_S_CACHE[offsets][0]

    res2 = flag_gems._nested_from_padded_tensor(padded, offsets, dummy, 1, None, None)
    new_sum_s = res2.values().shape[0]
    assert (
        new_sum_s == old_sum_s - 4
    ), f"cache not invalidated: old={old_sum_s}, new={new_sum_s}"

    ref_padded = utils.to_reference(padded)
    ref_offsets = utils.to_reference(offsets)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, ref_offsets, dummy, 1, None, None, None
    )
    assert res2.size() == ref_out.size()
    assert res2.values().numel() == ref_out.values().numel()
    utils.gems_assert_close(res2.values(), ref_out.values(), dtype)
    utils.gems_assert_equal(res2.offsets(), ref_out.offsets())


@pytest.mark.nested_from_padded_tensor
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_ragged_idx(dtype):
    # ragged_idx=2 on a 4D padded tensor [B, H, T, D]
    B, H, T, D = 2, 3, 4, 2
    padded = torch.randn(B, H, T, D, dtype=dtype, device=flag_gems.device)
    np.random.seed(42)
    lengths = np.random.randint(1, T + 1, size=B)
    offsets = torch.tensor(
        [0] + np.cumsum(lengths).tolist(), dtype=torch.int32, device=flag_gems.device
    )
    dummy = _nt_view_dummy()

    ref_padded = utils.to_reference(padded)
    ref_offsets = utils.to_reference(offsets)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, ref_offsets, dummy, 2, None, None, None
    )

    res_out = flag_gems._nested_from_padded_tensor(
        padded, offsets, dummy, 2, None, None, None
    )

    assert res_out.is_nested
    assert ref_out.is_nested
    assert res_out.size() == ref_out.size()
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
    utils.gems_assert_equal(res_out.offsets(), ref_out.offsets())


def _make_padded_grad_case(shape, lengths, ragged_idx, dtype, device):
    """Build a padded tensor plus ``offsets`` for a backward test."""
    padded = torch.randn(shape, dtype=dtype, device=device)
    offsets = torch.tensor(
        [0] + np.cumsum(lengths).tolist(), dtype=torch.int32, device=device
    )
    return padded, offsets


@pytest.mark.nested_from_padded_tensor_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("B,T,D", [(2, 8, 4), (8, 32, 16), (3, 5, 4)])
def test_nested_from_padded_tensor_backward_3d(dtype, B, T, D):
    """The FlagGems forward must expose a Triton-backed autograd backward."""
    lengths = [max(1, (i * 7) % T + 1) for i in range(B)]
    padded, offsets = _make_padded_grad_case(
        (B, T, D), lengths, 1, dtype, flag_gems.device
    )
    dummy = _nt_view_dummy()

    ref_padded = utils.to_reference(padded.detach()).requires_grad_(True)
    gems_padded = padded.detach().clone().requires_grad_(True)

    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, utils.to_reference(offsets), dummy, 1, None, None, None
    )
    res_out = flag_gems._nested_from_padded_tensor(
        gems_padded, offsets, dummy, 1, None, None, None
    )

    assert res_out.is_nested
    assert res_out.requires_grad
    assert type(res_out.grad_fn).__name__ == "NestedFromPaddedTensorFunctionBackward"

    torch.manual_seed(17)
    out_grad = torch.randn_like(ref_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_padded, out_grad)
    (res_grad,) = torch.autograd.grad(res_out, gems_padded, out_grad)

    assert res_grad.shape == ref_grad.shape
    assert res_grad.dtype == ref_grad.dtype
    utils.gems_assert_close(res_grad, ref_grad, dtype)

    # Padding positions must be exact zeros, matching ATen's zero fill.
    for b, length in enumerate(lengths):
        if length < T:
            assert res_grad[b, length:].abs().sum().item() == 0


@pytest.mark.nested_from_padded_tensor_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_backward_zero_length(dtype):
    """A component with zero real rows still produces a fully zero gradient."""
    B, T, D = 3, 5, 4
    lengths = [0, 2, T]
    padded, offsets = _make_padded_grad_case(
        (B, T, D), lengths, 1, dtype, flag_gems.device
    )
    dummy = _nt_view_dummy()

    ref_padded = utils.to_reference(padded.detach()).requires_grad_(True)
    gems_padded = padded.detach().clone().requires_grad_(True)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, utils.to_reference(offsets), dummy, 1, None, None, None
    )
    res_out = flag_gems._nested_from_padded_tensor(
        gems_padded, offsets, dummy, 1, None, None, None
    )

    torch.manual_seed(23)
    out_grad = torch.randn_like(ref_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_padded, out_grad)
    (res_grad,) = torch.autograd.grad(res_out, gems_padded, out_grad)

    utils.gems_assert_close(res_grad, ref_grad, dtype)
    assert res_grad[0].abs().sum().item() == 0


@pytest.mark.nested_from_padded_tensor_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_backward_ragged_idx(dtype):
    """ragged_idx=2 on a 4-D padded tensor keeps the interleaved layout."""
    B, H, T, D = 2, 3, 4, 2
    lengths = [1, T]
    padded, offsets = _make_padded_grad_case(
        (B, H, T, D), lengths, 2, dtype, flag_gems.device
    )
    dummy = _nt_view_dummy()

    ref_padded = utils.to_reference(padded.detach()).requires_grad_(True)
    gems_padded = padded.detach().clone().requires_grad_(True)
    ref_out = torch.ops.aten._nested_from_padded_tensor(
        ref_padded, utils.to_reference(offsets), dummy, 2, None, None, None
    )
    res_out = flag_gems._nested_from_padded_tensor(
        gems_padded, offsets, dummy, 2, None, None, None
    )

    torch.manual_seed(29)
    out_grad = torch.randn_like(ref_out)
    (ref_grad,) = torch.autograd.grad(ref_out, ref_padded, out_grad)
    (res_grad,) = torch.autograd.grad(res_out, gems_padded, out_grad)

    assert res_grad.shape == ref_grad.shape
    utils.gems_assert_close(res_grad, ref_grad, dtype)
    for b, length in enumerate(lengths):
        if length < T:
            assert res_grad[b, :, length:].abs().sum().item() == 0


@pytest.mark.nested_from_padded_tensor_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_nested_from_padded_tensor_backward_needs_input_grad(dtype):
    """A padded tensor that does not require grad keeps the plain forward path."""
    B, T, D = 2, 6, 3
    lengths = [4, 6]
    padded, offsets = _make_padded_grad_case(
        (B, T, D), lengths, 1, dtype, flag_gems.device
    )
    dummy = _nt_view_dummy()

    res_out = flag_gems._nested_from_padded_tensor(
        padded, offsets, dummy, 1, None, None, None
    )
    assert not res_out.requires_grad
    assert res_out.grad_fn is None

    ref_out = torch.ops.aten._nested_from_padded_tensor(
        utils.to_reference(padded),
        utils.to_reference(offsets),
        dummy,
        1,
        None,
        None,
        None,
    )
    utils.gems_assert_close(res_out.values(), ref_out.values(), dtype)
