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

"""Accuracy tests for ``aten::swapaxes(Tensor(a) self, int, int) -> Tensor(a)``.

``swapaxes`` is an alias of ``transpose``: a zero-copy metadata view. The tests
therefore check the view contract (shape/stride/storage-offset equality with
ATen, storage aliasing, write-through into the operand) in addition to the
element values, and cover negative axes, non-contiguous operands, scalar/empty
tensors, autograd and the ATen error messages.
"""

import logging

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# Axis pairs: first/last, middle pairs, negative axes, and the identity swap.
SWAPAXES_DIM_PAIRS = [
    (0, 1),
    (0, -1),
    (-1, -2),
    (1, 2),
    (1, -1),
    (0, 0),
]

# 0D/1D exercise the degenerate (no-op) path, 2D/3D/4D the real swaps, and the
# larger entries catch stride arithmetic that small shapes would hide.
SWAPAXES_SHAPES = [
    (),
    (5,),
    (2, 3),
    (128, 256),
    (2, 3, 4),
    (16, 128, 64),
    (2, 3, 4, 5),
    (8, 16, 32, 64),
]


def _valid_axis_cases():
    """Yield only (shape, dim_pair) combos where the pair is in range.

    0D/1D tensors accept only axes -1 and 0 (both mean the only axis); for
    rank >= 2 every pair in the list is in range.  Filtering invalid
    combinations here instead of skipping inside the test keeps the
    parametrization free of no-op cases.
    """
    for shape in SWAPAXES_SHAPES:
        ndim = len(shape)
        for dim_pair in SWAPAXES_DIM_PAIRS:
            if ndim <= 1:
                if dim_pair[0] not in (0, -1) or dim_pair[1] not in (0, -1):
                    continue
            elif not all(-ndim <= d < ndim for d in dim_pair):
                continue
            yield (shape, dim_pair)


def _check_view_metadata(res_out, inp, ref_out):
    """Assert the candidate view matches ATen and aliases the operand."""
    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out.data_ptr() == inp.data_ptr()
    assert res_out._is_view() == ref_out._is_view()


@pytest.mark.swapaxes
@pytest.mark.parametrize("shape, dim_pair", list(_valid_axis_cases()))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes(shape, dim_pair, dtype):
    axis0, axis1 = dim_pair
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    ref_out = torch.swapaxes(ref_inp, axis0, axis1)

    utils.gems_assert_close(res_out, ref_out, dtype)
    _check_view_metadata(res_out, inp, ref_out)


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes_negative_dims(dtype):
    # Full reversal via negative axes: -1/-2 and -3/-1 must normalize exactly
    # like ATen on every orbit of a 4D tensor.
    inp = torch.randn((3, 4, 5, 6), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for axis0, axis1 in [(-1, -2), (-1, -4), (-3, -1), (2, -4), (-4, 0)]:
        res_out = flag_gems.swapaxes(inp, axis0, axis1)
        ref_out = torch.swapaxes(ref_inp, axis0, axis1)
        utils.gems_assert_close(res_out, ref_out, dtype)
        _check_view_metadata(res_out, inp, ref_out)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    [
        ((2, 3, 4), 0, 2),
        ((3, 4, 5), 1, 2),
        ((8, 16, 32, 64), 0, -1),
        ((16, 7, 57, 32, 29), 1, 1),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes_non_contiguous(shape, axis0, axis1, dtype):
    # Slicing a contiguous base yields non-contiguous strides; the swap must
    # still only permute shape/stride metadata. The slice is built on both
    # devices before to_reference so both operands share one layout.
    base = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_base = utils.to_reference(base)
    inp = base[..., ::2]
    ref_inp = ref_base[..., ::2]

    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    ref_out = torch.swapaxes(ref_inp, axis0, axis1)

    utils.gems_assert_close(res_out, ref_out, dtype)
    _check_view_metadata(res_out, inp, ref_out)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    [
        # 0-D has no axes to swap: every in-range pair is a no-op view.
        ((), 0, 0),
        ((), -1, 0),
        ((), 0, -1),
        # 1-D: only axis 0/-1 exist, so any in-range pair is a no-op view.
        ((7,), 0, 0),
        ((7,), -1, 0),
        # Empty tensors keep their (possibly degenerate) strides.
        ((0,), 0, 0),
        ((5, 0, 3), 0, 1),
        ((5, 0, 3), 0, -1),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes_scalar_and_empty(shape, axis0, axis1, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    res_out = flag_gems.swapaxes(inp, axis0, axis1)
    ref_out = torch.swapaxes(ref_inp, axis0, axis1)

    assert res_out.shape == ref_out.shape
    assert res_out.stride() == ref_out.stride()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()
    assert res_out._is_view() == ref_out._is_view()
    if ref_out.numel() > 0:
        utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes_storage_offset_preserved(dtype):
    # A view into the middle of a larger allocation must keep its storage
    # offset: the swap carries the offset over unchanged rather than
    # recomputing it from a contiguous buffer.
    base = torch.randn((9, 8, 7), dtype=dtype, device=flag_gems.device)
    inp = base[2:7]
    offset = inp.storage_offset()
    assert offset != 0

    res_out = flag_gems.swapaxes(inp, 0, 2)
    # Compare metadata against same-device ATen: to_reference may copy across
    # devices, which resets the offset on the reference operand.
    aten_out = torch.swapaxes(inp, 0, 2)

    assert res_out.storage_offset() == aten_out.storage_offset() == offset
    assert res_out.stride() == aten_out.stride()
    assert res_out.untyped_storage().data_ptr() == inp.untyped_storage().data_ptr()

    ref_inp = utils.to_reference(inp)
    ref_out = torch.swapaxes(ref_inp, 0, 2)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.swapaxes
@pytest.mark.parametrize("shape", [(4, 5), (2, 3, 4), (8, 16, 32, 64)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes_shares_storage(shape, dtype):
    # Writing through the returned view must land in the operand's storage,
    # while the operand's own shape/stride/offset stay untouched.
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_shape, inp_stride, inp_offset = (
        tuple(inp.shape),
        inp.stride(),
        inp.storage_offset(),
    )
    snapshot = inp.clone()
    ref_inp = utils.to_reference(snapshot)

    res_out = flag_gems.swapaxes(inp, 0, -1)
    ref_out = torch.swapaxes(ref_inp, 0, -1)

    # Fill through the view; the base must reflect it exactly like ATen.
    res_out.fill_(1.0)
    ref_out.fill_(1.0)
    utils.gems_assert_close(inp, ref_inp, dtype)

    assert tuple(inp.shape) == inp_shape
    assert inp.stride() == inp_stride
    assert inp.storage_offset() == inp_offset


@pytest.mark.swapaxes
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_swapaxes_autograd(dtype):
    # swapaxes is a differentiable view: the gradient of the swapped output
    # must be routed back through the inverse swap.
    shape = (4, 5, 6)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)
    ref_inp = utils.to_reference(inp.detach().clone()).requires_grad_(True)

    res_out = flag_gems.swapaxes(inp, 0, 2)
    grad = torch.randn(res_out.shape, dtype=dtype, device=flag_gems.device)
    res_out.backward(grad)

    ref_out = torch.swapaxes(ref_inp, 0, 2)
    ref_out.backward(utils.to_reference(grad))

    assert inp.grad is not None and inp.grad.shape == shape
    utils.gems_assert_close(inp.grad, ref_inp.grad, dtype)


@pytest.mark.swapaxes
@pytest.mark.parametrize(
    "shape,axis0,axis1",
    [
        ((2, 3), 0, 5),  # axis1 out of range
        ((2, 3), -3, 0),  # axis0 too negative
        ((2, 3, 4), 0, 3),  # axis1 out of range for 3D
        ((2, 3, 4), -4, 0),  # axis0 too negative for 3D
        ((5,), 0, 1),  # 1D accepts only 0/-1
        ((5,), 1, 0),
        ((), 0, 1),  # 0D accepts only 0/-1
        ((), -2, 0),
    ],
)
def test_swapaxes_invalid_axes(shape, axis0, axis1):
    # The error type and message must match ATen exactly.
    inp = torch.randn(shape, device=flag_gems.device)
    with pytest.raises(IndexError) as ref_exc_info:
        torch.swapaxes(inp, axis0, axis1)
    with pytest.raises(IndexError) as res_exc_info:
        flag_gems.swapaxes(inp, axis0, axis1)
    assert str(res_exc_info.value) == str(ref_exc_info.value)


@pytest.mark.swapaxes
def test_swapaxes_dispatch_reaches_gems_impl(caplog):
    """``aten::swapaxes`` must actually run the FlagGems implementation.

    ``aten::swapaxes`` is registered as CompositeImplicitAutograd, unlike
    ``aten::transpose.int`` (CompositeExplicitAutograd), so it is not obvious
    that the plain device-key registration in ``_FULL_CONFIG`` intercepts it
    rather than letting the native decomposition run. Every other test in this
    file calls ``flag_gems.swapaxes`` directly and therefore passes whether or
    not registration works, so the dispatch path is verified here: register via
    the same registrar ``use_gems()`` uses and assert the impl's own log line
    fires for a ``torch.swapaxes`` call.
    """
    inp = torch.randn((3, 4, 5), dtype=torch.float32, device=flag_gems.device)
    lib = torch.library.Library("aten", "IMPL")

    caplog.set_level(logging.DEBUG, logger="flag_gems.ops.swapaxes")
    try:
        flag_gems.only_enable(lib=lib, include=["swapaxes"])
        assert "swapaxes" in flag_gems.all_registered_keys()

        res_out = torch.swapaxes(inp, 0, 2)

        assert "GEMS SWAPAXES" in caplog.text
    finally:
        lib._destroy()

    # The registered kernel is the FlagGems view, not a copy.
    ref_out = torch.swapaxes(inp, 0, 2)
    assert tuple(res_out.shape) == (5, 4, 3)
    assert res_out.stride() == ref_out.stride()
    assert res_out.data_ptr() == inp.data_ptr()
