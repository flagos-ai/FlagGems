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

"""Correctness tests for ``aten::set_data``.

``aten::set_data(Tensor(a!) self, Tensor new_data) -> ()`` returns ``None`` and
rebinds the target object onto the donor's type metadata and storage instead of
computing a tensor. The tests assert the adopted metadata and alias geometry, the
target's own autograd state, the integrity of the donor and of the target's
previous backing storage, and the native argument-type rejections. There is no
broadcast dimension and no scalar-operand overload: ``new_data`` must be a plain
dense Tensor.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Optional dtypes follow the active backend's capability flags, so a backend
# without an fp64/bf16/int64/fp8 path does not collect those cases. float16,
# int8, uint8, int32, float32 and complex64 have no capability flag in this tree
# and are always collected.
_OPTIONAL_DTYPE_FLAGS = {
    torch.float64: "support_fp64",
    torch.bfloat16: "support_bf16",
    torch.int64: "support_int64",
    torch.float8_e4m3fn: "support_fp8",
    torch.float8_e5m2: "support_fp8",
}


def _dtype_supported(dtype):
    flag = _OPTIONAL_DTYPE_FLAGS.get(dtype)
    if flag is None:
        return True
    return bool(getattr(flag_gems.runtime.device, flag, False))


def _supported(dtypes):
    return [dtype for dtype in dtypes if _dtype_supported(dtype)]


SUPPORTED_DTYPES = _supported(
    tu.REQUIRED_DTYPES + [torch.bool, torch.float64, torch.complex64]
)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _tensor_state(tensor):
    """The metadata and storage identity that set_data transplants."""
    return (
        tensor.dtype,
        tensor.layout,
        tensor.device,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.data_ptr(),
        tensor.untyped_storage().data_ptr(),
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _assert_replaced(target, source):
    assert _tensor_state(target) == _tensor_state(source)


def _donor_snapshot(source):
    """Donor state and values captured before the swap."""
    return _tensor_state(source), tu.to_reference(
        source.detach().resolve_conj().resolve_neg()
    )


def _assert_donor_intact(source, snapshot):
    state_before, values_before = snapshot
    assert _tensor_state(source) == state_before
    tu.assert_result_equal(source.resolve_conj().resolve_neg(), values_before)


def _as_view(storage, kind):
    """Materialize one of the target/donor view kinds from a fresh storage."""
    if kind in ("", "as_is"):
        return storage
    if kind == "transposed":
        return storage.transpose(0, 1)
    if kind == "stepped":
        return storage[:, ::2]
    if kind == "window":
        return storage[1:3, 2:6]
    if kind == "expanded":
        return storage.expand(5, storage.shape[1])
    return storage[::2, ::3].permute(1, 0)  # "permuted_slice"


def _autograd_state(tensor):
    return (
        tensor.requires_grad,
        tensor.is_leaf,
        type(tensor.grad_fn).__name__,
        tensor._version,
    )


def _attach_autograd(base, kind):
    """Target for ``kind``: plain leaf, grad-tracking leaf or non-leaf."""
    if kind == "leaf":
        return base
    if kind == "leaf_grad":
        return base.requires_grad_(True)
    return base.requires_grad_(True) * 2.0


def _grad_leaf(base, kind):
    return base if kind == "leaf_grad" else base.requires_grad_(True)


def _case_id(case):
    target_storage, target_view, source_storage, source_view = case
    return "%s-%s-%s-%s" % (
        target_storage,
        target_view or "flat",
        source_storage,
        source_view or "flat",
    )


# ---------------------------------------------------------------------------
# Case tables
# ---------------------------------------------------------------------------

# (target storage, target view, donor storage, donor view). Rows walk the
# rank/element-count changes (the target adopts the donor's shape), the empty
# tensors, and donors that are non-contiguous views, so the target has to adopt
# real strides and a nonzero storage offset instead of fresh contiguous
# metadata. Every cheap row runs in both levels; the four large rows stay in the
# default suite, where the quick subset would only repeat the same semantics at
# a much higher memory cost.
_LAYOUT_CHEAP_CASES = [
    ((), "", (), ""),
    ((), "", (100,), ""),
    ((100,), "", (), ""),
    ((256,), "", (4, 4, 4, 4), ""),
    ((4, 4, 4, 4), "", (256,), ""),
    ((3, 0), "", (0,), ""),
    ((0,), "", (3, 0), ""),
    ((2, 3), "", (4, 6), "as_is"),
    ((2, 3), "", (4, 6), "transposed"),
    ((2, 2), "", (4, 6), "stepped"),
    ((3, 4), "", (8, 8), "window"),
    ((), "", (1, 6), "expanded"),
    ((2, 2), "", (4, 6), "permuted_slice"),
    ((8, 8), "window", (6, 6), "transposed"),
    ((1, 6), "expanded", (8, 8), "window"),
    ((6, 8), "transposed", (4, 9), "stepped"),
]
_LAYOUT_LARGE_CASES = [
    ((1024, 1024), "", (20, 320, 15), ""),
    ((20, 320, 15), "", (1024, 1024), ""),
    ((16, 128, 64, 60), "", (16, 7, 57, 32, 29), ""),
    ((16, 7, 57, 32, 29), "", (16, 128, 64, 60), ""),
]
_LAYOUT_CASES = tu.selected_cases(
    _LAYOUT_CHEAP_CASES + _LAYOUT_LARGE_CASES, quick=_LAYOUT_CHEAP_CASES
)
_LAYOUT_DTYPES = _supported(
    [torch.float32, torch.float16, torch.int64, torch.float8_e4m3fn]
)

# set_data transplants type metadata, so the donor may have a different dtype
# without converting the donor values.
_CROSS_DTYPE_CASES = [
    (target, source)
    for target, source in [
        (torch.float32, torch.int64),
        (torch.int64, torch.float32),
        (torch.float64, torch.int8),
        (torch.int8, torch.float64),
        (torch.float16, torch.float32),
        (torch.float32, torch.float8_e4m3fn),
        (torch.bfloat16, torch.uint8),
        (torch.bool, torch.float32),
        (torch.float32, torch.bool),
        (torch.complex64, torch.float32),
        (torch.float64, torch.complex64),
        (torch.uint8, torch.int32),
        (torch.int32, torch.int64),
        (torch.float8_e5m2, torch.float16),
    ]
    if _dtype_supported(target) and _dtype_supported(source)
]

# Lazy conjugate/negative bits are metadata on the tensor object and move with
# the storage; "plain" checks that a donor without a bit cannot make the target
# acquire one.
_LAZY_FLAG_CASES = [
    (torch.float32, "neg"),
    (torch.float32, "plain"),
    (torch.complex64, "conj"),
    (torch.complex64, "conj_neg"),
]
_AUTOGRAD_DTYPES = _supported(
    [torch.float32, torch.float16, torch.bfloat16, torch.float64]
)
_AUTOGRAD_KINDS = ["leaf", "leaf_grad", "nonleaf"]

# The non-leaf donor keeps the target's pre-swap shape: its surviving grad_fn was
# built for that shape and native backward rejects a mismatch (RuntimeError from
# MulBackward0: "got [4, 3] but expected shape compatible with [4, 3]"). A leaf
# target may change shape because its gradient follows the new metadata.
_BACKWARD_CASES = tu.selected_cases(
    [("leaf_grad", (4, 3), (3, 5)), ("nonleaf", (4, 3), (4, 3))], quick=[]
)

# nan / inf / mixed donors. e4m3fn carries only nan because the format has no
# infinity representation, which tu.special_value_cases encodes.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(
        _supported(
            [
                torch.float32,
                torch.float64,
                torch.float16,
                torch.bfloat16,
                torch.float8_e4m3fn,
                torch.float8_e5m2,
            ]
        )
    ),
    quick=[],
)

_ALIAS_DTYPES = _supported(
    [torch.float32, torch.float16, torch.int64, torch.float8_e4m3fn]
)

# Non-Tensor arguments and incompatible tensor kinds fail through two distinct
# native error paths, so they are separate rows here as well.
_INVALID_SOURCE_KINDS = [
    "scalar_float",
    "scalar_int",
    "string",
    "none",
    "list",
    "sparse",
    "quantized",
    "meta",
]
_GRAD_TARGET_SOURCE_DTYPES = _supported(
    [torch.int8, torch.uint8, torch.int32, torch.int64, torch.bool]
)


# ---------------------------------------------------------------------------
# Value ranges and shapes
# ---------------------------------------------------------------------------


@pytest.mark.set_data
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
def test_set_data_value_ranges(dtype, shape, value_range):
    target = tu.make_input(dtype, shape, value_range)
    source = tu.make_input(dtype, shape, value_range)
    ref_target = tu.to_reference(target)
    ref_source = tu.to_reference(source)
    donor_snapshot = _donor_snapshot(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    assert tuple(target.shape) == tuple(ref_target.shape)
    tu.assert_result_equal(target, ref_target)
    _assert_donor_intact(source, donor_snapshot)


@pytest.mark.set_data
@pytest.mark.parametrize("case", _LAYOUT_CASES, ids=_case_id)
@pytest.mark.parametrize("dtype", _LAYOUT_DTYPES)
def test_set_data_layout_cases(dtype, case):
    target_storage, target_view, source_storage, source_view = case
    target = _as_view(tu.make_input(dtype, target_storage, ["-1", "1"]), target_view)
    source = _as_view(tu.make_input(dtype, source_storage, ["-1", "1"]), source_view)
    ref_target = tu.to_reference(target)
    ref_source = tu.to_reference(source)
    donor_snapshot = _donor_snapshot(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    assert target.shape == ref_target.shape
    assert target.stride() == ref_target.stride()
    assert target.storage_offset() == ref_target.storage_offset()
    tu.assert_result_equal(target, ref_target)
    _assert_donor_intact(source, donor_snapshot)


@pytest.mark.set_data
@pytest.mark.parametrize("target_dtype,source_dtype", _CROSS_DTYPE_CASES)
def test_set_data_cross_dtype(target_dtype, source_dtype):
    target = tu.make_input(target_dtype, (4, 5), ["-1", "1"])
    source = tu.make_input(source_dtype, (3, 2), ["-1", "1"])
    ref_target = tu.to_reference(target)
    ref_source = tu.to_reference(source)
    donor_snapshot = _donor_snapshot(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    assert target.dtype == source_dtype
    assert target.dtype == ref_target.dtype
    tu.assert_result_equal(target, ref_target)
    _assert_donor_intact(source, donor_snapshot)


def _lazy_source(dtype, shape, kind):
    storage = tu.make_input(dtype, (shape[0] + 1, shape[1] + 2), ["-1", "1"])
    donor = storage[1:, : shape[1]]  # non-contiguous, nonzero storage offset
    if kind == "conj":
        return donor.conj()
    if kind == "neg":
        return torch._neg_view(donor)
    if kind == "conj_neg":
        return torch._neg_view(donor.conj())
    return donor


@pytest.mark.set_data
@pytest.mark.parametrize("dtype,kind", _LAZY_FLAG_CASES)
def test_set_data_transplants_lazy_bits(dtype, kind):
    target = tu.make_input(dtype, (2, 5), ["-1", "1"])
    source = _lazy_source(dtype, (3, 4), kind)
    ref_target = tu.to_reference(target)
    ref_source = tu.to_reference(source)
    donor_snapshot = _donor_snapshot(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    # The lazy bits are raw tensor state, so they are compared before resolving
    # either side; the values need the combined resolution on both sides.
    assert target.is_conj() == ref_target.is_conj()
    assert target.is_neg() == ref_target.is_neg()
    tu.assert_result_equal(
        target.resolve_conj().resolve_neg(), ref_target.resolve_conj().resolve_neg()
    )
    _assert_donor_intact(source, donor_snapshot)


@pytest.mark.set_data
@pytest.mark.parametrize("kind", _AUTOGRAD_KINDS)
@pytest.mark.parametrize("dtype", _AUTOGRAD_DTYPES)
def test_set_data_preserves_target_autograd_state(dtype, kind):
    base = tu.make_input(dtype, (4, 3), ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    target = _attach_autograd(base, kind)
    ref_target = _attach_autograd(ref_base, kind)
    source = tu.make_input(dtype, (3, 5), ["-1", "1"])
    ref_source = tu.to_reference(source)
    state_before = _autograd_state(target)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    tu.assert_result_equal(target, ref_target)
    # A metadata swap is not a data write: the version counter, the leaf flag and
    # any surviving grad_fn must match both the reference and the pre-swap state.
    assert _autograd_state(target) == _autograd_state(ref_target)
    assert _autograd_state(target) == state_before


@pytest.mark.set_data
@pytest.mark.parametrize("kind,target_shape,source_shape", _BACKWARD_CASES)
@pytest.mark.parametrize("dtype", _AUTOGRAD_DTYPES)
def test_set_data_backward_through_original_leaf(
    dtype, kind, target_shape, source_shape
):
    base = tu.make_input(dtype, target_shape, ["-1", "1"])
    ref_base = tu.to_reference(base.detach())
    target = _attach_autograd(base, kind)
    ref_target = _attach_autograd(ref_base, kind)
    leaf = _grad_leaf(base, kind)
    ref_leaf = _grad_leaf(ref_base, kind)
    source = tu.make_input(dtype, source_shape, ["-1", "1"])
    ref_source = tu.to_reference(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    # Differentiating through the original leaf (not through the target object)
    # keeps the surviving graph honest: the non-leaf row can only produce the
    # gradient its MulBackward0 was built to emit.
    (res_grad,) = torch.autograd.grad((target * 3.0).sum(), leaf)
    (ref_grad,) = torch.autograd.grad((ref_target * 3.0).sum(), ref_leaf)
    assert res_grad.dtype == ref_grad.dtype
    assert res_grad.shape == ref_grad.shape
    tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.set_data
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_set_data_special_values(dtype, scenario):
    source = tu.make_special_input(dtype, scenario)
    target = tu.make_input(dtype, (1,), ["-1", "1"])
    ref_target = tu.to_reference(target)
    ref_source = tu.to_reference(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    _assert_replaced(target, source)
    tu.assert_result_equal(target, ref_target)
    # nan / +-inf must arrive unchanged, with matching nan positions.
    tu.assert_result_equal(target, ref_source)


@pytest.mark.set_data
@pytest.mark.parametrize("dtype", _ALIAS_DTYPES)
def test_set_data_rebinds_target_alias(dtype):
    buffer = tu.make_input(dtype, (4, 8), ["-1", "1"])
    target = buffer[1:3, 2:6]  # shape (2, 4), strides (8, 1), offset 10
    old_alias = target.detach()  # independent view of the pre-swap window
    old_ptr = target.data_ptr()
    buffer_before = tu.to_reference(buffer)
    old_alias_before = tu.to_reference(old_alias)
    donor = tu.make_input(dtype, (6, 6), ["-1", "1"])
    source = donor[1:4, ::2]  # shape (3, 3), strides (6, 2), offset 6
    donor_snapshot = _donor_snapshot(source)
    donor_before = tu.to_reference(donor)
    ref_target = tu.to_reference(target)
    ref_source = tu.to_reference(source)

    torch.ops.aten.set_data(ref_target, ref_source)
    res_ret = flag_gems.set_data(target, source)

    assert res_ret is None
    # Forward values are compared before any write touches the shared storage.
    _assert_replaced(target, source)
    tu.assert_result_equal(target, ref_target)
    _assert_donor_intact(source, donor_snapshot)
    assert target.data_ptr() != old_ptr
    # The fill_ below must not reach the window's previous backing storage.
    tu.assert_result_equal(buffer, buffer_before)
    tu.assert_result_equal(old_alias, old_alias_before)
    tu.assert_result_equal(donor, donor_before)

    # The swap aliases the donor storage: a write through the target is visible
    # in the donor window, which a value copy would not be.
    target.fill_(0)
    tu.assert_result_equal(source, tu.to_reference(torch.zeros_like(source)))
    tu.assert_result_equal(buffer, buffer_before)
    tu.assert_result_equal(old_alias, old_alias_before)
    donor_before[1:4, ::2].fill_(0)
    tu.assert_result_equal(donor, donor_before)


def _invalid_source(kind, template):
    if kind == "scalar_float":
        return 1.0
    if kind == "scalar_int":
        return 3
    if kind == "string":
        return "tensor"
    if kind == "none":
        return None
    if kind == "list":
        return [1.0, 2.0]
    if kind == "sparse":
        indices = torch.tensor([[0, 1], [1, 2]], device=template.device)
        values = torch.tensor([1.0, 2.0], device=template.device)
        return torch.sparse_coo_tensor(indices, values, (3, 3))
    if kind == "quantized":
        return torch.quantize_per_tensor(template.detach().cpu(), 0.1, 5, torch.quint8)
    return torch.empty(template.shape, dtype=template.dtype, device="meta")


@pytest.mark.set_data
@pytest.mark.parametrize("kind", _INVALID_SOURCE_KINDS)
def test_set_data_rejects_invalid_source(kind):
    target = tu.make_input(torch.float32, (2, 3), ["-1", "1"])

    # new_data must be a dense strided Tensor: any other python or tensor kind is
    # rejected natively (RuntimeError) and must be rejected here too.
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.set_data(target, _invalid_source(kind, target))


@pytest.mark.set_data
@pytest.mark.parametrize("source_dtype", _GRAD_TARGET_SOURCE_DTYPES)
def test_set_data_rejects_nonfloat_source_for_grad_target(source_dtype):
    target = tu.make_input(torch.float32, (2, 3), ["-1", "1"]).requires_grad_(True)
    source = tu.make_input(source_dtype, (2, 3), ["-1", "1"])

    # A grad-tracking target may only be rebound to a floating/complex donor.
    with pytest.raises(RuntimeError):
        flag_gems.set_data(target, source)
