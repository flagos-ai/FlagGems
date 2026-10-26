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
from . import test_utils as tu

# torch.ops.aten._debug_has_internal_overlap(self) -> int classifies layout:
# 0 = no internal overlap, 1 = an axis of size > 1 carrying stride 0, and
# 2 = TOO_HARD, i.e. the strides alone cannot decide whether elements alias.
# A 2 is indeterminate, not proof of actual overlap. The query reads only
# sizes/strides/storage_offset, so it never inspects element values: every dtype
# and value range is accepted and the result is a plain Python int. Nothing is
# differentiable (no backward) and the operator has no elementwise/reduction
# semantics (no broadcast grid). Layout is the only dimension that changes the
# answer, so the layout rows below carry the distinct behaviour.
_DTYPE_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.float64: utils.fp64_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float8_e4m3fn: utils.fp8_is_supported,
    torch.float8_e5m2: utils.fp8_is_supported,
}
_SUPPORTED_DTYPES = [
    dtype
    for dtype in tu.REQUIRED_DTYPES + [torch.float64, torch.bool, torch.complex64]
    if _DTYPE_FLAGS.get(dtype, True)
]
_UNIT_RANGE = ["-1", "1"]


def _make_input(dtype, shape, value_range):
    if flag_gems.vendor_name != "ascend" or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return tu.make_input(dtype, shape, value_range)
    # NPU can store FP8, but the shared factory's device-side cast cannot
    # encode it. Keep the original dtype/range recipe and transfer its bytes.
    low, high = (tu.resolve_bound(bound, dtype) for bound in value_range)
    if low == high:
        result = torch.full(shape, low, dtype=dtype, device="cpu")
    else:
        result = torch.testing.make_tensor(
            shape, dtype=dtype, device="cpu", low=low, high=high
        )
    return result.to(flag_gems.device)


def _make_special_input(dtype, scenario):
    if flag_gems.vendor_name != "ascend" or dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ):
        return tu.make_special_input(dtype, scenario)
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return (
        torch.tensor(payloads[scenario], dtype=torch.float32, device="cpu")
        .to(dtype)
        .to(flag_gems.device)
    )


def _channels_last(inp):
    if flag_gems.vendor_name != "ascend" or inp.device.type == "cpu":
        return inp.contiguous(memory_format=torch.channels_last)
    # NPU's contiguous(memory_format=channels_last) is unavailable. Construct
    # the same physical layout on CPU and transfer contiguous storage, then
    # restore its sizes, strides and offset on the original target device.
    cpu = inp.cpu().contiguous(memory_format=torch.channels_last)
    flat = torch.empty(0, dtype=cpu.dtype, device="cpu").set_(
        cpu.untyped_storage(),
        0,
        (cpu.untyped_storage().nbytes() // cpu.element_size(),),
        (1,),
    )
    storage = flat.to(inp.device).untyped_storage()
    return torch.empty(0, dtype=inp.dtype, device=inp.device).set_(
        storage, cpu.storage_offset(), cpu.size(), cpu.stride()
    )


def _input_metadata(inp):
    common = (
        inp.device,
        inp.dtype,
        inp.layout,
        inp.size(),
        inp.requires_grad,
        inp._version,
    )
    if inp.layout == torch.strided:
        storage = inp.untyped_storage()
        return common + (
            inp.stride(),
            inp.storage_offset(),
            inp.data_ptr(),
            storage._cdata,
            storage.nbytes(),
            inp.is_conj(),
            inp.is_neg(),
        )
    assert inp.layout == torch.sparse_coo
    return common + (inp.sparse_dim(), inp.dense_dim(), inp._nnz(), inp.is_coalesced())


def _storage_bytes(inp):
    # Byte observation works for FP8 and preserves NaNs and signed zero.
    # Clone the CPU snapshot too, so a CPU input cannot alias its saved bytes.
    storage = inp.untyped_storage().cpu()
    return (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(storage, 0, (storage.nbytes(),), (1,))
        .clone()
    )


def _input_snapshot(inp):
    metadata = _input_metadata(inp)
    # Retain each original storage so its identity cannot be recycled.
    if inp.layout == torch.strided:
        return metadata, _storage_bytes(inp), inp.untyped_storage()
    return (
        metadata,
        (_input_snapshot(inp._indices()), _input_snapshot(inp._values())),
        None,
    )


def _assert_input_unchanged(inp, before):
    metadata, payload, _storage_guard = before
    # Check the original device, dtype, geometry and storage identity before
    # copying values to CPU. Empty tensors still have a StorageImpl identity.
    assert _input_metadata(inp) == metadata, "query changed input metadata or storage"
    if inp.layout == torch.strided:
        assert torch.equal(_storage_bytes(inp), payload), "query changed input storage bytes"
    else:
        _assert_input_unchanged(inp._indices(), payload[0])
        _assert_input_unchanged(inp._values(), payload[1])


def _assert_same_code(res_out, ref_out):
    # The schema returns a plain Python int and the code is exact, so compare
    # the type and the value directly instead of wrapping it in a tensor.
    assert (
        type(res_out) is int
    ), f"expected a Python int layout code, got {type(res_out).__name__}"
    assert (
        res_out == ref_out
    ), f"layout code mismatch: candidate {res_out}, native {ref_out}"


# Bool input generation ignores value_range; keep one sample per workload.
# uint8's other ordinary ranges duplicate zero inputs; keep one zero and one wide range.
_DTYPE_RANGE_CASES = [
    (dtype, value_range)
    for dtype in _SUPPORTED_DTYPES
    for value_range in (
        tu.selected_ranges()[:1] if dtype == torch.bool else tu.selected_ranges()
    )
    if dtype != torch.uint8
    or value_range not in (["0", "1"], ["-1", "0"], ["min", "0"])
]


@pytest.mark.underscore_debug_has_internal_overlap
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("dtype,value_range", _DTYPE_RANGE_CASES)
def test_debug_has_internal_overlap(dtype, value_range, shape):
    inp = _make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    input_before = _input_snapshot(inp)
    ref_out = torch.ops.aten._debug_has_internal_overlap(ref_inp)
    res_out = flag_gems._debug_has_internal_overlap(inp)

    _assert_same_code(res_out, ref_out)
    _assert_input_unchanged(inp, input_before)


def _base_tensor(shape):
    if any(extent == 0 for extent in shape):
        # Zero-element tensors have no contents that could be read.
        return torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    return tu.make_input(torch.float32, shape, _UNIT_RANGE)


# Each row is one layout; bases are long enough for the as_strided rows, whose
# storage must cover offset + (size - 1) * stride + 1 elements.
_LAYOUT_ROWS = [
    ("contiguous_1d", (12,), lambda t: t),
    ("contiguous_2d", (4, 6), lambda t: t),
    ("scalar", (), lambda t: t),
    ("empty_1d", (0,), lambda t: t),
    ("empty_2d", (0, 3), lambda t: t),
    # Non-contiguous, non-overlapping views (code 0).
    ("transposed_2d", (4, 6), lambda t: t.t()),
    ("permuted_3d", (2, 3, 4), lambda t: t.permute(2, 0, 1)),
    ("narrowed_2d", (4, 6), lambda t: t.narrow(0, 2, 2)),
    (
        "channels_last_4d",
        (2, 3, 4, 4),
        _channels_last,
    ),
    # A size-1 axis is not a broadcast even with stride zero (code 0).
    ("size1_zero_stride", (8,), lambda t: torch.as_strided(t, (1, 4), (0, 1))),
    ("size1_single_element", (1,), lambda t: torch.as_strided(t, (1,), (7,))),
    # An axis of size > 1 whose stride is 0 is a broadcast (code 1).
    ("expanded_2d", (1, 4), lambda t: t.expand(3, 4)),
    ("as_strided_broadcast", (8,), lambda t: torch.as_strided(t, (3, 4), (0, 1))),
    # Strides that leave the answer indeterminate (code 2), including a view
    # whose strides genuinely make distinct elements alias the same storage.
    ("step_sliced_1d", (12,), lambda t: t[::2]),
    ("step_sliced_2d", (4, 6), lambda t: t[:, ::2]),
    ("diagonal_2d", (6, 6), lambda t: t.diagonal()),
    ("as_strided_window", (16,), lambda t: torch.as_strided(t, (2,), (2,))),
    (
        "as_strided_overlapping",
        (16,),
        lambda t: torch.as_strided(t, (4, 4), (1, 1), 1),
    ),
]

# Every row runs on a small base tensor, so the whole layout matrix stays in the
# quick smoke subset (a future large-base row could be left out of `quick`).
_LAYOUT_CASES = tu.selected_cases(
    [pytest.param(row, id=row[0]) for row in _LAYOUT_ROWS],
    quick=[pytest.param(row, id=row[0]) for row in _LAYOUT_ROWS],
)


@pytest.mark.underscore_debug_has_internal_overlap
@pytest.mark.parametrize("row", _LAYOUT_CASES)
def test_debug_has_internal_overlap_layout(row):
    _, base_shape, view_fn = row

    base = _base_tensor(base_shape)
    ref_base = tu.to_reference(base)
    inp = view_fn(base)
    ref_inp = view_fn(ref_base)

    # The query only inspects layout metadata, so snapshot both the metadata and
    # the stored values to prove it leaves the tensor it inspects untouched.
    layout_before = (inp.size(), inp.stride(), inp.storage_offset(), inp.data_ptr())
    values_before = tu.to_reference(inp)

    input_before = _input_snapshot(inp)
    ref_out = torch.ops.aten._debug_has_internal_overlap(ref_inp)
    res_out = flag_gems._debug_has_internal_overlap(inp)

    _assert_same_code(res_out, ref_out)
    _assert_input_unchanged(inp, input_before)
    assert (
        inp.size(),
        inp.stride(),
        inp.storage_offset(),
        inp.data_ptr(),
    ) == layout_before
    tu.assert_result_equal(inp, values_before)


# Sparse COO inputs are accepted by the native query and must keep the same code.
_SPARSE_SHAPES = [(4, 6), (0, 3)]


def _make_sparse(shape):
    # sparse_coo_tensor wants one index row per sparse dimension.
    nnz = 1 if all(extent > 0 for extent in shape) else 0
    indices = torch.zeros((len(shape), nnz), dtype=torch.int64, device=flag_gems.device)
    values = torch.ones(nnz, dtype=torch.float32, device=flag_gems.device)
    return torch.sparse_coo_tensor(indices, values, tuple(shape))


@pytest.mark.underscore_debug_has_internal_overlap
@pytest.mark.parametrize(
    "shape", tu.selected_cases(_SPARSE_SHAPES, quick=_SPARSE_SHAPES)
)
def test_debug_has_internal_overlap_sparse(shape):
    inp = _make_sparse(shape)
    ref_inp = tu.to_reference(inp)

    input_before = _input_snapshot(inp)
    ref_out = torch.ops.aten._debug_has_internal_overlap(ref_inp)
    res_out = flag_gems._debug_has_internal_overlap(inp)

    _assert_same_code(res_out, ref_out)
    _assert_input_unchanged(inp, input_before)


# Positive special values stay out of quick mode; the shared generator gives
# e4m3fn its nan-only scenario because that dtype cannot represent inf.
_SPECIAL_VALUE_CASES = tu.selected_cases(
    list(tu.special_value_cases(_SUPPORTED_DTYPES)),
    quick=[],
)


@pytest.mark.underscore_debug_has_internal_overlap
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_VALUE_CASES)
def test_debug_has_internal_overlap_special_values(dtype, scenario):
    inp = _make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    input_before = _input_snapshot(inp)
    ref_out = torch.ops.aten._debug_has_internal_overlap(ref_inp)
    res_out = flag_gems._debug_has_internal_overlap(inp)

    _assert_same_code(res_out, ref_out)
    _assert_input_unchanged(inp, input_before)


_INVALID_ARG_ROWS = [
    pytest.param(
        None,
        id="none",
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "hygon"
            and torch.__version__.split("+", 1)[0] == "2.4.1",
            reason="Hygon Torch 2.4.1 native query accepts None as an undefined Tensor",
        ),
    ),
    pytest.param(3.14, id="float"),
    pytest.param(1, id="int"),
    pytest.param("tensor", id="str"),
    pytest.param([1, 2], id="list"),
]
# Every negative row is kept in both modes.
_INVALID_ARG_CASES = tu.selected_cases(_INVALID_ARG_ROWS, quick=_INVALID_ARG_ROWS)


@pytest.mark.underscore_debug_has_internal_overlap
@pytest.mark.parametrize("bad_arg", _INVALID_ARG_CASES)
def test_debug_has_internal_overlap_invalid_argument(bad_arg):
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._debug_has_internal_overlap(bad_arg)


@pytest.mark.underscore_debug_has_internal_overlap
@pytest.mark.parametrize("argc", [0, 2])
def test_debug_has_internal_overlap_arity(argc):
    inp = tu.make_input(torch.float32, (4, 4), _UNIT_RANGE)
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        flag_gems._debug_has_internal_overlap(*([inp] * argc))
