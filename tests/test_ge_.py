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

"""Correctness tests for ``aten::ge_`` (in-place ``self >= other``)."""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from .conftest import TO_CPU

_CAPABILITY_FLAGS = {
    torch.bfloat16: utils.bf16_is_supported,
    torch.int64: utils.int64_is_supported,
    torch.float64: utils.fp64_is_supported,
}

DTYPES = [
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.bool,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    )
    if _CAPABILITY_FLAGS.get(dtype, True)
]

FLOAT_DTYPES = [dtype for dtype in DTYPES if dtype.is_floating_point]

# bf16 is the spec's preferred broadcast dtype, fp16 the supported fallback.
BROADCAST_DTYPE = torch.bfloat16 if torch.bfloat16 in DTYPES else torch.float16


# ---------------------------------------------------------------------------
# Shared value-range / shape helpers (mirrors the operator-test spec
# conventions; the full sets keep the comparison grid dense across ranks,
# value ranges and dtypes).
# ---------------------------------------------------------------------------
# Full set: ranks 0 through 5; extents chosen to exercise dense, transposed,
# sliced and expanded access patterns of an in-place elementwise write.
REQUIRED_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]

# Value ranges: [-1,1], [0,1], [-1,0], [0,max], [min,0]. Bounds resolve per
# dtype, so unsigned dtypes clamp a negative bound to their representable low.
REQUIRED_RANGES = [
    ["-1", "1"],
    ["0", "1"],
    ["-1", "0"],
    ["0", "max"],
    ["min", "0"],
]


def _resolve_bound(symbol, dtype):
    if dtype == torch.bool:
        low, high = 0, 1
    elif dtype.is_floating_point:
        low, high = torch.finfo(dtype).min, torch.finfo(dtype).max
    else:
        low, high = torch.iinfo(dtype).min, torch.iinfo(dtype).max
    table = {
        "-1": -1.0,
        "0": 0.0,
        "1": 1.0,
        "max": high,
        "min": low,
        "max/2": high / 2,
        "min/2": low / 2,
    }
    return table[symbol]


def make_input(dtype, shape, value_range):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    Bounds are clamped to the dtype's representable range first, so a range a
    dtype cannot represent (e.g. uint8 and ``[-1, 0]``) degenerates to a
    constant fill instead of raising.
    """
    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    if not (dtype.is_floating_point or dtype.is_complex):
        low, high = int(low), int(high)
        info = torch.iinfo(dtype)
        low = max(low, int(info.min))
        high = min(high, int(info.max))
        low = min(low, high)

    if low == high:
        return torch.full(shape, low, device=flag_gems.device, dtype=dtype)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def make_special_input(dtype, scenario):
    payloads = {
        "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
        "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
        "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
    }
    return torch.tensor(
        payloads[scenario], device=flag_gems.device, dtype=torch.float32
    ).to(dtype)


def to_reference(inp, upcast=False):
    """Independent tensor oracle for these tests, retaining view metadata.

    A plain clone can compact strides, reset offsets or materialize lazy
    flags, so the dense storage is copied with the original metadata instead.
    """
    if inp is None:
        return None
    if inp.layout != torch.strided or inp.is_quantized or inp.is_nested:
        reference = inp.detach().clone()
    else:
        reference = torch.empty(0, dtype=inp.dtype, device=inp.device).set_(
            inp.untyped_storage().clone(),
            inp.storage_offset(),
            inp.size(),
            inp.stride(),
        )
        if inp.is_conj():
            reference = reference.conj()
    # In quick-cpu mode the reference runs on CPU and the assert helpers
    # require both sides on the same device. .to() preserves size, stride
    # and storage offset, so the view metadata survives the move.
    if TO_CPU:
        reference = reference.to("cpu")
    reference.requires_grad_(inp.requires_grad)
    return reference


def assert_result_equal(result, reference):
    # Comparison results are exact 0/1 values; equal_nan keeps the NaN payload
    # rows comparable. gem-agnostic: moves res to the reference device in
    # quick-cpu mode via utils.gems_assert_equal's TO_CPU handling.
    utils.gems_assert_equal(result, reference, equal_nan=True)


def assert_result_close(result, reference, *, atol=1e-4):
    utils.gems_assert_close(result, reference, dtype=result.dtype, atol=atol)


def special_value_cases(dtypes):
    """Representable special scenarios, kept separate in collected case IDs."""
    cases = []
    for dtype in dtypes:
        if not dtype.is_floating_point:
            continue
        cases.append((dtype, "nan"))
        if dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ):
            cases.extend((dtype, kind) for kind in ("inf", "mixed"))
    return cases


# (receiver shape, other shape, receiver dtype, other dtype, value range).
_GRID_CASES = [
    (shape, shape, dtype, dtype, value_range)
    for dtype in DTYPES
    for shape in REQUIRED_SHAPES
    for value_range in REQUIRED_RANGES
]

_EXTRA_CASES = [
    # 1-dim and multi-dim broadcast; ge_ writes into its receiver, so the
    # receiver carries the full shape.
    ((20, 320, 15), (15,), BROADCAST_DTYPE, BROADCAST_DTYPE, ["-1", "1"]),
    ((16, 128, 64, 60), (16, 1, 1, 60), torch.int32, torch.int32, ["-1", "1"]),
    # Mixed dtypes compare promoted values and store 0/1 in the receiver dtype.
    ((1024, 1024), (1024, 1024), torch.int8, torch.float32, ["-1", "1"]),
    ((20, 320, 15), (20, 320, 15), torch.int32, torch.float64, ["-1", "1"]),
    ((16, 128, 64, 60), (16, 128, 64, 60), torch.float32, torch.int32, ["-1", "1"]),
    (
        (16, 7, 57, 32, 29),
        (16, 7, 57, 32, 29),
        torch.float16,
        torch.int32,
        ["-1", "1"],
    ),
]

TENSOR_CASES = _GRID_CASES + [
    row for row in _EXTRA_CASES if row[2] in DTYPES and row[3] in DTYPES
]

# Scalar sweep: no schema default, so zero, both signs, a fractional value and
# the float nan/inf boundaries.
_FLOAT_SCALARS = (
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    float("inf"),
    float("-inf"),
    float("nan"),
)
_INT_SCALARS = (0, 1, -1)

SCALAR_CASES = [
    (dtype, scalar)
    for dtype in DTYPES
    for scalar in (
        _FLOAT_SCALARS
        if dtype.is_floating_point
        else ((0, 1) if dtype == torch.bool else _INT_SCALARS)
    )
]

BACKWARD_CASES = [
    (dtype, shape, ["-1", "1"])
    for dtype in FLOAT_DTYPES
    for shape in ((256,), (20, 320, 15))
]

SPECIAL_CASES = special_value_cases(FLOAT_DTYPES)

# The operand is the receiver as well, so every element compares with itself.
ALIAS_CASES = [
    (dtype, shape, ["-1", "1"])
    for dtype in DTYPES
    if dtype in (torch.float32, torch.int32, torch.float16, torch.int8)
    for shape in ((20, 320, 15), (1024, 1024))
]

# Views of a larger storage: "transposed" is non-contiguous, "offset" starts at
# storage offset 16 * 32.
_VIEW_KINDS = {
    "transposed": ((8, 16, 32), (16, 8, 32)),
    "offset": ((4, 16, 32), (3, 16, 32)),
}

VIEW_CASES = [
    (dtype, kind)
    for dtype in (torch.float32, torch.bfloat16)
    if dtype in DTYPES
    for kind in _VIEW_KINDS
]

# The operand neither broadcasts up to the receiver nor yields an in-place
# output of the receiver's shape, so the native op rejects it.
NEGATIVE_CASES = [
    ((2, 19, 7), (5,)),
    ((2, 19, 7), (3, 2, 19, 7)),
]

# Dtypes measured to have no native comparison kernel on the vendor's backend
# ("compare_cuda" not implemented for float8 / complex); measured once during
# generation, never probed here so collection stays metadata-only. Unmeasured
# vendors get no rejection cases.
_COMPARE_REJECTED_DTYPES = {
    "nvidia": (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        torch.complex64,
    ),
}

REJECTED_DTYPES = [
    dtype
    for dtype in _COMPARE_REJECTED_DTYPES.get(flag_gems.vendor_name, ())
    if dtype is not None
]


def _view_of(storage, kind):
    return storage.transpose(0, 1) if kind == "transposed" else storage[1:]


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize(
    "shape,other_shape,dtype,other_dtype,value_range", TENSOR_CASES
)
def test_ge__tensor(shape, other_shape, dtype, other_dtype, value_range):
    inp = make_input(dtype, shape, value_range)
    other = make_input(other_dtype, other_shape, value_range)

    ref_inp = to_reference(inp)
    ref_other = to_reference(other)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_other)
    res_out = flag_gems.ge_tensor_(res_inp, other)

    assert_result_equal(res_out, ref_out)
    assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp
    assert_result_equal(other, ref_other)


@pytest.mark.ge_
@pytest.mark.ge_scalar_
@pytest.mark.parametrize("dtype,scalar", SCALAR_CASES)
def test_ge__scalar(dtype, scalar):
    inp = make_input(dtype, (1024, 1024), ["-1", "1"])

    ref_inp = to_reference(inp)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Scalar(ref_inp, scalar)
    res_out = flag_gems.ge_scalar_(res_inp, scalar)

    assert_result_equal(res_out, ref_out)
    assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize("dtype,shape,value_range", BACKWARD_CASES)
def test_ge__backward(dtype, shape, value_range):
    inp = make_input(dtype, shape, value_range)
    other = make_input(dtype, shape, value_range)

    ref_inp = to_reference(inp)
    ref_other = to_reference(other)

    # autograd rejects an in-place op on a grad-requiring leaf, so the receiver
    # is a non-leaf.
    ref_base = ref_inp.detach().clone().requires_grad_(True)
    ref_receiver = ref_base * 2.0
    ref_out = torch.ops.aten.ge_.Tensor(ref_receiver, ref_other)
    (ref_grad,) = torch.autograd.grad(
        ref_out, ref_base, grad_outputs=torch.ones_like(ref_out)
    )

    res_base = inp.detach().clone().requires_grad_(True)
    res_receiver = res_base * 2.0
    res_out = flag_gems.ge_tensor_(res_receiver, other)
    (res_grad,) = torch.autograd.grad(
        res_out, res_base, grad_outputs=torch.ones_like(res_out)
    )

    assert_result_equal(res_out, ref_out)
    assert res_out is res_receiver
    assert_result_equal(res_receiver, ref_receiver)
    assert_result_close(res_grad, ref_grad)
    assert_result_equal(other, ref_other)


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
def test_ge__special_values(dtype, scenario):
    inp = make_special_input(dtype, scenario)
    # The flipped operand pairs each special value with a different one, so nan
    # and inf take part on both sides of the comparison.
    other = make_special_input(dtype, scenario).flip(0)

    ref_inp = to_reference(inp)
    ref_other = to_reference(other)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_other)
    res_out = flag_gems.ge_tensor_(res_inp, other)

    assert_result_equal(res_out, ref_out)
    assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp
    assert_result_equal(other, ref_other)


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize("dtype,shape,value_range", ALIAS_CASES)
def test_ge__aliased_operand(dtype, shape, value_range):
    inp = make_input(dtype, shape, value_range)

    ref_inp = to_reference(inp)
    res_inp = inp.clone()

    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_inp)
    res_out = flag_gems.ge_tensor_(res_inp, res_inp)

    assert_result_equal(res_out, ref_out)
    assert_result_equal(res_inp, ref_inp)
    assert res_out is res_inp


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize("dtype,kind", VIEW_CASES)
def test_ge__view_receiver(dtype, kind):
    storage_shape, other_shape = _VIEW_KINDS[kind]
    storage = make_input(dtype, storage_shape, ["-1", "1"])
    other = make_input(dtype, other_shape, ["-1", "1"])

    ref_storage = to_reference(storage)
    res_storage = storage.clone()
    ref_inp = _view_of(ref_storage, kind)
    res_inp = _view_of(res_storage, kind)

    ref_other = to_reference(other)
    ref_out = torch.ops.aten.ge_.Tensor(ref_inp, ref_other)
    res_out = flag_gems.ge_tensor_(res_inp, other)

    assert_result_equal(res_out, ref_out)
    assert_result_equal(res_inp, ref_inp)
    # The whole backing storage shows the writes landed at the view's offsets.
    assert_result_equal(res_storage, ref_storage)
    assert res_out is res_inp
    assert_result_equal(other, ref_other)


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize("shape,other_shape", NEGATIVE_CASES)
def test_ge__invalid_operand(shape, other_shape):
    inp = torch.ones(shape, dtype=torch.float32, device=flag_gems.device)
    other = torch.ones(other_shape, dtype=torch.float32, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ge_tensor_(inp, other)


@pytest.mark.ge_
@pytest.mark.ge_tensor_
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
def test_ge__unsupported_dtype(dtype):
    inp = torch.ones((2, 19, 7), dtype=dtype, device=flag_gems.device)
    other = torch.ones((2, 19, 7), dtype=dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.ge_tensor_(inp, other)
