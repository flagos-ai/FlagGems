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

"""Correctness tests for ``aten::_autocast_to_reduced_precision``.

Schema: ``(self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype)``. On the
active device backend a float32 input is cast to ``cuda_dtype`` when
``cuda_enabled`` is set and ``cuda_dtype`` is not float32; every other source
dtype, a disabled ``cuda_enabled``, or ``cuda_dtype == float32`` makes the op
return ``self`` unchanged. ``cpu_enabled``/``cpu_dtype`` address CPU-resident
inputs on a CUDA-style backend, so for a device input they are schema arguments
only and are not allocations.

The conversion is a cast, not a promotion: it rounds to the target mantissa,
underflows to zero and overflows per target format, and its backward casts the
upstream gradient back to the source dtype. Comparisons are exact because
candidate and reference apply the same cast to the same values, not because the
values survive unchanged.

Recorded protocol gap (not worked around): the host CPU kernel selects its branch
with ``is_cpu() && cpu_enabled`` and then follows ``cpu_dtype``, so under
``--ref cpu`` the reference returns the float32 input where this device candidate
converts it. The converted workloads are kept because the reference normally
dispatches like the candidate; reconciling the two reference modes needs an
agreed selection policy rather than rewritten flags or dropped workloads.
"""

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import test_utils as tu

# ``cpu_dtype`` is inactive for a device input on this backend (see the module
# docstring); a value that differs from every ``cuda_dtype`` below makes a
# candidate that reads the wrong branch return the wrong dtype instead of
# passing silently.
_CPU_DTYPE = torch.float64

_UNIT_RANGE = ["-1", "1"]

# Static capability gates: import-time flags only, never probed at run time.
_SOURCE_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    _SOURCE_DTYPES.append(torch.bfloat16)
_SOURCE_DTYPES.append(torch.float32)
if utils.fp64_is_supported:
    _SOURCE_DTYPES.append(torch.float64)
if utils.fp8_is_supported:
    _SOURCE_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])
_SOURCE_DTYPES.extend([torch.int8, torch.uint8, torch.int32])
if utils.int64_is_supported:
    _SOURCE_DTYPES.append(torch.int64)
_SOURCE_DTYPES.append(torch.bool)

_REDUCED_TARGETS = [torch.float16]
if utils.bf16_is_supported:
    _REDUCED_TARGETS.append(torch.bfloat16)
if utils.fp8_is_supported:
    _REDUCED_TARGETS.extend([torch.float8_e4m3fn, torch.float8_e5m2])

# Native casts a float32 source only, and not even then when the target is
# float32; every other row exercises the pass-through branch of the same
# dispatch condition and is paired with one representative reduced target.
_CONVERSION_ROWS = [
    (torch.float32, target) for target in _REDUCED_TARGETS + [torch.float32]
] + [
    (src_dtype, torch.float16)
    for src_dtype in _SOURCE_DTYPES
    if src_dtype != torch.float32
]


def _assert_alias_matches_native(res_out, res_inp, ref_out, ref_inp):
    """``Tensor(a) -> Tensor(a)``: compare the candidate's object relation with
    the measured native relation. Identity is compared instead of a storage
    pointer, because a distinct view and an empty clone can both reuse the
    pointer of the tensor they were derived from."""
    assert (res_out is res_inp) == (ref_out is ref_inp)


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("shape", tu.selected_shapes())
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("src_dtype,cuda_dtype", _CONVERSION_ROWS)
def test__autocast_to_reduced_precision(shape, value_range, src_dtype, cuda_dtype):
    inp = tu.make_input(src_dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, cuda_dtype, _CPU_DTYPE
    )

    tu.assert_result_equal(res_out, ref_out)
    # Native hands back the input object for the pass-through rows and a fresh
    # tensor for the conversion rows; the expected relation is read from the
    # reference rather than assumed from the flag values.
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


# Default-only: the four flag pairs are a parameter sweep, so they stay out of
# the quick smoke grid while the negative tests below stay in both levels.
_ENABLE_FLAGS = tu.selected_cases(
    [(True, False), (True, True), (False, False), (False, True)], quick=[]
)


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("shape", [(20, 320, 15)])
@pytest.mark.parametrize("cuda_enabled,cpu_enabled", _ENABLE_FLAGS)
def test__autocast_to_reduced_precision_enable_flags(shape, cuda_enabled, cpu_enabled):
    inp = tu.make_input(torch.float32, shape, _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)
    cuda_dtype = torch.float16

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, cuda_enabled, cpu_enabled, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, cuda_enabled, cpu_enabled, cuda_dtype, _CPU_DTYPE
    )

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


# Views built from a dense base. ``transpose`` is a full transpose and
# ``strided_slice`` steps every other column, so both are non-contiguous;
# ``offset_slice`` takes ``base[1:, 1:]``, which inherits the base row stride 64
# while each row holds only 63 elements, so it is non-contiguous as well and
# starts at storage offset 65. Pass-through returns the input object without
# materialising it, so a silently copying candidate is caught by the alias
# relation below; the conversion rows allocate a cast result and there the
# relation is only compared against the native outcome.
_STRIDED_BASE_SHAPES = {
    "transpose": (64, 32),
    "strided_slice": (64, 64),
    "offset_slice": (64, 64),
}

# Default-only: non-contiguous layouts supplement the main shape grid.
_STRIDED_LAYOUTS = tu.selected_cases(
    ["transpose", "strided_slice", "offset_slice"], quick=[]
)

_CONVERTED_STRIDED_ROWS = [(torch.float32, target) for target in _REDUCED_TARGETS]

_IDENTITY_STRIDED_ROWS = [(torch.float16, torch.float16), (torch.int32, torch.float16)]
if utils.bf16_is_supported:
    _IDENTITY_STRIDED_ROWS.append((torch.bfloat16, torch.float16))
if utils.fp8_is_supported:
    _IDENTITY_STRIDED_ROWS.append((torch.float8_e4m3fn, torch.float16))
if utils.int64_is_supported:
    _IDENTITY_STRIDED_ROWS.append((torch.int64, torch.float16))


def _make_strided(layout, base):
    """Return the named non-contiguous view of ``base``."""
    if layout == "transpose":
        return base.t()
    if layout == "strided_slice":
        return base[:, ::2]
    return base[1:, 1:]


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("layout", _STRIDED_LAYOUTS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("src_dtype,cuda_dtype", _CONVERTED_STRIDED_ROWS)
def test__autocast_to_reduced_precision_strided_conversion(
    layout, value_range, src_dtype, cuda_dtype
):
    base = tu.make_input(src_dtype, _STRIDED_BASE_SHAPES[layout], value_range)
    inp = _make_strided(layout, base)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, cuda_dtype, _CPU_DTYPE
    )

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("layout", _STRIDED_LAYOUTS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("src_dtype,cuda_dtype", _IDENTITY_STRIDED_ROWS)
def test__autocast_to_reduced_precision_strided_pass_through(
    layout, value_range, src_dtype, cuda_dtype
):
    base = tu.make_input(src_dtype, _STRIDED_BASE_SHAPES[layout], value_range)
    inp = _make_strided(layout, base)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, cuda_dtype, _CPU_DTYPE
    )

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


_BACKWARD_SHAPES = tu.selected_cases([(2, 16), (4, 8, 16), (20, 320, 15)], quick=[])

_BACKWARD_TARGETS = [torch.float16]
if utils.bf16_is_supported:
    _BACKWARD_TARGETS.append(torch.bfloat16)

# Each row is ``(cuda_enabled, src_dtype, cuda_dtype, upstream_dtype)``. The
# upstream gradient keeps the forward output's dtype so both sides differentiate
# an identical call; the disabled-flag and pass-through rows cover the aliasing
# path whose gradient is the upstream tensor itself.
_BACKWARD_ROWS = [
    (True, torch.float32, target, target) for target in _BACKWARD_TARGETS
] + [
    (False, torch.float32, torch.float16, torch.float32),
    (True, torch.float16, torch.float16, torch.float16),
]


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("shape", _BACKWARD_SHAPES)
@pytest.mark.parametrize(
    "cuda_enabled,src_dtype,cuda_dtype,upstream_dtype", _BACKWARD_ROWS
)
def test__autocast_to_reduced_precision_backward(
    shape, cuda_enabled, src_dtype, cuda_dtype, upstream_dtype
):
    inp = tu.make_input(src_dtype, shape, _UNIT_RANGE).requires_grad_(True)
    ref_inp = tu.to_reference(inp)
    upstream = tu.make_input(upstream_dtype, shape, _UNIT_RANGE)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, cuda_enabled, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, cuda_enabled, False, cuda_dtype, _CPU_DTYPE
    )
    tu.assert_result_equal(res_out, ref_out)

    ref_grad = torch.autograd.grad(
        ref_out, ref_inp, grad_outputs=tu.to_reference(upstream)
    )[0]
    res_grad = torch.autograd.grad(res_out, inp, grad_outputs=upstream)[0]

    # Backward reuses the upstream gradient on the aliasing rows and casts it back
    # to the source dtype on the conversion rows; the same rule applies on both
    # sides, so the comparison is exact. The random upstream values are
    # deliberately non-uniform, so a dropped or rescaled gradient cannot pass.
    tu.assert_result_equal(res_grad, ref_grad)


_SPECIAL_DTYPES = [torch.float16]
if utils.bf16_is_supported:
    _SPECIAL_DTYPES.append(torch.bfloat16)
_SPECIAL_DTYPES.append(torch.float32)
if utils.fp64_is_supported:
    _SPECIAL_DTYPES.append(torch.float64)
if utils.fp8_is_supported:
    _SPECIAL_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])

_SPECIAL_TARGETS = [torch.float16]
if utils.bf16_is_supported:
    _SPECIAL_TARGETS.append(torch.bfloat16)

# ``tu.special_value_cases`` supplies nan for every floating dtype and inf/mixed
# wherever the dtype can represent it (e4m3fn is nan-only, e5m2 gets all three).
_SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(_SPECIAL_DTYPES), quick=[])


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("cuda_dtype", _SPECIAL_TARGETS)
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test__autocast_to_reduced_precision_special_values(dtype, scenario, cuda_dtype):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, cuda_dtype, _CPU_DTYPE
    )

    # Matching NaNs and matching infinities are both part of the contract.
    tu.assert_result_equal(res_out, ref_out)


_FP8_TARGETS = [
    dtype
    for dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if utils.fp8_is_supported
]

_FP8_SPECIAL_CASES = tu.selected_cases(["nan", "inf", "mixed"], quick=[])


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("cuda_dtype", _FP8_TARGETS)
@pytest.mark.parametrize("scenario", _FP8_SPECIAL_CASES)
def test__autocast_to_reduced_precision_fp8_target_special_values(cuda_dtype, scenario):
    inp = tu.make_special_input(torch.float32, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, cuda_dtype, _CPU_DTYPE
    )

    # Each FP8 format owns its rule for out-of-range values: e4m3fn has no
    # infinity and turns overflow into NaN, while e5m2 keeps +-inf. Both are
    # compared exactly, including matching NaNs.
    tu.assert_result_equal(res_out, ref_out)


# Finite values that separate the reduced targets: underflow to zero, rounding,
# and each format's overflow boundary.
_BOUNDARY_VALUES = [
    0.0,
    -0.0,
    1e-45,
    6e-8,
    0.1,
    1.0,
    -1.0,
    448.0,
    449.0,
    -449.0,
    57344.0,
    65504.0,
    65520.0,
    1e5,
    1e30,
]

_BOUNDARY_TARGETS = tu.selected_cases(_REDUCED_TARGETS, quick=[])


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("cuda_dtype", _BOUNDARY_TARGETS)
def test__autocast_to_reduced_precision_finite_boundaries(cuda_dtype):
    inp = torch.tensor(_BOUNDARY_VALUES, dtype=torch.float32, device=flag_gems.device)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, cuda_dtype, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, cuda_dtype, _CPU_DTYPE
    )

    # Underflow, rounding and overflow differ per target: fp16 and e5m2 overflow
    # to infinity, bfloat16 rounds up to its next representable value, and
    # e4m3fn (no infinity) turns its out-of-range values into NaN. 449 is rounded
    # to the nearest representable 448 rather than clamped to the maximum.
    tu.assert_result_equal(res_out, ref_out)


_KEYWORD_CASES = tu.selected_cases(
    [(True, False), (True, True), (False, False), (False, True)], quick=[]
)


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("shape", [(1024, 1024), (20, 320, 15)])
@pytest.mark.parametrize("cuda_enabled,cpu_enabled", _KEYWORD_CASES)
def test__autocast_to_reduced_precision_keyword_arguments(
    shape, cuda_enabled, cpu_enabled
):
    inp = tu.make_input(torch.float32, shape, _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp,
        cuda_enabled=cuda_enabled,
        cpu_enabled=cpu_enabled,
        cuda_dtype=torch.float16,
        cpu_dtype=_CPU_DTYPE,
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp,
        cuda_enabled=cuda_enabled,
        cpu_enabled=cpu_enabled,
        cuda_dtype=torch.float16,
        cpu_dtype=_CPU_DTYPE,
    )

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


# ``cpu_dtype`` is a ScalarType schema argument for the CPU branch and is inactive
# for a device input on this backend (see the module docstring). It is kept
# different from ``cuda_dtype`` so a candidate reading the wrong branch returns
# the wrong dtype. Both rows stay in the grid: float64 here is an argument, not an
# allocation, so the row does not depend on the device's FP64 capability.
_CPU_DTYPE_CASES = tu.selected_cases([torch.float16, torch.float64], quick=[])


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("cpu_dtype", _CPU_DTYPE_CASES)
def test__autocast_to_reduced_precision_cpu_dtype_ignored(cpu_dtype):
    inp = tu.make_input(torch.float32, (20, 320, 15), _UNIT_RANGE)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, True, torch.float16, cpu_dtype
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, True, torch.float16, cpu_dtype
    )

    # Both flags are set, but ``cpu_dtype`` must not reach a device input: the
    # result follows ``cuda_dtype``.
    tu.assert_result_equal(res_out, ref_out)


_GEOMETRY_CASES = tu.selected_cases(
    [
        ("zero_extent", (8,)),
        ("empty_batch", (4, 6, 3)),
        ("offset_slice", (12, 10)),
        ("expanded", (4, 1, 5)),
    ],
    quick=[],
)


def _make_geometry_view(kind, base):
    """Return a zero-extent, offset-slice or stride-0 (expanded) view of ``base``.

    ``offset_slice`` returns ``base[2:, 3:]``: both axes are sliced, the rows
    inherit the base row stride (10 for the (12, 10) base, against a 7-element
    row width), and the view starts at a non-zero storage offset, so it is an
    offset slice rather than a contiguous slice.
    """
    if kind == "zero_extent":
        return base[:0]
    if kind == "empty_batch":
        return base[:, :0, :]
    if kind == "offset_slice":
        return base[2:, 3:]
    return base.expand(base.shape[0], 3, base.shape[-1])


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("kind,base_shape", _GEOMETRY_CASES)
def test__autocast_to_reduced_precision_pass_through_geometry(kind, base_shape):
    base = tu.make_input(torch.float16, base_shape, _UNIT_RANGE)
    inp = _make_geometry_view(kind, base)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, torch.float16, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, torch.float16, _CPU_DTYPE
    )

    # Empty, offset-slice and stride-0 views must come back as the very same
    # object, not merely as a tensor with the same values.
    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("kind,base_shape", _GEOMETRY_CASES)
def test__autocast_to_reduced_precision_converted_geometry(kind, base_shape):
    base = tu.make_input(torch.float32, base_shape, _UNIT_RANGE)
    inp = _make_geometry_view(kind, base)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._autocast_to_reduced_precision(
        ref_inp, True, False, torch.float16, _CPU_DTYPE
    )
    res_out = flag_gems._autocast_to_reduced_precision(
        inp, True, False, torch.float16, _CPU_DTYPE
    )

    tu.assert_result_equal(res_out, ref_out)
    _assert_alias_matches_native(res_out, inp, ref_out, ref_inp)


@pytest.mark.autocast_to_reduced_precision
def test__autocast_to_reduced_precision_missing_argument_raises():
    inp = tu.make_input(torch.float32, (4,), _UNIT_RANGE)
    # ``cpu_dtype`` is a required argument of the schema.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._autocast_to_reduced_precision(inp, True, False, torch.float16)


@pytest.mark.autocast_to_reduced_precision
@pytest.mark.parametrize("invalid_dtype", [None, 1.5])
def test__autocast_to_reduced_precision_invalid_dtype_argument_raises(invalid_dtype):
    inp = tu.make_input(torch.float32, (4,), _UNIT_RANGE)
    # ATen requires a ScalarType for both dtype arguments.
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._autocast_to_reduced_precision(
            inp, True, False, invalid_dtype, invalid_dtype
        )


@pytest.mark.autocast_to_reduced_precision
def test__autocast_to_reduced_precision_invalid_flag_argument_raises():
    inp = tu.make_input(torch.float32, (4,), _UNIT_RANGE)
    # The flags are bool; a string is rejected (integer 1/0 is accepted).
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._autocast_to_reduced_precision(
            inp, "yes", False, torch.float16, _CPU_DTYPE
        )


@pytest.mark.autocast_to_reduced_precision
def test__autocast_to_reduced_precision_non_tensor_input_raises():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems._autocast_to_reduced_precision(
            [1.0, 2.0], True, False, torch.float16, _CPU_DTYPE
        )
