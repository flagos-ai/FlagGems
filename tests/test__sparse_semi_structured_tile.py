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

from . import test_utils as tu

MEASURED_VENDOR = "nvidia"


def _static_dtype_lists(support_bf16, support_fp64, support_int64, support_fp8):
    """(kernel dtypes, kernel-rejected dtypes) from static capabilities only.

    Only fp16 and bf16 reach the native kernel; every other dtype is rejected by
    the kernel dispatch with a message naming the scalar type (measured on the
    assigned nvidia device: Char, Byte, Int, Long, Float, Double, Bool,
    Float8_e4m3fn, Float8_e5m2). A dtype the device cannot even construct is
    left out of the negative rows instead of becoming a construction failure.
    """
    constructible = {
        torch.bfloat16: support_bf16,
        torch.float64: support_fp64,
        torch.int64: support_int64,
        torch.float8_e4m3fn: support_fp8,
        torch.float8_e5m2: support_fp8,
    }
    supported = [torch.float16] + ([torch.bfloat16] if support_bf16 else [])
    rejected = [
        dtype
        for dtype in list(tu.REQUIRED_DTYPES) + [torch.float64]
        if dtype not in supported and constructible.get(dtype, True)
    ]
    return supported, rejected


TILE_DTYPES, _KERNEL_REJECTED_DTYPES = _static_dtype_lists(
    flag_gems.runtime.device.support_bf16,
    flag_gems.runtime.device.support_fp64,
    flag_gems.runtime.device.support_int64,
    flag_gems.runtime.device.support_fp8,
)

# The per-dtype rejection is a property of the CUDA kernel it was measured on,
# so those negative workloads are scoped to that vendor (and are empty, i.e.
# not collected, elsewhere) rather than re-interpreted on an unknown kernel.
REJECTED_DTYPES = (
    _KERNEL_REJECTED_DTYPES if flag_gems.vendor_name == MEASURED_VENDOR else []
)

# ---------------------------------------------------------------------------
# Case data
# ---------------------------------------------------------------------------
# The native kernel is reproducible across repeated identical calls only where
# rows % 64 == 0 and cols % 64 == 0 (measured over 5 repeats against an
# independent ``tu.to_reference`` call: (64, 96), (96, 64), (32, 64) and
# (64, 32) disagree with it and with each other in several of the five
# components, while (64, 64) agrees exactly for both dtypes), so only 64-aligned
# tiles can serve as an oracle. The spec ranks 0..5 are mapped to 64-aligned 2-D
# tiles of the same element scale, rounded up to the next aligned factorisation
# where the exact product has none; no element-count cap is applied.
SSST_SHAPES = [
    (64, 64),  # 4_096 - spec () and (1,), the 2-D unit tile
    (64, 256),  # 16_384 - spec 1-D (256,)
    (64, 2_048),  # 131_072
    (1_024, 1_024),  # 1_048_576 - spec 2-D (1024, 1024) unchanged
    (256, 384),  # 98_304 - spec 3-D (20, 320, 15) = 96_000 rounded up
    (64, 4_800),  # 307_200 - additional mid-size tile
    (512, 4_096),  # 2_097_152
    (1_024, 4_096),  # 4_194_304
    (1_408, 4_224),  # 5_947_392 - spec 5-D (16, 7, 57, 32, 29) = 5_924_352 rounded up
    (2_048, 3_840),  # 7_864_320 - spec 4-D (16, 128, 64, 60) exactly
    (64, 65_536),  # 4_194_304 - additional extreme-aspect tile
]
# Quick keeps one compact aligned shape, the default value range and every
# supported dtype; the smoke subset carries no positive supplement case.
QUICK_SSST_SHAPES = [(64, 64)]

# Positive special values and deterministic tie / zero payload families. All are
# reproducible at 64-aligned geometry for both dtypes (measured over 3 repeats
# against an independent reference, NaN-aware comparison, input unchanged), and
# the payloads reach the result with nan and inf preserved. Kept out of quick:
# the smoke subset carries no positive special-value cases.
PAYLOAD_SHAPES = [(64, 64), (128, 128), (1_024, 1_024)]
SPECIAL_SCENARIOS = sorted(
    {scenario for _, scenario in tu.special_value_cases(TILE_DTYPES)}
)
PACKED_FAMILIES = ["all_equal", "tie_pairs", "zeros", "signed_zero"]
PAYLOAD_CASES = tu.selected_cases(
    [("special", scenario) for scenario in SPECIAL_SCENARIOS]
    + [("packed", family) for family in PACKED_FAMILIES],
    quick=[],
)

ALGORITHM_SHAPES = tu.selected_cases([(64, 64), (128, 128), (1_024, 1_024)], quick=[])
EXPLICIT_ALGORITHMS = tu.selected_cases(
    ["", "largest_values_greedy", "largest_abs_values_greedy"], quick=[]
)
# Every valid algorithm x use_cutlass combination; the empty tuple is the
# omitted algorithm argument, i.e. the schema default, a distinct call form from
# the explicit empty string. Measured: the omitted argument, '' and
# 'largest_values_greedy' agree on all five components, while
# 'largest_abs_values_greedy' differs from them in all five.
CUTLASS_FORMS = tu.selected_cases(
    [
        ((), True),
        ((), False),
        (("",), True),
        (("",), False),
        (("largest_values_greedy",), True),
        (("largest_values_greedy",), False),
        (("largest_abs_values_greedy",), True),
        (("largest_abs_values_greedy",), False),
    ],
    quick=[],
)
CUTLASS_SHAPES = tu.selected_cases(
    [(128, 128), (256, 512), (256, 384), (1_024, 1_024)], quick=[]
)
# 64-aligned but not 128-aligned: only the default use_cutlass path accepts it
# ('Only supports rows/cols multiples of 128' for the other path).
SMALL_ALIGN_SHAPES = tu.selected_cases([(64, 64), (192, 2_048), (320, 64)], quick=[])

# Layouts the native operator accepts: a contiguous view carrying a nonzero
# storage offset and a row-strided view; both were measured equal to their
# contiguous control over repeated calls.
VIEW_LAYOUTS = tu.selected_cases(["offset", "row_stride"], quick=[])

# Negative workloads, collected in quick as well.
BAD_RANK_SHAPES = [
    (),
    (64,),
    (2, 32, 64),
    (1, 64, 64),
    (2, 2, 64, 64),
    (1, 1, 1, 64, 64),
]
BAD_COLS_SHAPES = [(64, 16), (64, 17), (64, 48), (64, 80), (64, 112)]
BAD_LAYOUTS = ["column_stride", "transpose"]
BAD_ALGORITHMS = ["x", "largest_abs", "LARGEST_VALUES_GREEDY"]
BAD_ALGORITHM_TYPES = [None, 3, True, b"x", ["largest_values_greedy"]]
BAD_CUTLASS_SHAPES = [(64, 64), (64, 128), (128, 64)]

# Backward is exempt: the native operator has no derivative (measured
# RuntimeError 'derivative for aten::_sparse_semi_structured_tile is not
# implemented'), so no gradient oracle exists for a candidate to be compared
# against. The operator is unary, so broadcast is not applicable either.


def _special_input(dtype, scenario, shape):
    """Broadcast the shared special-value payload over a 2-D tile."""
    payload = tu.make_special_input(dtype, scenario)
    rows, cols = shape
    repeats = (rows * cols + payload.numel() - 1) // payload.numel()
    return payload.repeat(repeats)[: rows * cols].reshape(rows, cols)


def _packed_input(dtype, family, shape):
    """Deterministic tie / zero payload families.

    A random fixture cannot establish that a tile of equal values, or a tile that
    is all zero or alternating signed zero, is handled reproducibly; these
    families do that. In ``tie_pairs`` only every 4-value group ties. Signed zero
    is an input family here; the shared helper compares numerically, so it does
    not pin the sign of a zero result.
    """
    rows, cols = shape
    device = flag_gems.device
    if family == "all_equal":
        return torch.full(shape, 1.0, device=device, dtype=dtype)
    if family == "tie_pairs":
        column = torch.tensor(
            [(i // 2) % 3 + 1 for i in range(cols)], device=device, dtype=torch.float32
        )
        row = 1.0 + torch.arange(rows, device=device, dtype=torch.float32) / (2 * rows)
        return (column.unsqueeze(0) * row.unsqueeze(1)).to(dtype)
    if family == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    if family == "signed_zero":
        values = torch.zeros(shape, device=device, dtype=torch.float32)
        values[:, 1::2] = -0.0
        return values.to(dtype)
    raise ValueError("unknown payload family %r" % family)


def _payload_input(dtype, kind, name, shape):
    if kind == "special":
        return _special_input(dtype, name, shape)
    return _packed_input(dtype, name, shape)


def _assert_tile_outputs(res, ref):
    """Compare all five result components with the shared exact helper.

    The shared helper compares numerically exactly (rtol=atol=0) and allows
    matching NaNs; it is not a bitwise comparison, so a zero-sign difference is
    accepted. A candidate result shorter than the reference must not compare as a
    prefix of it.
    """
    assert len(res) == len(ref)
    for res_part, ref_part in zip(res, ref):
        tu.assert_result_equal(res_part, ref_part)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "shape", tu.selected_cases(SSST_SHAPES, quick=QUICK_SSST_SHAPES)
)
def test__sparse_semi_structured_tile_value_range(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    inp_before = inp.clone()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(ref_inp)
    res_out = flag_gems._sparse_semi_structured_tile(inp)

    _assert_tile_outputs(res_out, ref_out)
    # The operator only reads its input.
    tu.assert_result_equal(inp, inp_before)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", PAYLOAD_SHAPES)
@pytest.mark.parametrize("kind,name", PAYLOAD_CASES)
def test__sparse_semi_structured_tile_payload_families(kind, name, shape, dtype):
    inp = _payload_input(dtype, kind, name, shape)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(ref_inp)
    res_out = flag_gems._sparse_semi_structured_tile(inp)

    _assert_tile_outputs(res_out, ref_out)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", ALGORITHM_SHAPES)
def test__sparse_semi_structured_tile_default_algorithm_omitted(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(ref_inp)
    res_out = flag_gems._sparse_semi_structured_tile(inp)

    _assert_tile_outputs(res_out, ref_out)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("algorithm", EXPLICIT_ALGORITHMS)
@pytest.mark.parametrize("shape", ALGORITHM_SHAPES)
def test__sparse_semi_structured_tile_algorithm(algorithm, shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(ref_inp, algorithm)
    res_out = flag_gems._sparse_semi_structured_tile(inp, algorithm)

    _assert_tile_outputs(res_out, ref_out)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", CUTLASS_SHAPES)
@pytest.mark.parametrize("form", CUTLASS_FORMS)
def test__sparse_semi_structured_tile_algorithm_use_cutlass(form, shape, dtype):
    algorithm_args, use_cutlass = form
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(
        ref_inp, *algorithm_args, use_cutlass=use_cutlass
    )
    res_out = flag_gems._sparse_semi_structured_tile(
        inp, *algorithm_args, use_cutlass=use_cutlass
    )

    _assert_tile_outputs(res_out, ref_out)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", SMALL_ALIGN_SHAPES)
def test__sparse_semi_structured_tile_aligned64_geometry(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(ref_inp)
    res_out = flag_gems._sparse_semi_structured_tile(inp)

    _assert_tile_outputs(res_out, ref_out)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("layout", VIEW_LAYOUTS)
def test__sparse_semi_structured_tile_accepts_views(layout, dtype):
    # Native accepts a contiguous view with a nonzero storage offset and a
    # row-strided view; the reference receives the same view metadata.
    base = tu.make_input(dtype, (256, 64), ["-1", "1"])
    inp = base[32:96] if layout == "offset" else base[::2, :64]
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten._sparse_semi_structured_tile(ref_inp)
    res_out = flag_gems._sparse_semi_structured_tile(inp)

    _assert_tile_outputs(res_out, ref_out)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", BAD_RANK_SHAPES)
def test__sparse_semi_structured_tile_rejects_rank(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", BAD_COLS_SHAPES)
def test__sparse_semi_structured_tile_rejects_unaligned_cols(shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("layout", BAD_LAYOUTS)
def test__sparse_semi_structured_tile_rejects_layout(layout, dtype):
    # Independently probed rejects: a column-strided view and a transposed view.
    base = tu.make_input(dtype, (128, 128), ["-1", "1"])
    inp = base[:, ::2] if layout == "column_stride" else base.t()

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("algorithm", BAD_ALGORITHMS)
def test__sparse_semi_structured_tile_rejects_unknown_algorithm(algorithm, dtype):
    inp = tu.make_input(dtype, (128, 128), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp, algorithm)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("algorithm", BAD_ALGORITHM_TYPES)
def test__sparse_semi_structured_tile_rejects_algorithm_type(algorithm, dtype):
    # None/int/bool/list fail the 'str' schema conversion; b'x' is decoded and
    # then rejected as an unknown algorithm.
    inp = tu.make_input(dtype, (128, 128), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp, algorithm)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", TILE_DTYPES)
@pytest.mark.parametrize("shape", BAD_CUTLASS_SHAPES)
def test__sparse_semi_structured_tile_rejects_cutlass_geometry(shape, dtype):
    # use_cutlass=False requires rows and cols multiples of 128.
    inp = tu.make_input(dtype, shape, ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp, "", False)


@pytest.mark.sparse_semi_structured_tile
@pytest.mark.parametrize("dtype", REJECTED_DTYPES)
def test__sparse_semi_structured_tile_rejects_dtype(dtype):
    inp = tu.make_input(dtype, (128, 128), ["-1", "1"])

    with pytest.raises(RuntimeError):
        flag_gems._sparse_semi_structured_tile(inp)
