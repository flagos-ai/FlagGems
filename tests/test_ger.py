# Copyright 2026, The FlagOS Contributors.
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
from . import conftest as cfg

# aten::ger is the dispatch target the wrapper replaces, so the reference goes
# through it rather than through the torch.ger alias.
_ATEN_OP = torch.ops.aten.ger

# Static device capability flags decide which optional dtypes are declared.
GER_DTYPES = [
    torch.bool,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.float16,
    torch.float32,
]
if utils.int64_is_supported:
    GER_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    GER_DTYPES.append(torch.bfloat16)
if utils.fp64_is_supported:
    GER_DTYPES.append(torch.float64)
GER_DTYPES.append(torch.complex64)

GER_FLOAT_DTYPES = [
    dtype
    for dtype in GER_DTYPES
    if dtype in (torch.float16, torch.float32, torch.bfloat16, torch.float64)
]
GER_LAYOUT_DTYPES = [
    dtype for dtype in GER_DTYPES if dtype in (torch.float32, torch.bfloat16)
]


def _cases(cases, quick):
    """Pick the quick subset under --quick, the full list otherwise."""
    return list(quick) if cfg.QUICK_MODE else list(cases)


# ger is the rank-1 outer product, so both operands must be 1-D and a case is a
# (self_len, vec2_len) operand pair rather than a tensor shape. There is no
# broadcast dimension. The length-1 and lopsided pairs cover the degenerate
# grids, and the zero lengths are native-valid and reach the empty-output early
# return in the wrapper.
GER_SHAPE_PAIRS = [
    (1, 1),
    (1, 256),
    (256, 1),
    (256, 256),
    (512, 1024),
    (1024, 512),
    (1024, 1024),
    (0, 4),
    (4, 0),
    (0, 0),
]
# Quick mode keeps one small non-divisible pair as a smoke case.
QUICK_SHAPE_PAIRS = [(19, 7)]
VALUE_SHAPE_PAIRS = _cases(GER_SHAPE_PAIRS, QUICK_SHAPE_PAIRS)

# The product saturates differently at each end of a dtype's range, so the
# ranges are swept per dtype: [0, max] and [min, 0] push the operands to the
# dtype bounds where the product overflows for floats and wraps for integers,
# while the signed sub-ranges separate sign handling from magnitude.
GER_VALUE_RANGES = [
    ("-1", "1"),
    ("0", "1"),
    ("-1", "0"),
    ("0", "max"),
    ("min", "0"),
]
# The symmetric range alone already signs both operands, which is enough for a
# quick smoke run.
QUICK_VALUE_RANGES = [("-1", "1")]
VALUE_RANGES = _cases(GER_VALUE_RANGES, QUICK_VALUE_RANGES)


def _dtype_bounds(dtype):
    """Return the (min, max) representable values of ``dtype``."""
    if dtype == torch.bool:
        # bool carries no finfo/iinfo; its value domain is literally {0, 1}.
        return 0, 1
    if dtype.is_complex:
        real = {
            torch.complex32: torch.float16,
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[dtype]
        finfo = torch.finfo(real)
        return finfo.min, finfo.max
    if dtype.is_floating_point:
        finfo = torch.finfo(dtype)
        return finfo.min, finfo.max
    iinfo = torch.iinfo(dtype)
    return iinfo.min, iinfo.max


def _resolve_bound(symbol, dtype):
    """Resolve a range-bound symbol to a concrete value for ``dtype``."""
    low, high = _dtype_bounds(dtype)
    # The symbol table is the value-range vocabulary used by the case lists
    # above; "max"/"min" mean the dtype's own bounds rather than fixed numbers.
    return {"-1": -1.0, "0": 0.0, "1": 1.0, "max": high, "min": low}[symbol]


def _make_input(dtype, shape, value_range=("-1", "1")):
    """Build a tensor of ``shape`` / ``dtype`` with values in ``value_range``.

    bool ignores the range, integer ranges are snapped to ints and a degenerate
    range fills a constant. Bounds are clamped to the dtype's representable
    range first, so one range list works for every dtype: uint8 cannot hold
    ``-1``, so ``[-1, 0]`` degenerates into a constant zero fill instead of
    needing an unsigned special case.
    """
    if dtype == torch.bool:
        # randint covers both bool values uniformly; make_tensor does not
        # accept low/high for bool.
        return torch.randint(0, 2, shape, device=flag_gems.device).bool()

    low = _resolve_bound(value_range[0], dtype)
    high = _resolve_bound(value_range[1], dtype)

    if not (dtype.is_floating_point or dtype.is_complex):
        dtype_min, dtype_max = _dtype_bounds(dtype)
        low = max(int(low), int(dtype_min))
        high = min(int(high), int(dtype_max))
        low = min(low, high)

    if low == high:
        return torch.full(shape, low, dtype=dtype, device=flag_gems.device)

    return torch.testing.make_tensor(
        shape, dtype=dtype, device=flag_gems.device, low=low, high=high
    )


def _assert_close(result, reference, atol=1e-4):
    """Compare in the result dtype, exactly for bool and integral results."""
    dtype = result.dtype
    assert dtype == reference.dtype
    if dtype.is_floating_point or dtype.is_complex:
        utils.gems_assert_close(result, reference, dtype, equal_nan=True, atol=atol)
    else:
        utils.gems_assert_equal(result, reference)


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair", VALUE_SHAPE_PAIRS)
@pytest.mark.parametrize("value_range", VALUE_RANGES)
@pytest.mark.parametrize("dtype", GER_DTYPES)
def test_ger_values(shape_pair, value_range, dtype):
    self_len, vec2_len = shape_pair
    inp = _make_input(dtype, (self_len,), value_range)
    vec2 = _make_input(dtype, (vec2_len,), value_range)
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_out = _ATEN_OP(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    assert res_out.device == inp.device
    assert res_out.shape == (self_len, vec2_len)
    _assert_close(res_out, ref_out)


# The host-side tile selection in the wrapper switches configuration once the
# output reaches 1 << 22 elements, and the value sweep above tops out below
# that. These two pairs straddle the cutoff from both sides, with (97, 33)
# dividing neither tile size so every program takes the masked path.
GER_TILE_CONFIG_PAIRS = _cases([(97, 33), (4096, 1024)], [(97, 33)])


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair", GER_TILE_CONFIG_PAIRS)
@pytest.mark.parametrize("dtype", GER_LAYOUT_DTYPES)
def test_ger_tile_config(shape_pair, dtype):
    self_len, vec2_len = shape_pair
    inp = _make_input(dtype, (self_len,))
    vec2 = _make_input(dtype, (vec2_len,))
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_out = _ATEN_OP(ref_inp, ref_vec2)
    _assert_close(flag_gems.ger(inp, vec2), ref_out)


@pytest.mark.ger
def test_ger_element_order():
    # A product of two disjoint integer ramps makes every out[i, j] unique, so
    # a transposed or otherwise permuted write cannot compare equal.
    inp = torch.arange(1, 24, dtype=torch.int32, device=flag_gems.device)
    vec2 = torch.arange(100, 117, dtype=torch.int32, device=flag_gems.device)
    ref_out = _ATEN_OP(utils.to_reference(inp), utils.to_reference(vec2))
    utils.gems_assert_equal(flag_gems.ger(inp, vec2), ref_out)


# Sliced 1-D operands with stride > 1 and a nonzero storage offset. Both
# operands reach the kernel with their own stride, so they must be read through
# it rather than contiguously.
GER_STRIDED_SELECTIONS = _cases(
    [
        ((0, 64, 2), (0, 32, 2)),
        ((3, 35, 1), (5, 37, 3)),
        ((1, 17, 5), (2, 34, 2)),
    ],
    [((1, 17, 5), (2, 34, 2))],
)


@pytest.mark.ger
@pytest.mark.parametrize("inp_sel, vec2_sel", GER_STRIDED_SELECTIONS)
@pytest.mark.parametrize("dtype", GER_LAYOUT_DTYPES)
def test_ger_strided_input(inp_sel, vec2_sel, dtype):
    # One shared base buffer, so the two views also overlap in storage.
    base = _make_input(dtype, (64,))
    inp = base[inp_sel[0] : inp_sel[1] : inp_sel[2]]
    vec2 = base[vec2_sel[0] : vec2_sel[1] : vec2_sel[2]]
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_out = _ATEN_OP(ref_inp, ref_vec2)
    _assert_close(flag_gems.ger(inp, vec2), ref_out)


# Operands of different but compatible dtypes: the result takes the promoted
# common dtype, which the assertion below checks explicitly.
GER_MIXED_DTYPE_PAIRS = [
    pair
    for pair in _cases(
        [
            (torch.int8, torch.int32),
            (torch.uint8, torch.int8),
            (torch.int32, torch.int64),
            (torch.int64, torch.float16),
            (torch.bool, torch.float32),
            (torch.float16, torch.float32),
            (torch.float16, torch.bfloat16),
            (torch.float32, torch.float64),
            (torch.float32, torch.complex64),
        ],
        [(torch.float16, torch.float32)],
    )
    if all(
        (dtype is not torch.int64 or utils.int64_is_supported)
        and (dtype is not torch.bfloat16 or utils.bf16_is_supported)
        and (dtype is not torch.float64 or utils.fp64_is_supported)
        for dtype in pair
    )
]


@pytest.mark.ger
@pytest.mark.parametrize("dtype_pair", GER_MIXED_DTYPE_PAIRS)
def test_ger_mixed_dtype(dtype_pair):
    self_dtype, vec2_dtype = dtype_pair
    # Unequal lengths keep the two operands distinguishable in the output.
    inp = _make_input(self_dtype, (32,))
    vec2 = _make_input(vec2_dtype, (48,))
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_out = _ATEN_OP(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)

    assert res_out.dtype == torch.result_type(inp, vec2)
    _assert_close(res_out, ref_out)


# d(out)/d(self) = grad_out @ vec2 and d(out)/d(vec2) = grad_out^T @ self, so
# both operand orders are covered and a nonconstant shared upstream gradient
# keeps both reductions exercised.
GER_BACKWARD_PAIRS = _cases([(64, 32), (32, 64)], [(64, 32)])


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair", GER_BACKWARD_PAIRS)
@pytest.mark.parametrize("dtype", GER_FLOAT_DTYPES + [torch.complex64])
def test_ger_backward(shape_pair, dtype):
    self_len, vec2_len = shape_pair
    inp = _make_input(dtype, (self_len,)).requires_grad_(True)
    vec2 = _make_input(dtype, (vec2_len,)).requires_grad_(True)
    upstream = _make_input(dtype, (self_len, vec2_len))
    ref_inp = utils.to_reference(inp).detach().requires_grad_(True)
    ref_vec2 = utils.to_reference(vec2).detach().requires_grad_(True)
    ref_upstream = utils.to_reference(upstream)

    ref_out = _ATEN_OP(ref_inp, ref_vec2)
    res_out = flag_gems.ger(inp, vec2)
    _assert_close(res_out.detach(), ref_out.detach())

    ref_dself, ref_dvec2 = torch.autograd.grad(
        ref_out, (ref_inp, ref_vec2), grad_outputs=ref_upstream
    )
    res_dself, res_dvec2 = torch.autograd.grad(
        res_out, (inp, vec2), grad_outputs=upstream
    )

    _assert_close(res_dself, ref_dself)
    _assert_close(res_dvec2, ref_dvec2)


# Fixed payloads rather than random draws, so every non-finite value is
# guaranteed to reach the product.
GER_SPECIAL_PAYLOADS = {
    "nan": [float("nan"), 0.0, -0.0, 1.0, -1.0],
    "inf": [float("inf"), float("-inf"), 0.0, -0.0, 1.0],
    "mixed": [float("nan"), float("inf"), float("-inf"), 0.0, -0.0],
}
GER_SPECIAL_SCENARIOS = _cases(list(GER_SPECIAL_PAYLOADS), ["mixed"])


@pytest.mark.ger
@pytest.mark.parametrize("scenario", GER_SPECIAL_SCENARIOS)
@pytest.mark.parametrize("dtype", GER_FLOAT_DTYPES + [torch.complex64])
def test_ger_special_values(dtype, scenario):
    inp = torch.tensor(
        GER_SPECIAL_PAYLOADS[scenario], dtype=torch.float32, device=flag_gems.device
    ).to(dtype)
    # A length that shares no factor with the payload length keeps the
    # non-finite values spread across the output rows.
    vec2 = _make_input(dtype, (7,))
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_out = _ATEN_OP(ref_inp, ref_vec2)
    _assert_close(flag_gems.ger(inp, vec2), ref_out)


# ger takes no scalar parameters, so the rank guard is its only shape
# validation: 0-dim, 2-dim and 3-dim operands are all rejected on either side.
GER_INVALID_RANK_CASES = [
    ((), (4,)),
    ((2, 3), (4,)),
    ((4,), ()),
    ((4,), (2, 3)),
    ((1, 1, 4), (4,)),
]


@pytest.mark.ger
@pytest.mark.parametrize("inp_shape, vec2_shape", GER_INVALID_RANK_CASES)
def test_ger_invalid_rank(inp_shape, vec2_shape):
    inp = _make_input(torch.float32, inp_shape)
    vec2 = _make_input(torch.float32, vec2_shape)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.ger(inp, vec2)


# fp8 operands would need an fp8 multiply kernel, which the NVIDIA backend does
# not provide ("mul_cuda" is not implemented for Float8), so the rejected cases
# are declared for that vendor only instead of being skipped at test time.
GER_FP8_REJECTED_DTYPES = (
    [torch.float8_e4m3fn, torch.float8_e5m2]
    if getattr(flag_gems, "vendor_name", "") == "nvidia"
    else []
)


@pytest.mark.ger
@pytest.mark.parametrize("dtype", GER_FP8_REJECTED_DTYPES)
def test_ger_unsupported_dtype(dtype):
    inp = torch.randn((4,), device=flag_gems.device).to(dtype)
    vec2 = torch.randn((4,), device=flag_gems.device).to(dtype)
    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.ger(inp, vec2)


# aten::ger.out is a real overload whose return value is the provided buffer
# itself, hence the identity check. Integral dtypes are included because the
# out overload accepts them even though the differentiable path does not.
GER_OUT_DTYPES = [torch.float32, torch.int32]
if utils.int64_is_supported:
    GER_OUT_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    GER_OUT_DTYPES.append(torch.bfloat16)
GER_OUT_SHAPE_PAIRS = _cases([(64, 64), (256, 256)], [(64, 64)])


@pytest.mark.ger
@pytest.mark.parametrize("shape_pair", GER_OUT_SHAPE_PAIRS)
@pytest.mark.parametrize("dtype", GER_OUT_DTYPES)
def test_ger_out(shape_pair, dtype):
    self_len, vec2_len = shape_pair
    inp = _make_input(dtype, (self_len,))
    vec2 = _make_input(dtype, (vec2_len,))
    ref_inp = utils.to_reference(inp)
    ref_vec2 = utils.to_reference(vec2)

    ref_buf = torch.empty(
        (self_len, vec2_len), dtype=ref_inp.dtype, device=ref_inp.device
    )
    torch.ops.aten.ger.out(ref_inp, ref_vec2, out=ref_buf)

    out = torch.empty((self_len, vec2_len), dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.ger(inp, vec2, out=out)

    assert res_out is out
    assert out.device == inp.device
    _assert_close(res_out, ref_buf)

    positional = torch.empty((self_len, vec2_len), dtype=dtype, device=flag_gems.device)
    assert flag_gems.ger(inp, vec2, positional) is positional
    _assert_close(positional, ref_buf)


@pytest.mark.ger
def test_ger_invalid_out():
    # Lengths 8 and 6 make a mismatched out buffer unambiguous in both dims.
    inp = _make_input(torch.float32, (8,))
    vec2 = _make_input(torch.float32, (6,))
    with pytest.raises(ValueError):
        flag_gems.ger(inp, vec2, out=torch.empty((8, 5), device=flag_gems.device))
    with pytest.raises(ValueError):
        flag_gems.ger(
            inp,
            vec2,
            out=torch.empty((8, 6), dtype=torch.float16, device=flag_gems.device),
        )
    with pytest.raises(TypeError):
        flag_gems.ger(inp, vec2, out=object())


@pytest.mark.ger
def test_ger_non_tensor_operand():
    inp = _make_input(torch.float32, (8,))
    # A Python scalar is rejected on either side; ger does not accept one.
    with pytest.raises(TypeError):
        flag_gems.ger(inp, 3.0)
    with pytest.raises(TypeError):
        flag_gems.ger(3.0, inp)
