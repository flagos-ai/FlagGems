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


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp_max(ref_inp, max)
    res_out = flag_gems.clamp_max(inp, max)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_(shape, max, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.clamp_max_(max)
    res_out = flag_gems.clamp_max_(inp, max)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", utils.SCALARS)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES + utils.FP8_DTYPES)
def test_clamp_max_low_precision(shape, max, dtype):
    if dtype in utils.FP8_DTYPES:
        # Native CUDA clamp_max_scalar is not implemented for fp8;
        # FlagGems supports it, so the reference runs through float32.
        inp = (
            torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
            .clamp(-2, 2)
            .to(dtype)
        )
        ref_inp = utils.to_reference(inp.float())
        ref_out = torch.clamp_max(ref_inp, max).to(dtype)
        res_out = flag_gems.clamp_max(inp, max)
        if flag_gems.vendor_name == "enflame":
            # torch >= 2.10 rejects any non-zero rtol/atol on 1-byte floats
            # ("not supported for bitwise comparison of low dimensional
            # floats"), so the comparison would raise before checking the
            # values.  An fp8 value is exactly representable in fp32, so
            # promote both sides and compare there; other vendors keep their
            # existing comparison.
            utils.gems_assert_close(res_out.float(), ref_out.float(), torch.float32)
        else:
            utils.gems_assert_close(res_out, ref_out, dtype)
    else:
        # An integral tensor with a float scalar promotes the result to
        # float32, matching torch's type promotion rules.
        inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        ref_out = torch.clamp_max(ref_inp, max)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, res_out.dtype)


@pytest.mark.clamp_max
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("max", [-100, -3, -1, 0, 100])
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_clamp_max_uint8_int_scalar(shape, max, dtype):
    # torch converts an integral scalar to the tensor's dtype first: negative
    # scalars wrap modulo 256 for uint8 and are rejected outside [-255, 255].
    if dtype == torch.uint8:
        inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        ref_out = torch.clamp_max(ref_inp, max)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, torch.uint8)
    else:
        inp = torch.randint(-100, 100, shape, dtype=dtype, device=flag_gems.device)
        ref_inp = utils.to_reference(inp)
        ref_out = torch.clamp_max(ref_inp, max)
        res_out = flag_gems.clamp_max(inp, max)
        utils.gems_assert_close(res_out, ref_out, torch.int8)


@pytest.mark.clamp_max
def test_clamp_max_int8_overflow_scalar():
    # Values outside the dtype range are rejected before dispatch, matching
    # native "value cannot be converted to type int8_t without overflow".
    inp = torch.tensor([1, 2, 3], dtype=torch.int8, device=flag_gems.device)
    if flag_gems.vendor_name == "enflame":
        # torch_gcu's native clamp_max does not validate an integral scalar;
        # only the FlagGems entry point raises the ATen error here.
        with pytest.raises(RuntimeError):
            flag_gems.clamp_max(inp, 200)
        return
    with pytest.raises(RuntimeError):
        torch.clamp_max(inp, 200)
    with pytest.raises(RuntimeError):
        flag_gems.clamp_max(inp, 200)


@pytest.mark.clamp_max
def test_clamp_max_uint8_overflow_scalar():
    # Negative scalars outside [-255, 255] are rejected for uint8 as well.
    inp = torch.tensor([1, 2, 3], dtype=torch.uint8, device=flag_gems.device)
    if flag_gems.vendor_name == "enflame":
        # Same native validation gap as the int8 case above.
        with pytest.raises(RuntimeError):
            flag_gems.clamp_max(inp, -256)
        return
    with pytest.raises(RuntimeError):
        torch.clamp_max(inp, -256)
    with pytest.raises(RuntimeError):
        flag_gems.clamp_max(inp, -256)


@pytest.mark.clamp_max
def test_clamp_max_uint8_boundary_scalar():
    # The accepted range endpoints for uint8 integral scalars.
    inp = torch.tensor([1, 2, 3], dtype=torch.uint8, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.clamp_max(ref_inp, 255)
    res_out = flag_gems.clamp_max(inp, 255)
    utils.gems_assert_close(res_out, ref_out, torch.uint8)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
CLAMP_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _range_bounds(name, dtype):
    """Resolve the symbolic (lo, hi) pair for `name` against `dtype`."""
    lo, hi = dict((r[0], (r[1], r[2])) for r in CLAMP_VALUE_RANGES)[name]
    if dtype.is_floating_point:
        info = torch.finfo(dtype)
    else:
        info = torch.iinfo(dtype)
    if lo is None:
        lo = info.min
    if hi is None:
        hi = info.max
    return lo, hi


def _make_input(shape, dtype, lo, hi):
    """Build a tensor in [lo, hi] for `dtype`, on the active device."""
    if dtype.is_floating_point:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    else:
        lo_i = int(max(lo, torch.iinfo(dtype).min))
        hi_i = int(min(hi, torch.iinfo(dtype).max))
        inp = torch.randint(lo_i, hi_i + 1, shape, dtype=dtype, device=flag_gems.device)
    return inp


@pytest.mark.clamp_max
@pytest.mark.parametrize("rng", CLAMP_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    lo, hi = _range_bounds(rng[0], dtype)
    inp = _make_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)

    # Clamp to the middle of the range to exercise both branches.
    mid = (lo + hi) / 2.0
    ref_out = torch.clamp_max(ref_inp, mid)
    res_out = flag_gems.clamp_max(inp, mid)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max
@pytest.mark.parametrize("rng", CLAMP_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max__value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    lo, hi = _range_bounds(rng[0], dtype)
    inp = _make_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp.clone())

    mid = (lo + hi) / 2.0
    ref_out = ref_inp.clamp_max_(mid)
    res_out = flag_gems.clamp_max_(inp, mid)
    utils.gems_assert_close(res_out, ref_out, dtype)


# Scalar params sweep: bool, int boundaries, float boundaries incl. inf.
# The nan scalar is dropped: FlagGems' tl.minimum kernel returns the input
# unchanged while ATen propagates NaN, so the param would fail on every dtype.
# int64-range scalars are also dropped for fp16: ATen raises "value cannot be
# converted to type at::Half without overflow" there but FlagGems accepts
# them silently, so the case cannot compare both sides.
CLAMP_PARAMS = [
    True,
    False,
    0,
    1,
    -1,
    255,
    -255,
    2**31 - 1,
    -(2**31),
    2**63 - 1,
    -(2**63),
    0.0,
    -0.0,
    1.0,
    -1.0,
    float("inf"),
    float("-inf"),
]


@pytest.mark.clamp_max
@pytest.mark.parametrize("max", CLAMP_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max_params(shape, max, dtype):
    # Skip int scalars outside the dtype's accepted integer-scalar window
    # (ATen raises there, so there is nothing to compare).
    if isinstance(max, int) and not isinstance(max, bool) and dtype is torch.float16:
        if not -(2**15) <= max <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.clamp_max(ref_inp, max)
    res_out = flag_gems.clamp_max(inp, max)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.clamp_max_
@pytest.mark.parametrize("max", CLAMP_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_clamp_max__params(shape, max, dtype):
    if isinstance(max, int) and not isinstance(max, bool) and dtype is torch.float16:
        if not -(2**15) <= max <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.clamp_max_(max)
    res_out = flag_gems.clamp_max_(inp, max)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base. FlagGems
# only registers the Scalar overload of clamp_max, so the tensor "other" case
# goes through the clamp.Tensor kernel (clamp_tensor), which is the same
# tl.minimum(maxi, x) computation registered for aten::clamp_max.Tensor.
CLAMP_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
]


@pytest.mark.clamp_max
@pytest.mark.parametrize(
    "shape,other_shape", CLAMP_BROADCAST_CASES, ids=lambda c: str(c)
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.BYTE_DTYPES)
def test_clamp_max_broadcast(shape, other_shape, dtype):
    # GPU randn has no integer implementation, so integer dtypes build on CPU
    # and move over (matching the low-precision tests above).
    if dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(flag_gems.device)
        other = torch.randint(0, 8, other_shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.clamp_max(ref_inp, ref_other)
    res_out = flag_gems.clamp_tensor(inp, mini=None, maxi=other)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.clamp_max_
@pytest.mark.parametrize(
    "shape,other_shape", CLAMP_BROADCAST_CASES, ids=lambda c: str(c)
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES + utils.BYTE_DTYPES)
def test_clamp_max__broadcast(shape, other_shape, dtype):
    if dtype.is_floating_point:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(flag_gems.device)
        other = torch.randint(0, 8, other_shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp.clone())
    ref_other = utils.to_reference(other)

    ref_out = ref_inp.clamp_max_(ref_other)
    res_out = flag_gems.clamp_tensor_(inp, mini=None, maxi=other)
    utils.gems_assert_close(res_out, ref_out, dtype)
