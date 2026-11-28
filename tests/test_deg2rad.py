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


@pytest.mark.deg2rad
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.deg2rad(ref_inp)
    res_out = flag_gems.deg2rad(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.deg2rad_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_inp.deg2rad_()
    flag_gems.deg2rad_(inp)

    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.empty_like(ref_inp)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty_like(inp)
    flag_gems.deg2rad_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_deg2rad_out_byte(shape, dtype):
    # torch promotes integral deg2rad inputs to float32 before scaling.
    inp = torch.randint(1, 8, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty(shape, dtype=torch.float32, device=ref_inp.device)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    flag_gems.deg2rad_out(inp, out=res_out)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
DEG2RAD_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _make_deg2rad_input(shape, dtype, lo, hi):
    """Build a float tensor in [lo, hi] on the active device."""
    if lo is None or hi is None:
        info = torch.finfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    return inp


@pytest.mark.deg2rad
@pytest.mark.parametrize("rng", DEG2RAD_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_deg2rad_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.deg2rad(ref_inp)
    res_out = flag_gems.deg2rad(inp)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.deg2rad_
@pytest.mark.parametrize("rng", DEG2RAD_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad__value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_deg2rad_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp, True)

    ref_inp.deg2rad_()
    flag_gems.deg2rad_(inp)
    utils.gems_assert_close(inp, ref_inp, dtype)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("rng", DEG2RAD_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_out_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_deg2rad_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty_like(inp)
    flag_gems.deg2rad_out(inp, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
DEG2RAD_PARAMS = [
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
    float("nan"),
]


@pytest.mark.deg2rad
@pytest.mark.parametrize("val", DEG2RAD_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_params(shape, val, dtype):
    # int64-range scalars overflow at::Half on GPU init, matching ATen's
    # integer-scalar conversion window; skip those combos.
    if dtype is torch.float16 and isinstance(val, int) and not isinstance(val, bool):
        if not -(2**15) <= val <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    inp = torch.full(shape, float(val), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.deg2rad(ref_inp)
    res_out = flag_gems.deg2rad(inp)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.deg2rad_out
@pytest.mark.parametrize("val", DEG2RAD_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_out_params(shape, val, dtype):
    if dtype is torch.float16 and isinstance(val, int) and not isinstance(val, bool):
        if not -(2**15) <= val <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    inp = torch.full(shape, float(val), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.deg2rad(ref_inp, out=ref_out)

    res_out = torch.empty_like(inp)
    flag_gems.deg2rad_out(inp, out=res_out)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
DEG2RAD_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
]


@pytest.mark.deg2rad
@pytest.mark.parametrize("shapes", DEG2RAD_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_deg2rad_broadcast(shapes, dtype):
    # Unary op, so "broadcast" means feeding a stride-0 expanded view: the
    # small tensor's values, seen through the large shape's strides.
    shape, other_shape = shapes
    inp = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    expanded = inp.expand(shape)
    ref_inp = utils.to_reference(expanded)

    ref_out = torch.deg2rad(ref_inp)
    res_out = flag_gems.deg2rad(expanded)
    utils.gems_assert_close(res_out, ref_out, dtype)
