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


@pytest.mark.copysign
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign(shape, dtype):
    # Test copysign: magnitude of input, sign of other
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)
    ref_out = torch.copysign(ref_input, ref_other)

    res_out = flag_gems.copysign(input, other)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.copysign_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_out(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(input)

    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)
    ref_out = torch.empty_like(ref_input)

    torch.copysign(ref_input, ref_other, out=ref_out)
    flag_gems.copysign_out(input, other, out=out)

    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.copysign
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_copysign_byte(shape, dtype):
    # torch promotes integral copysign inputs to float32. Sweep the full dtype
    # range: values >= 128 in uint8 have the top bit set but are not negative,
    # and abs(int8 min) must not wrap in integer arithmetic.
    info = torch.iinfo(dtype)
    input = torch.randint(info.min, info.max + 1, shape, dtype=dtype).to(
        flag_gems.device
    )
    other = torch.randint(info.min, info.max + 1, shape, dtype=dtype).to(
        flag_gems.device
    )

    ref_input = utils.to_reference(input)
    ref_other = utils.to_reference(other)
    ref_out = torch.copysign(ref_input, ref_other)
    res_out = flag_gems.copysign(input, other)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
COPYSIGN_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _make_copysign_input(shape, dtype, lo, hi):
    """Build a float tensor in [lo, hi] on the active device."""
    if lo is None or hi is None:
        info = torch.finfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    return inp


@pytest.mark.copysign
@pytest.mark.parametrize("rng", COPYSIGN_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_copysign_input(shape, dtype, lo, hi)
    other = _make_copysign_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.copysign(ref_inp, ref_other)
    res_out = flag_gems.copysign(inp, other)
    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.copysign_out
@pytest.mark.parametrize("rng", COPYSIGN_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_out_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_copysign_input(shape, dtype, lo, hi)
    other = _make_copysign_input(shape, dtype, lo, hi)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)
    ref_out = torch.empty_like(ref_inp)

    torch.copysign(ref_inp, ref_other, out=ref_out)
    flag_gems.copysign_out(inp, other, out=out)
    utils.gems_assert_close(out, ref_out, dtype)


@pytest.mark.copysign_
@pytest.mark.parametrize("rng", COPYSIGN_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign__value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_copysign_input(shape, dtype, lo, hi)
    other = _make_copysign_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp.clone(), True)
    ref_other = utils.to_reference(other, True)

    ref_out = ref_inp.copysign_(ref_other)
    res_out = flag_gems.copysign_(inp, other)
    utils.gems_assert_close(res_out, ref_out, dtype)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
COPYSIGN_PARAMS = [
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


@pytest.mark.copysign
@pytest.mark.parametrize("other", COPYSIGN_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_params(shape, other, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.copysign(ref_inp, other)
    res_out = flag_gems.copysign(inp, other)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.copysign_out
@pytest.mark.parametrize("other", COPYSIGN_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_out_params(shape, other, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.empty_like(ref_inp)
    torch.copysign(ref_inp, other, out=ref_out)
    flag_gems.copysign_out(inp, other, out=out)
    utils.gems_assert_close(out, ref_out, dtype, equal_nan=True)


@pytest.mark.copysign_
@pytest.mark.parametrize("other", COPYSIGN_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign__params(shape, other, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone(), True)

    ref_out = ref_inp.copysign_(other)
    res_out = flag_gems.copysign_(inp, other)
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
COPYSIGN_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
]


@pytest.mark.copysign
@pytest.mark.parametrize("shapes", COPYSIGN_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_copysign_broadcast(shapes, dtype):
    shape, other_shape = shapes
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    other = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)
    ref_other = utils.to_reference(other)

    ref_out = torch.copysign(ref_inp, ref_other)
    res_out = flag_gems.copysign(inp, other)
    utils.gems_assert_close(res_out, ref_out, dtype)
