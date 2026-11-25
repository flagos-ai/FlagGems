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


@pytest.mark.floor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.floor(ref_inp)
    res_out = flag_gems.floor(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_backward
@pytest.mark.parametrize("shape", [(), (0,), (3, 4)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_floor_backward(shape, dtype, noncontiguous):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if noncontiguous and inp.ndim == 2:
        inp = inp.T
    grad = torch.randn_like(inp)
    ref = utils.to_reference(inp.detach(), False).requires_grad_()
    actual = inp.detach().requires_grad_()
    expected = torch.autograd.grad(
        torch.floor(ref), ref, utils.to_reference(grad, False)
    )[0]
    result = torch.autograd.grad(flag_gems.floor(actual), actual, grad)[0]
    utils.gems_assert_equal(result, expected)


@pytest.mark.floor_
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = ref_inp.floor_()
    res_out = flag_gems.floor_(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.floor_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    out = torch.empty_like(inp)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty_like(ref_inp)

    torch.floor(ref_inp, out=ref_out)
    flag_gems.floor_out(inp, out=out)

    utils.gems_assert_equal(out, ref_out)


@pytest.mark.floor
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_floor_byte(shape, dtype):
    # floor on integers is the identity and keeps the input dtype.
    inp = torch.randint(0, 8, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.floor(ref_inp)
    res_out = flag_gems.floor(inp)

    assert res_out.dtype == ref_out.dtype
    utils.gems_assert_equal(res_out, ref_out)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
FLOOR_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _make_floor_input(shape, dtype, lo, hi):
    """Build a float tensor in [lo, hi] on the active device."""
    if lo is None or hi is None:
        info = torch.finfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    return inp


@pytest.mark.floor
@pytest.mark.parametrize("rng", FLOOR_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_floor_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.floor(ref_inp)
    res_out = flag_gems.floor(inp)
    utils.gems_assert_equal(res_out, ref_out)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
FLOOR_PARAMS = [
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


@pytest.mark.floor
@pytest.mark.parametrize("val", FLOOR_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_params(shape, val, dtype):
    # int64-range scalars overflow at::Half on GPU init, matching ATen's
    # integer-scalar conversion window; skip those combos.
    if dtype is torch.float16 and isinstance(val, int) and not isinstance(val, bool):
        if not -(2**15) <= val <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    inp = torch.full(shape, float(val), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.floor(ref_inp)
    res_out = flag_gems.floor(inp)
    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
FLOOR_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
]


@pytest.mark.floor
@pytest.mark.parametrize("shapes", FLOOR_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_floor_broadcast(shapes, dtype):
    # Unary op: broadcast means a stride-0 expanded view of the small tensor.
    shape, other_shape = shapes
    inp = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    expanded = inp.expand(shape)
    ref_inp = utils.to_reference(expanded)

    ref_out = torch.floor(ref_inp)
    res_out = flag_gems.floor(expanded)
    utils.gems_assert_equal(res_out, ref_out)
