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

from .accuracy_utils import (
    BYTE_DTYPES,
    FLOAT_DTYPES,
    POINTWISE_SHAPES,
    SCALARS,
    gems_assert_close,
    ref_fmod,
    to_reference,
)


@pytest.mark.fmod
@pytest.mark.fmod_tensor
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_tensor(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = ref_fmod(ref_inp1, ref_inp2)
    res_out = flag_gems.fmod_tensor(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmod
@pytest.mark.fmod_scalar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar if scalar != 0 else 1.0
    ref_inp1 = to_reference(inp1, True)
    ref_out = ref_fmod(ref_inp1, inp2)
    res_out = flag_gems.fmod_scalar(inp1, inp2)
    atol = 1e-3 if abs(scalar) < 0.01 else 1e-4
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.fmod_tensor_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_tensor_inplace(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.where(inp2 == 0, torch.ones_like(inp2), inp2)
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_inp2 = to_reference(inp2, True)
    ref_out = ref_fmod(ref_inp1, ref_inp2)
    res_out = flag_gems.fmod_tensor_(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmod_scalar_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar_inplace(shape, scalar, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = scalar if scalar != 0 else 1.0
    ref_inp1 = to_reference(inp1.clone(), True)
    ref_out = ref_fmod(ref_inp1, inp2)
    res_out = flag_gems.fmod_scalar_(inp1, inp2)
    atol = 1e-3 if abs(scalar) < 0.01 else 1e-4
    gems_assert_close(res_out, ref_out, dtype, atol=atol)


@pytest.mark.fmod_scalar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BYTE_DTYPES)
def test_fmod_scalar_byte(shape, dtype):
    # fmod with an integral scalar keeps the input dtype.
    inp1 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)

    ref_out = torch.fmod(ref_inp1, 3)
    res_out = flag_gems.fmod_scalar(inp1, 3)

    assert res_out.dtype == ref_out.dtype
    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.fmod_scalar_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", BYTE_DTYPES)
def test_fmod_scalar_inplace_byte(shape, dtype):
    inp1 = torch.randint(1, 100, shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1.clone())

    torch.fmod(ref_inp1, 3, out=ref_inp1)
    res_out = flag_gems.fmod_scalar_(inp1, 3)

    assert res_out.dtype == ref_inp1.dtype
    gems_assert_close(res_out, ref_inp1, dtype)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
FMOD_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _make_fmod_input(shape, dtype, lo, hi):
    """Build a float tensor in [lo, hi] on the active device."""
    if lo is None or hi is None:
        info = torch.finfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    return inp


@pytest.mark.fmod_scalar
@pytest.mark.parametrize("rng", FMOD_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp1 = _make_fmod_input(shape, dtype, lo, hi)
    ref_inp1 = to_reference(inp1, True)

    ref_out = ref_fmod(ref_inp1, 3.0)
    res_out = flag_gems.fmod_scalar(inp1, 3.0)
    gems_assert_close(res_out, ref_out, dtype)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
FMOD_PARAMS = [
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


@pytest.mark.fmod_scalar
@pytest.mark.parametrize("val", FMOD_PARAMS)
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar_params(shape, val, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = val if val != 0 else 1.0
    ref_inp1 = to_reference(inp1, True)

    ref_out = ref_fmod(ref_inp1, inp2)
    res_out = flag_gems.fmod_scalar(inp1, inp2)
    gems_assert_close(res_out, ref_out, dtype, atol=1e-3, equal_nan=True)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
FMOD_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
]


@pytest.mark.fmod_scalar
@pytest.mark.parametrize("shapes", FMOD_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_fmod_scalar_broadcast(shapes, dtype):
    # Unary w.r.t. the tensor (scalar divisor): broadcast means a stride-0
    # expanded view of the small tensor.
    shape, other_shape = shapes
    inp1 = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    expanded = inp1.expand(shape)
    ref_inp1 = to_reference(expanded, True)

    ref_out = ref_fmod(ref_inp1, 3.0)
    res_out = flag_gems.fmod_scalar(expanded, 3.0)
    gems_assert_close(res_out, ref_out, dtype)
