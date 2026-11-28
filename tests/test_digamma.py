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


@pytest.mark.digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma(shape, dtype):
    inp = torch.rand(shape, dtype=dtype, device=flag_gems.device) + 1.0
    ref_inp = utils.to_reference(inp)

    ref_out = torch.digamma(ref_inp)
    res_out = flag_gems.digamma(inp)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.digamma
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.BYTE_DTYPES)
def test_digamma_byte(shape, dtype):
    # torch promotes integral digamma inputs to float32.
    inp = torch.randint(1, 8, shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.digamma(ref_inp)
    res_out = flag_gems.digamma(inp)

    utils.gems_assert_close(res_out, ref_out, torch.float32)


@pytest.mark.digamma_
def test_digamma_int_rejects_inplace():
    # torch.digamma_ rejects integral inputs -- "result type Float can't be
    # cast to the desired output type Int"; so does FlagGems.
    inp = torch.tensor([1, 2, 3], dtype=torch.int32, device=flag_gems.device)
    ref_inp = inp.cpu()
    with pytest.raises(RuntimeError):
        ref_inp.digamma_()
    with pytest.raises(RuntimeError):
        flag_gems.digamma_(inp)


# ---------------------------------------------------------------------------
# Extended coverage: value-range grid x shape x dtype, scalar params, broadcast.
# ---------------------------------------------------------------------------

# Five value ranges per the coverage spec, over [-1, 1] plus the dtype edges.
DIGAMMA_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),
    ("[min,0]", None, 0.0),
]


def _make_digamma_input(shape, dtype, lo, hi):
    """Build a float tensor in [lo, hi] on the active device."""
    if lo is None or hi is None:
        info = torch.finfo(dtype)
        lo = info.min if lo is None else lo
        hi = info.max if hi is None else hi
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device).uniform_(lo, hi)
    return inp


@pytest.mark.digamma
@pytest.mark.parametrize("rng", DIGAMMA_VALUE_RANGES, ids=lambda r: r[0])
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_value_ranges(shape, rng, dtype):
    _, lo, hi = rng
    inp = _make_digamma_input(shape, dtype, lo, hi)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.digamma(ref_inp)
    res_out = flag_gems.digamma(inp)
    _assert_digamma_close(res_out, ref_out, inp, dtype)


# Scalar params sweep: bool, int boundaries, float boundaries incl. nan/inf.
DIGAMMA_PARAMS = [
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
    0.5,
    -0.5,
    2.5,
    -2.5,
    float("inf"),
    float("-inf"),
    float("nan"),
]


@pytest.mark.digamma
@pytest.mark.parametrize("val", DIGAMMA_PARAMS)
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_params(shape, val, dtype):
    # int64-range scalars overflow at::Half on GPU init, matching ATen's
    # integer-scalar conversion window; skip those combos.
    if dtype is torch.float16 and isinstance(val, int) and not isinstance(val, bool):
        if not -(2**15) <= val <= 2**15 - 1:
            pytest.skip("int64-range scalar overflows fp16 in ATen")
    inp = torch.full(shape, float(val), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.digamma(ref_inp)
    res_out = flag_gems.digamma(inp)
    _assert_digamma_close(res_out, ref_out, inp, dtype)


# Broadcast shapes: 1-D and multi-D broadcasts against the 4D base.
DIGAMMA_BROADCAST_CASES = [
    ((16, 128, 64, 60), (60,)),
    ((16, 128, 64, 60), (1, 60)),
    ((16, 128, 64, 60), (1, 1, 64, 60)),
    ((16, 128, 64, 60), (16, 1, 1, 1)),
    ((20, 320, 15), (15,)),
    ((1024, 1024), (1,)),
]


@pytest.mark.digamma
@pytest.mark.parametrize("shapes", DIGAMMA_BROADCAST_CASES, ids=lambda c: str(c))
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_digamma_broadcast(shapes, dtype):
    # Unary op: broadcast means a stride-0 expanded view of the small tensor.
    shape, other_shape = shapes
    inp = torch.randn(other_shape, dtype=dtype, device=flag_gems.device)
    expanded = inp.expand(shape)
    ref_inp = utils.to_reference(expanded)

    ref_out = torch.digamma(ref_inp)
    res_out = flag_gems.digamma(expanded)
    _assert_digamma_close(res_out, ref_out, expanded, dtype)


def _assert_digamma_close(res_out, ref_out, inp, dtype):
    """digamma comparison that tolerates the pole-neighbourhood divergence.

    FlagGems computes digamma in fp32 while ATen's CPU reference computes in
    double internally. Near the non-positive-integer poles digamma grows like
    1/(x - pole), so a few-ULP difference in x becomes a relative difference
    that grows without bound as the sample approaches the pole; large-sample
    tests always land points within a few ULP of a pole. Elements within 1e-2
    of a pole are excluded from the comparison instead of loosening the
    tolerance for every element.
    """
    res = res_out.cpu().float()
    ref = ref_out.float().cpu()
    x = inp.float().cpu()
    lower = torch.floor(x)
    dist = torch.minimum(x - lower, lower + 1 - x)
    near_pole = (x <= 0) & (dist < 1e-2)
    bad = (res - ref).abs() > 3e-4 + 1e-3 * ref.abs()
    bad = bad & ~((res.isnan()) & (ref.isnan()))
    bad = bad & ~near_pole
    assert (
        not bad.any()
    ), f"{bad.sum().item()} elements diverge beyond pole-neighbourhood tolerance"
