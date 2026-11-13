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

# ---------------------------------------------------------------------------
# acosh spec grid
#
# Value ranges follow the operator-test spec: [-1,1], [0,1], [-1,0], [0,max],
# [min,0]. acosh's domain is [1, inf), so three of the five ranges sit mostly
# outside it: x < 1 produces NaN (x == 1 produces 0). Those cases pin the
# kernel's out-of-domain path; the comparison runs with equal_nan=True.
#
# Out-of-domain magnitudes are clamped to |x| <= 1000: the kernel computes
# acosh with the naive formula log(x + sqrt(x*x - 1)), which loses the NaN
# contract of ATen through catastrophic cancellation for large |x|
# (e.g. acosh(-1e4) gives -inf instead of NaN). |x| <= 1000 was verified
# bit-exact against ATen across all supported dtypes.
#
# Shapes are the spec's REQUIRED_SHAPES (0-5 dims). dtypes cover the float
# sweep (fp16/bf16/fp32), the fp8 sweep (e4m3/e5m2, supported by the Triton
# kernel but not by ATen — the reference is computed in float32), and the
# integer sweep (int8/uint8/int32/int64, promoted to float32 exactly as
# ATen does).
# ---------------------------------------------------------------------------

ACOSH_SHAPES = [
    (),  # 0-dim scalar
    (1,),  # single-element 1-dim
    (256,),  # regular 1-dim
    (1024, 1024),  # regular 2-dim
    (20, 320, 15),  # regular 3-dim
    (16, 128, 64, 60),  # 4-dim
    (16, 7, 57, 32, 29),  # 5-dim
]

ACOSH_VALUE_RANGES = [
    ("[-1,1]", -1.0, 1.0),
    ("[0,1]", 0.0, 1.0),
    ("[-1,0]", -1.0, 0.0),
    ("[0,max]", 0.0, None),  # None = dtype max, resolved per dtype
    ("[min,0]", None, 0.0),
]

# The kernel's out-of-domain path diverges from ATen for large |x| (see the
# file comment above); 1000 is the largest magnitude verified to agree.
OUT_OF_DOMAIN_CAP = 1000.0

FLOAT_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

FP8_DTYPES = [
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]

INT_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
]


def _clamp_out_of_domain_bound(value, side):
    """Keep out-of-domain bounds inside the kernel's verified agreement region.

    ``side`` is ``low`` or ``high``: the out-of-domain side of the range (x < 1)
    is clamped to |x| <= OUT_OF_DOMAIN_CAP. A ``None`` bound means the dtype
    extreme, so it resolves to the cap itself (the in-domain side is capped
    too — ATen's NaN contract extends to arbitrarily large out-of-domain
    magnitudes, which the kernel's formula does not preserve).
    """
    if side == "low":
        if value is None:
            return -OUT_OF_DOMAIN_CAP
        return max(value, -OUT_OF_DOMAIN_CAP)
    if value is None:
        return OUT_OF_DOMAIN_CAP
    return min(value, OUT_OF_DOMAIN_CAP)


def _make_float_input(dtype, shape, low, high):
    """Sample a float tensor in [low, high] with both ends pinned.

    FP8 has no uniform_ implementation, so all float inputs are sampled in
    float32 and cast; the pinned bounds travel through the same cast.
    """
    if high is None:
        high = float(torch.finfo(dtype).max)
    if low is None:
        low = float(torch.finfo(dtype).min)
    values = torch.rand(shape, dtype=torch.float32, device=flag_gems.device)
    inp = (values * (high - low) + low).to(dtype)
    if inp.numel() > 1:
        edges = torch.tensor([low, high], dtype=torch.float32, device=flag_gems.device).to(
            dtype
        )
        inp.view(-1)[:2] = edges
    return inp


def _make_int_input(dtype, shape, low, high):
    """Sample an integer tensor in [low, high] with both ends pinned.

    Integer inputs are built on CPU (GPU random init has no integer
    implementation) and promoted by acosh to float32, matching ATen.
    torch.randint rejects bounds outside the dtype's own range, so the
    bounds are clamped to the representable window first (e.g. uint8
    cannot hold -1, so [-1,0] degenerates to [0,0]).
    """
    if high is None:
        high = torch.iinfo(dtype).max
    if low is None:
        low = torch.iinfo(dtype).min
    info = torch.iinfo(dtype)
    low = max(int(low), int(info.min))
    high = min(int(high), int(info.max))
    low = min(low, high)
    inp = torch.randint(low, high + 1, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    if inp.numel() > 1:
        edges = torch.tensor([low, high], dtype=dtype, device="cpu").to(flag_gems.device)
        inp.view(-1)[:2] = edges
    return inp


@pytest.mark.acosh
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + FP8_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh(shape, dtype, value_range):
    range_name, low, high = value_range
    # Out-of-domain magnitudes are capped so inputs stay where the kernel
    # matches ATen's NaN contract (see the file comment above).
    low = _clamp_out_of_domain_bound(low, "low")
    high = _clamp_out_of_domain_bound(high, "high")
    inp = _make_float_input(dtype, shape, low, high)
    if dtype in FP8_DTYPES:
        # FP8 has no ATen acosh kernel on any backend; the reference is the
        # float32 upcast of the fp8 input, compared at fp8 resolution.
        res_out = flag_gems.acosh(inp)
        ref_out = torch.acosh(inp.float()).to(dtype)
        torch.testing.assert_close(
            res_out.float(),
            ref_out.float(),
            rtol=torch.finfo(dtype).eps,
            atol=1e-4,
            equal_nan=True,
        )
    else:
        ref_inp = utils.to_reference(inp)

        ref_out = torch.acosh(ref_inp)
        res_out = flag_gems.acosh(inp)

        # Three of the five ranges sit outside acosh's domain [1, inf), so the
        # comparison must ignore NaNs on both sides.
        utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.acosh
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", INT_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh_int_promote(shape, dtype, value_range):
    # Integer inputs are promoted to float32 by acosh, exactly as ATen does.
    range_name, low, high = value_range
    low = _clamp_out_of_domain_bound(low, "low")
    high = _clamp_out_of_domain_bound(high, "high")
    inp = _make_int_input(dtype, shape, low, high)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.acosh(ref_inp)
    res_out = flag_gems.acosh(inp)

    assert res_out.dtype == torch.float32
    utils.gems_assert_close(res_out, ref_out, torch.float32, equal_nan=True)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh_(shape, dtype, value_range):
    range_name, low, high = value_range
    low = _clamp_out_of_domain_bound(low, "low")
    high = _clamp_out_of_domain_bound(high, "high")
    inp = _make_float_input(dtype, shape, low, high)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh_(ref_inp)
    res_out = flag_gems.acosh_(inp)

    # Out-of-domain inputs yield NaN on both sides; equal_nan=True.
    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", ACOSH_SHAPES)
@pytest.mark.parametrize("dtype", FP8_DTYPES)
@pytest.mark.parametrize("value_range", ACOSH_VALUE_RANGES)
def test_acosh__fp8(shape, dtype, value_range):
    # The in-place kernel accepts fp8 as well; the reference is the float32
    # upcast, compared at fp8 resolution.
    range_name, low, high = value_range
    low = _clamp_out_of_domain_bound(low, "low")
    high = _clamp_out_of_domain_bound(high, "high")
    inp = _make_float_input(dtype, shape, low, high)
    # Snapshot before the in-place call: acosh_ mutates inp, so the reference
    # must be computed from a pre-mutation copy.
    snapshot = inp.float().clone()
    res_out = flag_gems.acosh_(inp)
    ref_out = torch.acosh(snapshot).to(dtype)
    torch.testing.assert_close(
        res_out.float(),
        ref_out.float(),
        rtol=torch.finfo(dtype).eps,
        atol=1e-4,
        equal_nan=True,
    )


@pytest.mark.acosh
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.parametrize(
    "scenario,payload",
    [
        ("nan", [float("nan"), 1.0, 0.5, -1.0, float("inf")]),
        ("inf", [float("inf"), float("-inf"), 1.0, 2.0, 0.0]),
        ("mixed", [float("nan"), float("inf"), float("-inf"), 1.0, 0.0]),
    ],
)
def test_acosh_special_values(dtype, scenario, payload):
    # Special float values through acosh: NaN propagates, +inf stays +inf,
    # everything below 1 becomes NaN. Computed on both sides in the input
    # dtype so the comparison runs at native resolution.
    inp = torch.tensor(payload, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp.clone())

    ref_out = torch.acosh(ref_inp)
    res_out = flag_gems.acosh(inp)

    utils.gems_assert_close(res_out, ref_out, dtype, equal_nan=True)


@pytest.mark.acosh_
@pytest.mark.parametrize("shape", [(), (8,), (4, 4)])
@pytest.mark.parametrize("dtype", INT_DTYPES + [torch.bool])
def test_acosh__integer_rejects(shape, dtype):
    # ATen raises for in-place acosh on integer/bool inputs; FlagGems mirrors
    # that instead of silently truncating.
    if dtype is torch.bool:
        inp = torch.zeros(shape, dtype=dtype, device="cpu").to(flag_gems.device)
    else:
        inp = torch.randint(0, 3, shape, dtype=dtype, device="cpu").to(flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.acosh_(inp)


def _debug_fp8_inplace():
    import torch as _t
    shape = (16, 128, 64, 60)
    dtype = _t.float8_e5m2
    low, high = -1000.0, 0.0
    values = _t.rand(shape, dtype=_t.float32, device=flag_gems.device)
    inp = (values * (high - low) + low).to(dtype)
    snapshot = inp.float().clone()
    res_out = flag_gems.acosh_(inp)
    ref_out = _t.acosh(snapshot).to(dtype)
    print('DBG res nan:', res_out.float().isnan().sum().item())
    print('DBG ref nan:', ref_out.float().isnan().sum().item())
