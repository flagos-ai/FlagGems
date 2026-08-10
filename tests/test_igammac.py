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
import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# fp64 is not supported on every platform (e.g. ascend, iluvatar).
_IGAMMAC_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64:
    _IGAMMAC_DTYPES.append(torch.float64)


@pytest.mark.igammac
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out = torch.special.gammaincc(ref_x, ref_y)

    with flag_gems.use_gems():
        res_out = torch.special.gammaincc(x, y)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.igammac_out
@pytest.mark.parametrize("shape", utils.POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_out(shape, dtype):
    x = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1
    y = torch.rand(shape, dtype=dtype, device=flag_gems.device) * 10 + 0.1

    ref_x = utils.to_reference(x, True)
    ref_y = utils.to_reference(y, True)
    ref_out_buf = torch.empty(shape, dtype=ref_x.dtype, device=ref_x.device)
    ref_out = torch.ops.aten.special_gammaincc.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.ops.aten.special_gammaincc.out(x, y, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_boundary_x_zero(dtype):
    """Q(a, 0) = 1 for all a > 0."""
    a_vals = torch.tensor(
        [0.5, 1.0, 2.0, 5.0, 10.0], dtype=dtype, device=flag_gems.device
    )
    x_vals = torch.zeros_like(a_vals)

    ref_a = utils.to_reference(a_vals, True)
    ref_x = utils.to_reference(x_vals, True)
    ref_out = torch.special.gammaincc(ref_a, ref_x)

    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_vals, x_vals)

    utils.gems_assert_close(res, ref_out, dtype)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_boundary_a_one(dtype):
    """Q(1, x) = exp(-x)."""
    x = torch.linspace(0.1, 20.0, 100, dtype=dtype, device=flag_gems.device)
    a = torch.ones_like(x)

    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.special.gammaincc(ref_a, ref_x)

    with flag_gems.use_gems():
        res = torch.special.gammaincc(a, x)

    utils.gems_assert_close(res, ref_out, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_boundary_large_x(dtype):
    """Q(a, x) -> 0 as x >> a."""
    a = torch.tensor([0.5, 1.0, 2.0], dtype=dtype, device=flag_gems.device)
    x = torch.full_like(a, 100.0)

    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.special.gammaincc(ref_a, ref_x)

    with flag_gems.use_gems():
        res = torch.special.gammaincc(a, x)

    utils.gems_assert_close(res, ref_out, dtype)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (1.0, 0.999),
        (1.0, 1.001),
        (20.0, 20.0),
    ],
)
def test_igammac_extreme_asym(dtype, a_val, x_val):
    """a≈x critical region — uses asymptotic expansion for a>20."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (78.0, 77.0),
        (78.0, 79.0),
        (100.0, 97.0),
        (200.0, 200.0),
        (500.0, 500.0),
        (2000.0, 2000.0),
        (10000.0, 10000.0),
    ],
)
def test_igammac_extreme_asym_large(dtype, a_val, x_val):
    """a≈x with a>20 (asymptotic expansion needed). Precision bound by algorithm diff."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (float("inf"), 1.0),
        (1.0, float("inf")),
        (float("inf"), float("inf")),
        (float("nan"), 1.0),
        (1.0, float("nan")),
        (-1.0, 1.0),
        (1.0, -1.0),
        (0.0, 0.0),
        (1e30, 1.0),
        (1.0, 1e30),
        (1e-30, 1.0),
    ],
)
def test_igammac_inf_nan(dtype, a_val, x_val):
    """Infinity and NaN boundary handling."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    res_v = float("nan") if torch.isnan(res) else res.item()
    ref_v = float("nan") if torch.isnan(ref) else ref.item()
    both_nan = math.isnan(res_v) and math.isnan(ref_v)
    match = both_nan or abs(res_v - ref_v) < 1e-5
    assert match, (
        f"Mismatch at (a={a_val}, x={x_val}): " f"igammac={res_v}, torch={ref_v}"
    )


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (0.01, 1000.0),
        (1000.0, 0.01),
        (0.1, 10000.0),
        (10000.0, 0.1),
        (1.0, 10000.0),
        (10000.0, 1.0),
    ],
)
def test_igammac_extreme_ratios(dtype, a_val, x_val):
    """Extreme a/x or x/a ratios."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (20.0, 0.01),
        (50.0, 0.01),
        (50.0, 0.05),
        (100.0, 0.01),
        (100.0, 0.001),
        (500.0, 0.001),
    ],
)
def test_igammac_large_a_small_x(dtype, a_val, x_val):
    """Large a with very small x (P≈1, 1−P subtraction zone)."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (a_val, a_val + 1.0 + eps)
        for a_val in [0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 200.0, 1000.0]
        for eps in [-0.1, -0.01, 0.0, 0.01, 0.1]
    ],
)
def test_igammac_series_cf_boundary(dtype, a_val, x_val):
    """x ≈ a+1 — the switchover between the series and continued-fraction paths."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (19.0, 19.0),
        (20.0, 20.0),
        (21.0, 21.0),
        (19.0, 24.0),
        (20.0, 25.0),
        (21.0, 26.0),
        (19.0, 15.0),
        (20.0, 16.0),
        (21.0, 17.0),
        (199.0, 199.0),
        (200.0, 200.0),
        (201.0, 201.0),
        (200.0, 260.0),
        (201.0, 261.0),
        (200.0, 150.0),
        (201.0, 151.0),
    ],
)
def test_igammac_asym_threshold(dtype, a_val, x_val):
    """Both sides of the asymptotic-expansion activation thresholds (a=20, a=200)."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
@pytest.mark.parametrize(
    "a_val,x_val",
    [
        (200.0, 260.0),
        (500.0, 650.0),
        (1000.0, 1100.0),
        (1000.0, 1050.0),
        (2000.0, 2200.0),
        (5000.0, 5500.0),
        (10000.0, 10500.0),
        (10000.0, 10100.0),
        (10000.0, 11000.0),
        (50000.0, 50500.0),
    ],
)
def test_igammac_large_a_moderate_x(dtype, a_val, x_val):
    """Large a with x moderately above a — outside the asymptotic region."""
    a_t = torch.tensor([a_val], dtype=dtype, device=flag_gems.device)
    x_t = torch.tensor([x_val], dtype=dtype, device=flag_gems.device)
    ref_a = utils.to_reference(a_t, True)
    ref_x = utils.to_reference(x_t, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a_t, x_t)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_log_uniform(dtype):
    """Random a, x spanning several orders of magnitude."""
    torch.manual_seed(0)
    n = 100000
    a = torch.exp(
        torch.rand(n, dtype=dtype, device=flag_gems.device) * math.log(1e6)
        + math.log(1e-2)
    )
    x = torch.exp(
        torch.rand(n, dtype=dtype, device=flag_gems.device) * math.log(1e6)
        + math.log(1e-3)
    )
    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref = torch.special.gammaincc(ref_a, ref_x)
    with flag_gems.use_gems():
        res = torch.special.gammaincc(a, x)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)
