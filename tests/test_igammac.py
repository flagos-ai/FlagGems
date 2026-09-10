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
    ref_out = torch.igammac(ref_x, ref_y)

    res_out = flag_gems.igammac(x, y)

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
    ref_out = torch.ops.aten.igammac.out(ref_x, ref_y, out=ref_out_buf)

    res_out_buf = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.igammac(x, y, out=res_out_buf)

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
    ref_out = torch.igammac(ref_a, ref_x)

    res = flag_gems.igammac(a_vals, x_vals)

    utils.gems_assert_close(res, ref_out, dtype)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_boundary_a_one(dtype):
    """Q(1, x) = exp(-x)."""
    x = torch.linspace(0.1, 20.0, 100, dtype=dtype, device=flag_gems.device)
    a = torch.ones_like(x)

    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.igammac(ref_a, ref_x)

    res = flag_gems.igammac(a, x)

    utils.gems_assert_close(res, ref_out, dtype, atol=1e-5)


@pytest.mark.igammac
@pytest.mark.parametrize("dtype", _IGAMMAC_DTYPES)
def test_igammac_boundary_large_x(dtype):
    """Q(a, x) -> 0 as x >> a."""
    a = torch.tensor([0.5, 1.0, 2.0], dtype=dtype, device=flag_gems.device)
    x = torch.full_like(a, 100.0)

    ref_a = utils.to_reference(a, True)
    ref_x = utils.to_reference(x, True)
    ref_out = torch.igammac(ref_a, ref_x)

    res = flag_gems.igammac(a, x)

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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a_t, x_t)
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
    ref = torch.igammac(ref_a, ref_x)
    res = flag_gems.igammac(a, x)
    utils.gems_assert_close(res, ref, dtype, atol=1e-5)


def _lgamma_lanczos(x):
    """log-gamma via Lanczos (g=7, n=9) using only device-native torch
    primitives. torch.lgamma falls back to CPU on NPU, so we inline it.
    Kept byte-identical to benchmark/test_igammac.py so the benchmark baseline
    is exactly what this test validates."""
    x = x.to(torch.float32)
    zm1 = x - 1.0
    t = zm1 + 7.5
    return (
        0.5 * torch.log(torch.tensor(6.283185307179586, device=x.device))
        + (zm1 + 0.5) * torch.log(t)
        - t
        + torch.log(
            0.99999999999980993
            + 676.5203681218851 / (zm1 + 1.0)
            + -1259.1392167224028 / (zm1 + 2.0)
            + 771.32342877765313 / (zm1 + 3.0)
            + -176.61502916214059 / (zm1 + 4.0)
            + 12.507343278686905 / (zm1 + 5.0)
            + -0.13857109526572012 / (zm1 + 6.0)
            + 9.9843695780195716e-6 / (zm1 + 7.0)
            + 1.5056327351493116e-7 / (zm1 + 8.0)
        )
    )


# Same iteration count as the kernel's series branch (SERIES_ITERS=50 in
# _launch_igammac) so the comparison is fair: both sides evaluate the same
# number of series terms (measured: 50 terms already converge on the benchmark
# input domain, matching the 128-term result to ~4e-6).
_SERIES_ITERS = 50


def _igammac_composed(a, x):
    """Device-native fixed-N power-series reference for Q(a, x)."""
    af = a.to(torch.float32)
    xf = x.to(torch.float32)
    log_gamma_a = _lgamma_lanczos(af)
    log_x_term = af * torch.log(xf) - xf - log_gamma_a
    term = torch.ones_like(af) / af
    series_sum = term.clone()
    for i in range(1, _SERIES_ITERS):
        term = term * xf / (af + i)
        series_sum = series_sum + term
    q = 1.0 - torch.exp(log_x_term) * series_sum
    return torch.clamp(q, 0.0, 1.0)


@pytest.mark.igammac
@pytest.mark.parametrize("shape", [(4096,), (64, 64), (10000, 256)])
def test_igammac_composed_baseline_matches_torch(shape):
    """Validate the benchmark's composed 50-term power-series baseline against
    torch.special.gammaincc over the benchmark input domain (a, x in
    [0.1, 10.1]). On backends without a native torch.igammac kernel (e.g.
    ascend, where torch falls back to CPU and the AI Core baseline is composed
    from device-native primitives instead) the benchmark times the gems kernel
    against this reference; this test proves the reference itself is correct,
    so the benchmark comparison is meaningful."""
    torch.manual_seed(0)
    a = torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 10 + 0.1
    x = torch.rand(shape, dtype=torch.float32, device=flag_gems.device) * 10 + 0.1
    res = _igammac_composed(a, x)
    ref = torch.special.gammaincc(a.double().cpu(), x.double().cpu())
    err = (res.double().cpu() - ref).abs().max().item()
    assert err < 1e-4, f"composed baseline off by {err:.2e}"
    assert int(torch.isnan(res).sum()) == 0
    # domain corners on a small grid
    grid_v = torch.tensor(
        [0.1, 0.5, 5.0, 9.9, 10.0, 10.1],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ga, gx = torch.meshgrid(grid_v, grid_v, indexing="ij")
    ga = ga.reshape(-1)
    gx = gx.reshape(-1)
    res_g = _igammac_composed(ga, gx)
    ref_g = torch.special.gammaincc(ga.double().cpu(), gx.double().cpu())
    err_g = (res_g.double().cpu() - ref_g).abs().max().item()
    assert err_g < 1e-4, f"composed grid corner off by {err_g:.2e}"
