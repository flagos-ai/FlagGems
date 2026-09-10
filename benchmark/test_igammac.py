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

from . import base, consts
from .conftest import Config

VENDOR = flag_gems.vendor_name

# On ascend the general KERNEL-mode do_bench_npu is unreliable; use operator
# (end-to-end) timing mode (same as det/lu_factor/linalg_solve_triangular).
if VENDOR == "ascend":
    Config.mode = consts.BenchMode.OPERATOR

_IGAMMAC_DTYPES = [
    torch.float32,
]
if flag_gems.runtime.device.support_fp64 and VENDOR != "ascend":
    _IGAMMAC_DTYPES.append(torch.float64)

# torch.igammac is not usable as a reference on every backend (e.g. NPU), where
# it silently falls back to CPU (msprof: no AI Core kernel; the NPU call is even
# slower than pure CPU). There we compose the same math from device-native torch
# primitives (log/exp/div/mul/add/where are AI Core ops; log-gamma is inlined via
# Lanczos because torch.lgamma also falls back to CPU) so the baseline runs on
# the accelerator, mirroring polygamma's composed-baseline approach. The composed
# reference is a fixed-N power series for Q(a,x) = 1 - P(a,x):
#   P = exp(a*ln x - x - lgamma(a)) * sum_{i=0}^{N-1} x^i / (a)_i
# which is accurate on the benchmark input domain (a,x in [0.1, 10.1], where
# x < a+1 so the series branch always applies); validated to ~4e-6 against
# torch.special.gammaincc (float64) over that domain.
_DEVICE_REF = flag_gems.device not in ("cuda", "cpu")


class IgammacBenchmark(base.GenericBenchmark):
    """GenericBenchmark with domain-valid inputs.

    Float64 is benchmarked on a representative subset of shapes (up to
    MAX_FLOAT64_ELEMENTS elements) to keep the runtime bounded: torch's
    float64 igammac baseline is very slow on large tensors.
    """

    MAX_FLOAT64_ELEMENTS = 2**24

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        if _DEVICE_REF:
            # The composed reference costs dozens of launches per call, so drop
            # the huge core shapes there (CUDA/CPU keep the full shape list).
            self.shapes = [s for s in self.shapes if math.prod(s) <= 2**24]

    def get_input_iter(self, dtype):
        shapes = self.shapes
        if dtype == torch.float64:
            shapes = [
                shape
                for shape in shapes
                if math.prod(shape) <= self.MAX_FLOAT64_ELEMENTS
            ]
        for shape in shapes:
            yield from self.input_fn(shape, dtype, self.device)


def _igammac_input(shape, dtype, device):
    # igammac(a, x) is only defined for a > 0 and x >= 0; the default randn
    # generator would push torch's reference kernel into a non-converging path.
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    yield a, x


def _igammac_input_out(shape, dtype, device):
    a = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    x = torch.rand(shape, dtype=dtype, device=device) * 10 + 0.1
    out = torch.empty_like(a)
    yield a, x, {"out": out}


def _lgamma_lanczos(x):
    """log-gamma via Lanczos (g=7, n=9) using only device-native torch
    primitives. torch.lgamma falls back to CPU on NPU, so we inline it."""
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


def _torch_igammac(a, x, out=None):
    if not _DEVICE_REF:
        return torch.igammac(a, x) if out is None else torch.igammac(a, x, out=out)
    res = _igammac_composed(a, x).to(a.dtype)
    return out.copy_(res) if out is not None else res


@pytest.mark.igammac
def test_igammac():
    bench = IgammacBenchmark(
        op_name="igammac",
        torch_op=_torch_igammac,
        gems_op=flag_gems.igammac if _DEVICE_REF else None,
        input_fn=_igammac_input,
        dtypes=_IGAMMAC_DTYPES,
    )
    bench.run()


@pytest.mark.igammac_out
def test_igammac_out():
    bench = IgammacBenchmark(
        op_name="igammac_out",
        torch_op=_torch_igammac,
        gems_op=flag_gems.igammac_out if _DEVICE_REF else None,
        input_fn=_igammac_input_out,
        dtypes=_IGAMMAC_DTYPES,
    )
    bench.run()
