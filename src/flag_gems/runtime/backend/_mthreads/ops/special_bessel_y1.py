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

"""Triton implementation of aten::special_bessel_y1 (Bessel Y, order one).

Elementwise unary op.  The implementation avoids any libdevice ``y1`` call and
uses two explicit expansions of the DLMF 10.8.1 formula (0 < x < 8) plus the
large-argument asymptotic expansion (x >= 8):

    Y1(x) = (2/pi) * (ln(x/2) + gamma) * J1(x) - 2/(pi*x) - S(x)/pi
    J1(x) = (x/2) * A(w),  S(x) = (x/2) * B(w),  w = (x/2)^2
    A(w) = sum_k (-1)^k w^k / (k! (k+1)!)
    B(w) = sum_k (-1)^k (H_k + H_{k+1}) w^k / (k! (k+1)!)

which folds the Euler constant of the psi terms into a single logarithm:

    Y1(x) = ( (2*L*A - B) * (x/2) - 2/x ) / pi,   L = ln(x/2) + gamma

Both polynomial chains are evaluated with Horner and are independent, giving
two-way ILP.  Special values are applied elementwise after the numeric path.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_bessel_y1 import (
    special_bessel_y1 as default_special_bessel_y1,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64, so only fp32 is specialized;
# every other dtype falls back to the generic implementation.
_SUPPORTED_DTYPES = {torch.float32}

_BLOCK = 1024
_NUM_WARPS = 4

_INV_PI = tl.constexpr(0.3183098861837907)
_TWO_OVER_PI = tl.constexpr(0.6366197723675814)
_GAMMA_MINUS_LN2 = tl.constexpr(-0.11593151565841245)  # euler_gamma - ln(2)
_SPLIT = tl.constexpr(8.0)

_NEG_INF = tl.constexpr(float("-inf"))
_POS_INF = tl.constexpr(float("inf"))


@triton.jit
def _log_gamma(x):
    """ln(x) + (euler_gamma - ln 2), the log factor of the series branch."""
    # No cheaper log exists on this backend: libdevice.fast_logf and
    # fast_log2f are both unimplemented stubs on the MUSA build.
    return tl.log(x) + _GAMMA_MINUS_LN2


@triton.jit
def _y1_small(x):
    """Series branch, valid (and accurate) for 0 < x <= 8."""
    w = x * x * 0.25
    lx = _log_gamma(x)

    if x.type.element_ty == tl.float64:
        # 20 terms: truncation at x=8 is ~1e-14 absolute
        a = -3.37933e-36
        a = a * w + 1.28391e-33
        a = a * w - 4.39129e-31
        a = a * w + 1.3437e-28
        a = a * w - 3.65436e-26
        a = a * w + 8.7713e-24
        a = a * w - 1.84213e-21
        a = a * w + 3.3531e-19
        a = a * w - 5.229988896665e-17
        a = a * w + 6.903689469952366e-15
        a = a * w - 7.594058416947603e-13
        a = a * w + 6.834652585313960e-11
        a = a * w - 4.920949861426052e-9
        a = a * w + 2.7557319223985893e-7
        a = a * w - 1.1574074074074073e-5
        a = a * w + 3.4722222222222222e-4
        a = a * w - 6.9444444444444444e-3
        a = a * w + 8.3333333333333333e-2
        a = a * w - 0.5
        a = a * w + 1.0

        b = -2.414626e-35
        b = b * w + 9.042277e-33
        b = b * w - 3.045329e-30
        b = b * w + 9.164108e-28
        b = b * w - 2.448166e-25
        b = b * w + 5.762647e-23
        b = b * w - 1.184883e-20
        b = b * w + 2.106839e-19
        b = b * w - 3.202375e-16
        b = b * w + 4.106884e-14
        b = b * w - 4.372008e-12
        b = b * w + 3.791061e-10
        b = b * w - 2.613278e-8
        b = b * w + 1.389676e-6
        b = b * w - 5.478395061728395e-5
        b = b * w + 1.5162037037037037e-3
        b = b * w - 2.7199074074074074e-2
        b = b * w + 2.7777777777777778e-1
        b = b * w - 1.25
        b = b * w + 1.0
    else:
        # 14 terms: truncation error at x=8 is ~9e-6 absolute, 12x inside
        # atol=1e-4, and far smaller over the range randn actually reaches.
        a = -1.84213e-21
        a = a * w + 3.3531e-19
        a = a * w - 5.229988896665e-17
        a = a * w + 6.903689469952366e-15
        a = a * w - 7.594058416947603e-13
        a = a * w + 6.834652585313960e-11
        a = a * w - 4.920949861426052e-9
        a = a * w + 2.7557319223985893e-7
        a = a * w - 1.1574074074074073e-5
        a = a * w + 3.4722222222222222e-4
        a = a * w - 6.9444444444444444e-3
        a = a * w + 8.3333333333333333e-2
        a = a * w - 0.5
        a = a * w + 1.0

        b = -1.184883e-20
        b = b * w + 2.106839e-19
        b = b * w - 3.202375e-16
        b = b * w + 4.106884e-14
        b = b * w - 4.372008e-12
        b = b * w + 3.791061e-10
        b = b * w - 2.613278e-8
        b = b * w + 1.389676e-6
        b = b * w - 5.478395061728395e-5
        b = b * w + 1.5162037037037037e-3
        b = b * w - 2.7199074074074074e-2
        b = b * w + 2.7777777777777778e-1
        b = b * w - 1.25
        b = b * w + 1.0

    t = x * 0.5
    # 2/(pi*x): the approximate reciprocal is <=2 ulp, i.e. <=2.4e-7 relative,
    # inside the 1.3e-6 fp32 rtol that governs this term for small x.
    if x.type.element_ty == tl.float64:
        return ((2.0 * lx * a - b) * t - 2.0 / x) * _INV_PI
    return ((2.0 * lx * a - b) * t - tl.fdiv(2.0, x, ieee_rounding=False)) * _INV_PI


@triton.jit
def _y1_lo(x):
    """Short series arm for tiles whose maximum is below 4 (fp32 only).

    At the x=4 boundary the first omitted term is 1.3e-8 of A, so the total
    truncation error stays near 1e-6 absolute, ~100x inside atol=1e-4.
    """
    w = x * x * 0.25
    lx = _log_gamma(x)

    a = -7.594058416947603e-13
    a = a * w + 6.834652585313960e-11
    a = a * w - 4.920949861426052e-9
    a = a * w + 2.7557319223985893e-7
    a = a * w - 1.1574074074074073e-5
    a = a * w + 3.4722222222222222e-4
    a = a * w - 6.9444444444444444e-3
    a = a * w + 8.3333333333333333e-2
    a = a * w - 0.5
    a = a * w + 1.0

    b = -4.372008e-12
    b = b * w + 3.791061e-10
    b = b * w - 2.613278e-8
    b = b * w + 1.389676e-6
    b = b * w - 5.478395061728395e-5
    b = b * w + 1.5162037037037037e-3
    b = b * w - 2.7199074074074074e-2
    b = b * w + 2.7777777777777778e-1
    b = b * w - 1.25
    b = b * w + 1.0

    t = x * 0.5
    return ((2.0 * lx * a - b) * t - tl.fdiv(2.0, x, ieee_rounding=False)) * _INV_PI


@triton.jit
def _y1_large(x):
    """Asymptotic branch, valid for x >= 8."""
    inv = 1.0 / x
    inv2 = inv * inv
    # amp = sqrt(2/(pi*x))
    amp = tl.sqrt(_TWO_OVER_PI * inv)
    # P = sum_k (-1)^k a_{2k} / x^{2k}
    p = 1.0 + inv2 * (0.1171875 - inv2 * 0.144195556640625)
    # Q = sum_k (-1)^k a_{2k+1} / x^{2k+1}
    q = inv * (0.375 - inv2 * (0.1025390625 - inv2 * 0.2775764465332031))
    phi = x - 2.356194490192345  # x - 3*pi/4
    return amp * (tl.sin(phi) * p + tl.cos(phi) * q)


@triton.jit
def _y1_mixed(x):
    # NaN propagates naturally: negatives take the series arm (log of a negative
    # is NaN) and a NaN input takes the asymptotic arm (every term is NaN).
    y = tl.where(x < _SPLIT, _y1_small(x), _y1_large(x))
    # x == +inf
    return tl.where(x == _POS_INF, 0.0, y)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": b}, num_warps=w)
        for b, w in (
            (128, 1),  # 4 elements/thread, warp-local max only (no barrier)
            (256, 1),  # 8, no barrier
            (256, 2),  # 4
            (512, 1),  # 16, no barrier
            (512, 2),  # 8
            (512, 4),  # 4
            (1024, 8),  # 4
            (1024, 4),  # 8
            (1024, 2),  # 16
            (2048, 8),  # 8
            (2048, 4),  # 16
            (4096, 8),  # 16
            (4096, 4),  # 32
            (8192, 8),  # 32
        )
    ],
    key=["n"],
)
@triton.jit
def _bessel_y1_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.max_contiguous(
        tl.multiple_of(pid * BLOCK + tl.arange(0, BLOCK), BLOCK), BLOCK
    )
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy="evict_first")
    # randn-like tiles never reach 8, so the asymptotic branch is dead work:
    # skip it whenever the whole tile stays on the series branch.  A tighter
    # per-tile bound then selects a shorter series.  Negative and NaN inputs
    # already yield NaN from the series arm itself.
    xm = tl.max(x, axis=0)
    if x.type.element_ty == tl.float64:
        if xm < _SPLIT:
            y = _y1_small(x)
        else:
            y = _y1_mixed(x)
    else:
        if xm < 4.0:
            y = _y1_lo(x)
        else:
            if xm < _SPLIT:
                y = _y1_small(x)
            else:
                y = _y1_mixed(x)
    # x == 0 (including -0.0) maps to -inf in every arm
    tl.store(
        out_ptr + offs,
        tl.where(x == 0.0, _NEG_INF, y),
        mask=mask,
        eviction_policy="evict_first",
    )


def _use_triton_kernel(A):
    """Whether ``A`` can be handled by the MUSA elementwise kernel."""
    if not isinstance(A, torch.Tensor):
        return False
    if A.device.type != "musa" or A.dtype not in _SUPPORTED_DTYPES:
        return False
    if A.numel() == 0:
        return False
    return True


def special_bessel_y1(self):
    logger.debug("GEMS_MTHREADS SPECIAL_BESSEL_Y1")

    if not _use_triton_kernel(self):
        return default_special_bessel_y1(self)

    out = torch.empty_like(self)
    n = self.numel()
    with torch_device_fn.device(out.device):
        _bessel_y1_kernel[lambda meta: ((n + meta["BLOCK"] - 1) // meta["BLOCK"],)](
            self, out, n
        )
    return out
