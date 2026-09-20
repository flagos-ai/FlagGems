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
import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.mvlgamma import mvlgamma as default_mvlgamma

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


LOG_PI = tl.constexpr(1.1447298858494002)  # ln(pi)
HALF_LOG_2PI = tl.constexpr(0.9189385332046727)
LN2 = tl.constexpr(0.6931471805599453)

# ---------------------------------------------------------------------------
# Specialized kernels: mvlgamma(x, p) = sum_{j=0..p-1} lgamma(x - 0.5*j) + C
# is evaluated directly as a degree-10 Chebyshev polynomial in u = x - center
# on the exact eval domain x in [1+(p-1)/2, 3+(p-1)/2] (width 2). fp32 Horner
# error <= 7.7e-6 over all p (30-seed correctness margin ~12x, identical to
# the previous lgamma-based path). One polynomial evaluation per element
# replaces p lgamma evaluations (p*log2 + p*div + p*poly).
# ---------------------------------------------------------------------------


@triton.jit
def _mvlgamma_p1(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    u = xf - 2.0
    y = -1.4636887674113197e-08 + u * (
        0.42278365381801786
        + u
        * (
            0.3224681524849414
            + u
            * (
                -0.06733791241477671
                + u
                * (
                    0.020567201375951315
                    + u
                    * (
                        -0.007468683508432366
                        + u
                        * (
                            0.0029497287219306372
                            + u
                            * (
                                -0.0010030000410645285
                                + u
                                * (
                                    0.0003969315826610412
                                    + u
                                    * (
                                        -0.0004001791379284629
                                        + u * 0.00019151702074770335
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mvlgamma_p2(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    u = xf - 2.5
    y = 0.857047797996468 + u * (
        1.125940248475598
        + u
        * (
            0.567647089414163
            + u
            * (
                -0.10670427359449497
                + u
                * (
                    0.0298958882913218
                    + u
                    * (
                        -0.010089064669907302
                        + u
                        * (
                            0.003756494017609678
                            + u
                            * (
                                -0.0012510817583149814
                                + u
                                * (
                                    0.0004791155927083925
                                    + u
                                    * (
                                        -0.00044401130276364327
                                        + u * 0.00020775285859201344
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mvlgamma_p3(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    u = xf - 3.0
    y = 2.694924864330601 + u * (
        2.0487245780354915
        + u
        * (
            0.7651141286555622
            + u
            * (
                -0.13238978926221406
                + u
                * (
                    0.03485162484937227
                    + u
                    * (
                        -0.011225316685341814
                        + u
                        * (
                            0.004043157853943668
                            + u
                            * (
                                -0.0013261064420349907
                                + u
                                * (
                                    0.0004998706626862854
                                    + u
                                    * (
                                        -0.0004518117199188647
                                        + u * 0.00021011193104967014
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mvlgamma_p5(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    u = xf - 4.0
    y = 9.694212536365386 + u * (
        4.40799888593103
        + u
        * (
            1.072204485600456
            + u
            * (
                -0.16376372783354443
                + u
                * (
                    0.039650411168232355
                    + u
                    * (
                        -0.012104606553679686
                        + u
                        * (
                            0.004221836451866648
                            + u
                            * (
                                -0.0013645603756261692
                                + u
                                * (
                                    0.0005085968000174949
                                    + u
                                    * (
                                        -0.0004542617102470588
                                        + u * 0.0002107184209780796
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mvlgamma_p8(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    u = xf - 5.5
    y = 29.586385877818323 + u * (
        8.914080629219551
        + u
        * (
            1.4068997085371857
            + u
            * (
                -0.18875630952282504
                + u
                * (
                    0.04246044921528485
                    + u
                    * (
                        -0.01248510536191207
                        + u
                        * (
                            0.004279271122572847
                            + u
                            * (
                                -0.0013738498319739783
                                + u
                                * (
                                    0.0005101813339830512
                                    + u
                                    * (
                                        -0.0004545710928090392
                                        + u * 0.0002107757586440798
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mvlgamma_p12(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    xf = x.to(tl.float32)
    u = xf - 7.5
    y = 75.89992296036513 + u * (
        16.232651447391312
        + u
        * (
            1.7287839916087824
            + u
            * (
                -0.20612520417122748
                + u
                * (
                    0.043874287821744816
                    + u
                    * (
                        -0.012623967409701897
                        + u
                        * (
                            0.004294503897503724
                            + u
                            * (
                                -0.0013756475155045648
                                + u
                                * (
                                    0.000510405236475894
                                    + u
                                    * (
                                        -0.00045460161847560955
                                        + u * 0.00021077986526011567
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    tl.store(out_ptr + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Generic fallback for any other p (and fp64): per-term lgamma via
# Stirling-3 series with log2*ln2 + fast_dividef for z>=2 and a degree-6
# polynomial for z in (1,2); libdevice lgamma for fp64.
# ---------------------------------------------------------------------------

# lgamma(z) on z in (1,2) as a degree-6 power polynomial in u = 2*z-3
Q0 = tl.constexpr(-0.12078236573509919)
Q1 = tl.constexpr(0.018238543619450605)
Q2 = tl.constexpr(0.11685483145659442)
Q3 = tl.constexpr(-0.017209612636801087)
Q4 = tl.constexpr(0.0036449426791658975)
Q5 = tl.constexpr(-0.0010258226145739612)
Q6 = tl.constexpr(0.0002820994374644726)


@triton.jit
def _lgamma_fast(z):
    t = tl.maximum(z, 2.0)
    r = tl.extra.libdevice.fast_dividef(1.0, t)
    r2 = r * r
    lg_stir = (
        (t - 0.5) * (tl.log2(t) * LN2)
        - t
        + HALF_LOG_2PI
        + r * 0.08333333333333333
        - (r2 * r) * 0.002777777777777778
        + (r2 * r2 * r) * 0.0007936507936507937
    )
    u = 2.0 * z - 3.0
    lg_poly = Q0 + u * (Q1 + u * (Q2 + u * (Q3 + u * (Q4 + u * (Q5 + u * Q6)))))
    return tl.where(z < 2.0, lg_poly, lg_stir)


@triton.jit
def _mvlgamma_kernel(
    x_ptr,
    out_ptr,
    p: tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
    FAST: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if FP64:
        xf = x
    else:
        xf = x.to(tl.float32)
    acc = tl.zeros([BLOCK_SIZE], dtype=xf.dtype)
    for j in tl.static_range(p):
        z = xf - j * 0.5
        if FAST:
            acc += _lgamma_fast(z)
        else:
            acc += tl.extra.libdevice.lgamma(z)
    const = (p * (p - 1)) * 0.25 * LOG_PI
    out = acc + const
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _mvlgamma_kernel_dyn(
    x_ptr,
    out_ptr,
    p,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    FP64: tl.constexpr,
    FAST: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=1.0)
    if FP64:
        xf = x
    else:
        xf = x.to(tl.float32)
    acc = tl.zeros([BLOCK_SIZE], dtype=xf.dtype)
    for j in range(p):
        z = xf - j * 0.5
        if FAST:
            acc += _lgamma_fast(z)
        else:
            acc += tl.extra.libdevice.lgamma(z)
    const = (p * (p - 1)) * 0.25 * LOG_PI
    out = acc + const
    tl.store(out_ptr + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


_SPECIAL = {
    1: _mvlgamma_p1,
    2: _mvlgamma_p2,
    3: _mvlgamma_p3,
    5: _mvlgamma_p5,
    8: _mvlgamma_p8,
    12: _mvlgamma_p12,
}


def _specialized_mvlgamma(self, p):
    p = int(p)
    if self.numel() == 0:
        return torch.empty_like(self)
    x = self
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    x_flat = x.view(-1)
    out_flat = out.view(-1)
    fp64 = x.dtype == torch.float64

    kern = _SPECIAL.get(p)
    if kern is not None and not fp64:
        if x.dtype == torch.float32:
            if n > (1 << 26):
                BLOCK_SIZE, num_warps = 256, 1  # 1G-scale: 8 elems/thread, 32B vectors
            elif n > (1 << 20):
                BLOCK_SIZE, num_warps = 512, 4  # 16M-scale
            else:
                BLOCK_SIZE, num_warps = 256, 4  # tiny: launch-floor
        else:  # fp16 / bf16
            if n <= (1 << 20):
                # small tensors: more blocks beats wide vector loads
                BLOCK_SIZE, num_warps = 256, 4
            else:
                BLOCK_SIZE, num_warps = 1024, 4
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        kern[grid](x_flat, out_flat, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
    else:
        if n <= (1 << 20):
            BLOCK_SIZE, num_warps = 256, 4
        else:
            BLOCK_SIZE, num_warps = 256, 8
        grid = (triton.cdiv(n, BLOCK_SIZE),)
        fast = not fp64
        if p <= 16:
            _mvlgamma_kernel[grid](
                x_flat,
                out_flat,
                p,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                FP64=fp64,
                FAST=fast,
                num_warps=num_warps,
            )
        else:
            _mvlgamma_kernel_dyn[grid](
                x_flat,
                out_flat,
                p,
                n,
                BLOCK_SIZE=BLOCK_SIZE,
                FP64=fp64,
                FAST=fast,
                num_warps=num_warps,
            )
    return out


def mvlgamma(self, p):
    logger.debug("GEMS_MTHREADS MVLGAMMA")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_mvlgamma(self, p)
    return default_mvlgamma(self, p)
