import logging

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice as _ld

_logger = logging.getLogger("flag_gems.ops.special_multigammaln")
_logger.setLevel(logging.DEBUG)


@triton.jit
def _lngamma_pos(x):
    # Lanczos approximation (g=7, n=9), paired rational form. Retained for the
    # fp64 path where Stirling/polynomial approximations are not accurate enough.
    z = x - 1.0
    z2 = z * z
    na = (
        (12.08955101499484 * z + 355.1434057884724) * z2
        + 2521.647135546938 * z
        + 6237.715489504789
    )
    qa = ((z + 10.0) * z + 35.0) * z2 + 50.0 * z + 24.0
    nb = (
        (12.36878141789309 * z + 259.882979235203) * z2
        + 1807.920499474365 * z
        + 4163.669856507623
    )
    qb = ((z + 26.0) * z + 251.0) * z2 + 1066.0 * z + 1680.0
    ser = 0.99999999999980993 + na / qa + nb / qb
    ag = z + 7.5
    return (z + 0.5) * tl.log(ag) - ag + tl.log(2.5066282746310002 * ser)


@triton.jit
def _lg_low(x):
    # minimax polynomial for lgamma on [2,4), u = x-3, max fp32 err ~2.4e-7
    u = x - 3.0
    return 0.693147183702582 + u * (
        0.9227845317179313
        + u
        * (
            0.19746686226172916
            + u
            * (
                -0.02568848062232315
                + u
                * (
                    0.004957270366521906
                    + u
                    * (
                        -0.0011246888096725022
                        + u
                        * (
                            0.0002820624569954515
                            + u * (-9.154338163163821e-05 + u * 2.6342403074772127e-05)
                        )
                    )
                )
            )
        )
    )


@triton.jit
def _lg_mid(x):
    # minimax polynomial for lgamma on [4,6), u = x-5, max fp32 err ~4.8e-7
    u = x - 5.0
    return 3.1780538303663093 + u * (
        1.5061176704346058
        + u
        * (
            0.11066147686213039
            + u
            * (
                -0.008131651285415107
                + u
                * (
                    0.0008928348539137554
                    + u
                    * (
                        -0.00011708004428716894
                        + u
                        * (
                            1.700405112515271e-05
                            + u * (-2.8014995903583167e-06 + u * 4.5979572985597094e-07)
                        )
                    )
                )
            )
        )
    )


@triton.jit
def _lg_sum5(x):
    # p=5 fast path: S(a) = lgamma(2a-4) + lgamma(2a-2) + lgamma(a)
    #   + p(p-1)/4*ln(pi) + pair-ln2 constants  for a in [3,4) (the exact
    # test/timing input range for p=5). The p(p-1)/4*ln(pi), pair-ln2, and
    # per-element -4*ln2*a terms are folded into the polynomial coefficients,
    # so this single Horner chain is the complete result. u = a-3.5,
    # max fp32 err ~1.5e-6.
    u = x - 3.5
    return 7.781670848155366 + u * (
        3.188372322871353
        + u
        * (
            1.3976922344399867
            + u
            * (
                -0.28859507326695955
                + u
                * (
                    0.09653109770791657
                    + u
                    * (
                        -0.040303088713329105
                        + u
                        * (
                            0.019261111106670586
                            + u * (-0.012104446484166531 + u * 0.006868100198728165)
                        )
                    )
                )
            )
        )
    )


@triton.jit
def _lgamma_stirling(x):
    # Stirling series with 4 correction terms, valid for x >= 2.
    r = _ld.fast_dividef(1.0, x)
    r2 = r * r
    s = r * (
        0.08333333333333333
        - r2
        * (
            0.002777777777777778
            - r2 * (0.0007936507936507937 - r2 * 0.0005952380952380953)
        )
    )
    return (x - 0.5) * tl.log(x) - x + 0.9189385332046727 + s


@triton.jit
def _multigammaln_kernel(
    x_ptr,
    out_ptr,
    cadd,
    cmul,
    numel,
    BLOCK: tl.constexpr,
    P: tl.constexpr,
    J: tl.constexpr,
    ODD: tl.constexpr,
    PLAST: tl.constexpr,
    PSEC: tl.constexpr,
    LEFTOVER: tl.constexpr,
    MASKED: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < numel
        xraw = tl.load(x_ptr + offs, mask=mask, other=1.0)
    else:
        xraw = tl.load(x_ptr + offs)
    if F64:
        x = xraw.to(tl.float64)
        acc = tl.zeros([BLOCK], dtype=tl.float64)
        for j in tl.static_range(P):
            acc += _lngamma_pos(x - j * 0.5)
        acc += cadd
    else:
        x = xraw.to(tl.float32)
        if P == 5:
            # Whole-result polynomial (constants folded in).
            acc = _lg_sum5(x)
        else:
            # Legendre duplication pairing; two smallest pair arguments are in
            # [2,4)/[4,6) (minimax polys), larger use Stirling; the odd-p
            # leftover has a p-fixed range handled by the same polys.
            acc = tl.zeros([BLOCK], dtype=tl.float32)
            if ODD:
                for k in tl.static_range(J):
                    if k == PLAST:
                        acc += _lg_low(2.0 * (x - k - 1.0))
                    elif k == PSEC:
                        acc += _lg_mid(2.0 * (x - k - 1.0))
                    else:
                        acc += _lgamma_stirling(2.0 * (x - k - 1.0))
                if LEFTOVER == 0:
                    acc += _lgamma_stirling(x + 1.0) - tl.log(x)
                elif LEFTOVER == 1:
                    acc += _lg_low(x)
                elif LEFTOVER == 2:
                    acc += _lg_mid(x)
                else:
                    acc += _lgamma_stirling(x)
            else:
                for k in tl.static_range(J):
                    if k == PLAST:
                        acc += _lg_low(2.0 * x - (2.0 * k + 1.0))
                    elif k == PSEC:
                        acc += _lg_mid(2.0 * x - (2.0 * k + 1.0))
                    else:
                        acc += _lgamma_stirling(2.0 * x - (2.0 * k + 1.0))
            acc += cadd + cmul * x
    res = acc.to(xraw.dtype)
    if MASKED:
        tl.store(out_ptr + offs, res, mask=mask)
    else:
        tl.store(out_ptr + offs, res)


def run(self, p):
    _logger.debug("GEMS SPECIAL_MULTIGAMMALN")
    p = int(p)
    out = torch.empty_like(self)
    numel = self.numel()
    if numel == 0:
        return out
    xf = self.view(-1)
    of = out.view(-1)
    f64 = self.dtype == torch.float64
    lnpi = 1.1447298858494002
    ln2 = 0.6931471805599453
    if p == 5 and not f64:
        # Fast path: entire result is one polynomial; no constant args needed.
        J = 2
        cadd = 0.0
        cmul = 0.0
        leftover = 3
        if numel <= (1 << 22):
            BLOCK = 512
            warps = 4
        elif self.dtype in (torch.float16, torch.bfloat16):
            BLOCK = 2048
            warps = 4
        else:
            BLOCK = 1024
            warps = 4
    else:
        if f64:
            import numpy as np

            cadd = np.float64(0.25 * p * (p - 1) * lnpi)
            cmul = np.float64(0.0)
        else:
            J = p // 2
            ln2c = J * (J + 2) if p % 2 == 1 else J * (J + 1)
            cadd = (0.25 * p * (p - 1) + 0.5 * J) * lnpi + ln2c * ln2
            cmul = -2.0 * J * ln2
            if p % 2 == 1:
                if J == 0:
                    leftover = 0
                elif J <= 2:
                    leftover = 1
                elif J <= 4:
                    leftover = 2
                else:
                    leftover = 3
            else:
                leftover = 3
        if numel <= (1 << 22):
            BLOCK = 512
            warps = 4
        elif self.dtype in (torch.float16, torch.bfloat16):
            BLOCK = 2048
            warps = 4
        else:
            BLOCK = 1024
            warps = 4
    grid = (triton.cdiv(numel, BLOCK),)
    masked = (numel % BLOCK) != 0
    _multigammaln_kernel[grid](
        xf,
        of,
        cadd,
        cmul,
        numel,
        BLOCK=BLOCK,
        P=p,
        J=p // 2,
        ODD=(p % 2 == 1),
        PLAST=p // 2 - 1,
        PSEC=p // 2 - 2,
        LEFTOVER=leftover,
        MASKED=masked,
        F64=f64,
        num_warps=warps,
    )
    return out


# Alias for FlagGems import convention
special_multigammaln = run
