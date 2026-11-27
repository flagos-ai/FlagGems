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

"""Fused moving-average observer + fake-quant helper (QAT), Triton/MUSA.

Semantics reproduced bit-for-bit from the target's ATen reference
(established by focused numerical probes on the target):

  * observer: running_min/max <- fma(avg, chan_min - running_min, running_min)
    (float32 fma; the reference's separate mul+add differs in the last bit)
  * scale (asymmetric): (max(running_max,0) - min(running_min,0)) / (qmax-qmin)
    computed with the *exact* difference (double-float TwoSum residual) and a
    double-float reciprocal, because ATen's value is the correctly rounded
    quotient and libdevice div_rn on this target is 1 ULP off for ~10% of the
    ranges.  zero span -> 0.1.
  * scale (symmetric, only when running_min<0<running_max):
    max(-min_neg/(zp-qmin), max_pos/(qmax-zp)), zp = rint((qmin+qmax)/2)
  * zero_point = clamp(qmin - rint(min_neg/scale), qmin, qmax); midpoint for the
    symmetric branch
  * fake quant: q = rint(x * (1/scale)) + zp; mask = q in [qmin, qmax];
    out = (clamp(q) - zp) * scale
  * qparams are written only when fake_quant_on is set; running stats only when
    observer_on is set; with fake_quant_on unset the output is a copy of self and
    the mask is all-true.

The work is a two pass stream: one exact min/max reduction per channel group and
one read+write quantize pass.  observer_on / fake_quant_on are read *inside* the
kernels so a call never synchronizes the host, and every kernel early-exits when
its flag is clear.
"""

import logging
import struct

import torch
import triton
import triton.language as tl

from flag_gems.ops._fused_moving_avg_obs_fq_helper import (
    _fused_moving_avg_obs_fq_helper as default_fused_moving_avg_obs_fq_helper,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import tl_extra_shim

libdevice = tl_extra_shim

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_SCALAR_TYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
}

_BLOCK_R = 4096
_BLOCK_A = 4096
_BLOCK_F = 2048
_BLOCK_C = 256
_TARGET_PROGS = 480


@triton.jit
def _div_accurate(a, b):
    # a / b with a residual correction (libdevice div_rn is ~1 ULP off here)
    inv = libdevice.div_rn(1.0, b)
    r = libdevice.fma(-inv, b, 1.0)
    q = libdevice.fma(a, inv * r, a * inv)
    rem = libdevice.fma(-q, b, a)
    return libdevice.fma(rem, inv, q)


@triton.jit
def _qparam(
    mn, mx, qmnf, qmxf, den, chi, clo, chi_n, clo_n, chi_p, clo_p, SYM: tl.constexpr
):
    # exact difference mx - mn as (s_hi, err) via TwoSum
    s_hi = mx - mn
    b = -mn
    bv = s_hi - mx
    err = (mx - (s_hi - bv)) + (b - bv)
    # (s_hi + err) / den with a double-float reciprocal and one residual step
    q0 = libdevice.fma(s_hi, chi, s_hi * clo)
    rem = libdevice.fma(-q0, den, s_hi) + err
    sc = libdevice.fma(rem, chi, q0)
    sc = tl.where(s_hi == 0.0, 0.1, sc)
    if SYM:
        both = (mn < 0.0) & (mx > 0.0)
        sc_sym = tl.maximum(
            libdevice.fma(-mn, chi_n, -mn * clo_n), libdevice.fma(mx, chi_p, mx * clo_p)
        )
        sc = tl.where(both, sc_sym, sc)
        zmid = libdevice.rint(libdevice.div_rn(qmnf + qmxf, 2.0))
        z = tl.where(both, zmid, qmnf - libdevice.rint(_div_accurate(mn, sc)))
    else:
        z = qmnf - libdevice.rint(_div_accurate(mn, sc))
    z = tl.minimum(tl.maximum(z, qmnf), qmxf)
    return sc, z


@triton.jit
def _reduce_kernel(
    x_ptr,
    cmin_ptr,
    cmax_ptr,
    obs_ptr,
    C,
    I,
    S,
    NF,
    CT,
    BLOCK: tl.constexpr,
    TAIL: tl.constexpr,
    FLAG_PTR: tl.constexpr,
    OBS: tl.constexpr,
):
    if FLAG_PTR:
        active = tl.load(obs_ptr).to(tl.int32) != 0
    else:
        active = OBS != 0
    if active:
        c = tl.program_id(0)
        o = tl.program_id(1)
        s = tl.program_id(2)
        base = (o * C + c) * I
        acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
        acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
        # Each program owns a contiguous run of whole blocks, so every bulk load
        # is unmasked (and vectorizable); the single ragged block is picked up by
        # the last program only.  The old form was fully predicated whenever
        # S * CT != tiles, which is the case for every large non-per-channel
        # tensor.
        start = s * CT
        end = tl.minimum(start + CT, NF)
        for k in range(start, end):
            offs = k * BLOCK + tl.arange(0, BLOCK)
            v = tl.load(x_ptr + base + offs)
            acc_min = tl.minimum(acc_min, v)
            acc_max = tl.maximum(acc_max, v)
        if TAIL:
            if s == S - 1:
                offs = NF * BLOCK + tl.arange(0, BLOCK)
                msk = offs < I
                v = tl.load(x_ptr + base + offs, mask=msk, other=0.0)
                acc_min = tl.minimum(acc_min, tl.where(msk, v, float("inf")))
                acc_max = tl.maximum(acc_max, tl.where(msk, v, float("-inf")))
        idx = (c * tl.num_programs(1) + o) * S + s
        tl.store(cmin_ptr + idx, tl.min(acc_min, axis=0))
        tl.store(cmax_ptr + idx, tl.max(acc_max, axis=0))


@triton.jit
def _fused_chan_kernel(
    x_ptr,
    out_ptr,
    mask_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    obs_ptr,
    fq_ptr,
    C,
    I,
    avg,
    qmin,
    qmax,
    chi,
    clo,
    chi_n,
    clo_n,
    chi_p,
    clo_p,
    FLAG_PTR: tl.constexpr,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    RBLK: tl.constexpr,
    ABLK: tl.constexpr,
    EVEN_R: tl.constexpr,
    EVEN_A: tl.constexpr,
):
    # One program owns a whole channel, so the min/max reduction, the moving
    # average, the qparams and the quantize pass all happen in one launch and the
    # quantize pass re-reads data the same program just touched (cache reuse).
    # A channel is only ever read and written by this program, so the in-place
    # state updates cannot race with another program.
    if FLAG_PTR:
        do_obs = tl.load(obs_ptr).to(tl.int32) != 0
        do_fq = tl.load(fq_ptr).to(tl.int32) != 0
    else:
        do_obs = OBS != 0
        do_fq = FQ != 0
    c = tl.program_id(0)
    base = c * I
    rmin = tl.load(rmin_ptr + c).to(tl.float32)
    rmax = tl.load(rmax_ptr + c).to(tl.float32)
    if do_obs:
        acc_min = tl.full((RBLK,), float("inf"), tl.float32)
        acc_max = tl.full((RBLK,), float("-inf"), tl.float32)
        ntr = tl.cdiv(I, RBLK)
        for k in range(ntr):
            offs = k * RBLK + tl.arange(0, RBLK)
            if EVEN_R:
                v = tl.load(x_ptr + base + offs, eviction_policy="evict_last")
                acc_min = tl.minimum(acc_min, v)
                acc_max = tl.maximum(acc_max, v)
            else:
                msk = offs < I
                v = tl.load(
                    x_ptr + base + offs,
                    mask=msk,
                    other=0.0,
                    eviction_policy="evict_last",
                )
                acc_min = tl.minimum(acc_min, tl.where(msk, v, float("inf")))
                acc_max = tl.maximum(acc_max, tl.where(msk, v, float("-inf")))
        rmin = libdevice.fma(avg, tl.min(acc_min, axis=0) - rmin, rmin)
        rmax = libdevice.fma(avg, tl.max(acc_max, axis=0) - rmax, rmax)
        tl.store(rmin_ptr + c, rmin)
        tl.store(rmax_ptr + c, rmax)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    sc, z = _qparam(
        tl.minimum(rmin, 0.0),
        tl.maximum(rmax, 0.0),
        qmnf,
        qmxf,
        qmxf - qmnf,
        chi,
        clo,
        chi_n,
        clo_n,
        chi_p,
        clo_p,
        SYM,
    )
    nta = tl.cdiv(I, ABLK)
    if do_fq:
        inv = libdevice.div_rn(1.0, sc)
        zi = z.to(tl.int32)
        zf = zi.to(tl.float32)
        tl.store(scale_ptr + c, sc.to(scale_ptr.dtype.element_ty))
        tl.store(zp_ptr + c, zi.to(zp_ptr.dtype.element_ty))
        for k in range(nta):
            # walk backwards: the tiles read last in the reduce phase are the
            # ones most likely still resident
            offs = (nta - 1 - k) * ABLK + tl.arange(0, ABLK)
            if EVEN_A:
                x = tl.load(x_ptr + base + offs, eviction_policy="evict_first")
                q = libdevice.rint(x * inv).to(tl.int32) + zi
                valid = (q >= qmin) & (q <= qmax)
                qc = tl.minimum(tl.maximum(q, qmin), qmax)
                tl.store(
                    out_ptr + base + offs,
                    (qc.to(tl.float32) - zf) * sc,
                    eviction_policy="evict_first",
                )
                tl.store(
                    mask_ptr + base + offs,
                    valid.to(tl.uint8),
                    eviction_policy="evict_first",
                )
            else:
                msk = offs < I
                x = tl.load(
                    x_ptr + base + offs,
                    mask=msk,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                q = libdevice.rint(x * inv).to(tl.int32) + zi
                valid = (q >= qmin) & (q <= qmax)
                qc = tl.minimum(tl.maximum(q, qmin), qmax)
                tl.store(
                    out_ptr + base + offs,
                    (qc.to(tl.float32) - zf) * sc,
                    mask=msk,
                    eviction_policy="evict_first",
                )
                tl.store(
                    mask_ptr + base + offs,
                    valid.to(tl.uint8),
                    mask=msk,
                    eviction_policy="evict_first",
                )
    else:
        for k in range(nta):
            offs = (nta - 1 - k) * ABLK + tl.arange(0, ABLK)
            if EVEN_A:
                x = tl.load(x_ptr + base + offs, eviction_policy="evict_first")
                tl.store(out_ptr + base + offs, x, eviction_policy="evict_first")
                tl.store(
                    mask_ptr + base + offs,
                    tl.full((ABLK,), 1, tl.uint8),
                    eviction_policy="evict_first",
                )
            else:
                msk = offs < I
                x = tl.load(
                    x_ptr + base + offs,
                    mask=msk,
                    other=0.0,
                    eviction_policy="evict_first",
                )
                tl.store(
                    out_ptr + base + offs, x, mask=msk, eviction_policy="evict_first"
                )
                tl.store(
                    mask_ptr + base + offs,
                    tl.full((ABLK,), 1, tl.uint8),
                    mask=msk,
                    eviction_policy="evict_first",
                )


@triton.jit
def _fin_scalar_kernel(
    cmin_ptr,
    cmax_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    obs_ptr,
    fq_ptr,
    S,
    avg,
    qmin,
    qmax,
    chi,
    clo,
    chi_n,
    clo_n,
    chi_p,
    clo_p,
    FLAG_PTR: tl.constexpr,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if FLAG_PTR:
        do_obs = tl.load(obs_ptr).to(tl.int32) != 0
        do_fq = tl.load(fq_ptr).to(tl.int32) != 0
    else:
        do_obs = OBS != 0
        do_fq = FQ != 0
    offs = tl.arange(0, BLOCK)
    rmin = tl.load(rmin_ptr).to(tl.float32)
    rmax = tl.load(rmax_ptr).to(tl.float32)
    if do_obs:
        m = offs < S
        cmin = tl.min(tl.load(cmin_ptr + offs, mask=m, other=float("inf")), axis=0)
        cmax = tl.max(tl.load(cmax_ptr + offs, mask=m, other=float("-inf")), axis=0)
        rmin = libdevice.fma(avg, cmin - rmin, rmin)
        rmax = libdevice.fma(avg, cmax - rmax, rmax)
        tl.store(rmin_ptr, rmin)
        tl.store(rmax_ptr, rmax)
    if do_fq:
        qmnf = qmin.to(tl.float32)
        qmxf = qmax.to(tl.float32)
        sc, z = _qparam(
            tl.minimum(rmin, 0.0),
            tl.maximum(rmax, 0.0),
            qmnf,
            qmxf,
            qmxf - qmnf,
            chi,
            clo,
            chi_n,
            clo_n,
            chi_p,
            clo_p,
            SYM,
        )
        tl.store(scale_ptr, sc.to(scale_ptr.dtype.element_ty))
        tl.store(zp_ptr, z.to(tl.int32).to(zp_ptr.dtype.element_ty))


@triton.jit
def _fin_vec_kernel(
    cmin_ptr,
    cmax_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    obs_ptr,
    fq_ptr,
    C,
    NP,
    avg,
    qmin,
    qmax,
    chi,
    clo,
    chi_n,
    clo_n,
    chi_p,
    clo_p,
    FLAG_PTR: tl.constexpr,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if FLAG_PTR:
        do_obs = tl.load(obs_ptr).to(tl.int32) != 0
        do_fq = tl.load(fq_ptr).to(tl.int32) != 0
    else:
        do_obs = OBS != 0
        do_fq = FQ != 0
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < C
    rmin = tl.load(rmin_ptr + idx, mask=m, other=0.0).to(tl.float32)
    rmax = tl.load(rmax_ptr + idx, mask=m, other=0.0).to(tl.float32)
    if do_obs:
        pmin = tl.full((BLOCK,), float("inf"), tl.float32)
        pmax = tl.full((BLOCK,), float("-inf"), tl.float32)
        for k in range(NP):
            pmin = tl.minimum(
                pmin, tl.load(cmin_ptr + idx * NP + k, mask=m, other=float("inf"))
            )
            pmax = tl.maximum(
                pmax, tl.load(cmax_ptr + idx * NP + k, mask=m, other=float("-inf"))
            )
        rmin = libdevice.fma(avg, pmin - rmin, rmin)
        rmax = libdevice.fma(avg, pmax - rmax, rmax)
        tl.store(rmin_ptr + idx, rmin, mask=m)
        tl.store(rmax_ptr + idx, rmax, mask=m)
    if do_fq:
        qmnf = qmin.to(tl.float32)
        qmxf = qmax.to(tl.float32)
        sc, z = _qparam(
            tl.minimum(rmin, 0.0),
            tl.maximum(rmax, 0.0),
            qmnf,
            qmxf,
            qmxf - qmnf,
            chi,
            clo,
            chi_n,
            clo_n,
            chi_p,
            clo_p,
            SYM,
        )
        tl.store(scale_ptr + idx, sc.to(scale_ptr.dtype.element_ty), mask=m)
        tl.store(zp_ptr + idx, z.to(tl.int32).to(zp_ptr.dtype.element_ty), mask=m)


@triton.jit
def _apply_kernel(
    x_ptr,
    out_ptr,
    mask_ptr,
    scale_ptr,
    zp_ptr,
    fq_ptr,
    C,
    I,
    qmin,
    qmax,
    PC: tl.constexpr,
    FLAG_PTR: tl.constexpr,
    FQ: tl.constexpr,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    if FLAG_PTR:
        do_fq = tl.load(fq_ptr).to(tl.int32) != 0
    else:
        do_fq = FQ != 0
    tile = tl.program_id(0)
    co = tl.program_id(1)
    if PC:
        c = co % C
        o = co // C
    else:
        c = 0
        o = 0
    base = (o * C + c) * I
    if PC:
        sin = tl.load(scale_ptr + c).to(tl.float32)
        zi = tl.load(zp_ptr + c).to(tl.int32)
    else:
        sin = tl.load(scale_ptr).to(tl.float32)
        zi = tl.load(zp_ptr).to(tl.int32)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    if do_fq:
        inv = libdevice.div_rn(1.0, sin)
        zf = zi.to(tl.float32)
        if EVEN:
            x = tl.load(x_ptr + base + offs)
            q = libdevice.rint(x * inv).to(tl.int32) + zi
            valid = (q >= qmin) & (q <= qmax)
            qc = tl.minimum(tl.maximum(q, qmin), qmax)
            tl.store(out_ptr + base + offs, (qc.to(tl.float32) - zf) * sin)
            tl.store(mask_ptr + base + offs, valid.to(tl.uint8))
        else:
            msk = offs < I
            x = tl.load(x_ptr + base + offs, mask=msk, other=0.0)
            q = libdevice.rint(x * inv).to(tl.int32) + zi
            valid = (q >= qmin) & (q <= qmax)
            qc = tl.minimum(tl.maximum(q, qmin), qmax)
            tl.store(out_ptr + base + offs, (qc.to(tl.float32) - zf) * sin, mask=msk)
            tl.store(mask_ptr + base + offs, valid.to(tl.uint8), mask=msk)
    else:
        if EVEN:
            x = tl.load(x_ptr + base + offs)
            tl.store(out_ptr + base + offs, x)
            tl.store(mask_ptr + base + offs, tl.full((BLOCK,), 1, tl.uint8))
        else:
            msk = offs < I
            x = tl.load(x_ptr + base + offs, mask=msk, other=0.0)
            tl.store(out_ptr + base + offs, x, mask=msk)
            tl.store(mask_ptr + base + offs, tl.full((BLOCK,), 1, tl.uint8), mask=msk)


def _f32(v):
    return struct.unpack("<f", struct.pack("<f", float(v)))[0]


def _recip(v):
    hi = _f32(1.0 / v)
    lo = _f32(1.0 / v - hi)
    return hi, lo


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _fused_moving_avg_obs_fq_helper(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    logger.debug("GEMS_MTHREADS _FUSED_MOVING_AVG_OBS_FQ_HELPER")

    if not _use_triton_kernel(self):
        return default_fused_moving_avg_obs_fq_helper(
            self,
            observer_on,
            fake_quant_on,
            running_min,
            running_max,
            scale,
            zero_point,
            averaging_const,
            quant_min,
            quant_max,
            ch_axis,
            per_row_fake_quant,
            symmetric_quant,
        )

    with torch_device_fn.device(self.device):
        return _launch_fused_moving_avg(
            self,
            observer_on,
            fake_quant_on,
            running_min,
            running_max,
            scale,
            zero_point,
            averaging_const,
            quant_min,
            quant_max,
            ch_axis,
            per_row_fake_quant,
            symmetric_quant,
        )


def _launch_fused_moving_avg(
    self,
    observer_on,
    fake_quant_on,
    running_min,
    running_max,
    scale,
    zero_point,
    averaging_const,
    quant_min,
    quant_max,
    ch_axis,
    per_row_fake_quant=False,
    symmetric_quant=False,
):
    if self.dtype is not torch.float32:
        raise RuntimeError(
            "expected scalar type Float but found "
            + _SCALAR_TYPE_NAMES.get(self.dtype, str(self.dtype))
        )

    sym = bool(symmetric_quant)
    pc = bool(per_row_fake_quant)
    avg = float(averaging_const)
    qmin = int(quant_min)
    qmax = int(quant_max)

    # The flags are device tensors; read them inside the kernels so a call never
    # synchronizes the host.  A CPU flag (defensive) is read on the host and
    # passed as a compile-time constant instead.
    flag_ptr = observer_on.device.type != "cpu" and fake_quant_on.device.type != "cpu"
    if flag_ptr:
        obs_ptr = observer_on
        fq_ptr = fake_quant_on
        obs_c = 0
        fq_c = 0
    else:
        obs_c = int(observer_on.item())
        fq_c = int(fake_quant_on.item())
        obs_ptr = scale
        fq_ptr = scale

    N = self.numel()
    if pc:
        nd = self.dim()
        ch = int(ch_axis) % nd if nd > 0 else 0
        C = int(self.shape[ch])
        INNER = 1
        for d in range(ch + 1, nd):
            INNER *= int(self.shape[d])
        OUTER = N // (C * INNER) if C * INNER else 1
    else:
        C, INNER, OUTER = 1, N, 1

    dev = self.device
    xf = self.view(-1)
    out = torch.empty_like(self)
    outf = out.view(-1)
    mask = torch.empty(self.shape, dtype=torch.bool, device=dev)
    maskf = mask.view(torch.uint8)

    den = float(qmax - qmin)
    chi, clo = _recip(den)
    zmid = float(round((qmin + qmax) / 2.0))
    chi_n, clo_n = _recip(float(-qmin) + zmid)
    chi_p, clo_p = _recip(float(qmax) - zmid)

    tiles = (INNER + _BLOCK_R - 1) // _BLOCK_R
    S = -(-_TARGET_PROGS // (C * OUTER))
    S = max(1, min(1024, S))
    S = min(S, max(1, tiles))
    nfull = INNER // _BLOCK_R
    ct = -(-nfull // S) if nfull else 0
    if pc and OUTER == 1 and C >= 128:
        # one program per channel: single launch, and the quantize pass re-reads
        # the channel the same program just streamed.  The tile is half the size
        # used elsewhere because this kernel holds two reduce accumulators plus
        # the tile, so a smaller tile halves the per-lane register footprint and
        # lets more channel programs reside per SM.
        _fused_chan_kernel[(C,)](
            xf,
            outf,
            maskf,
            running_min,
            running_max,
            scale,
            zero_point,
            obs_ptr,
            fq_ptr,
            C,
            INNER,
            avg,
            qmin,
            qmax,
            chi,
            clo,
            chi_n,
            clo_n,
            chi_p,
            clo_p,
            FLAG_PTR=flag_ptr,
            OBS=obs_c,
            FQ=fq_c,
            SYM=sym,
            RBLK=_BLOCK_F,
            ABLK=_BLOCK_F,
            EVEN_R=(INNER % _BLOCK_F == 0),
            EVEN_A=(INNER % _BLOCK_F == 0),
            num_warps=4,
        )
        return (out, mask)

    cmin = torch.empty(C * OUTER * S, dtype=torch.float32, device=dev)
    cmax = torch.empty(C * OUTER * S, dtype=torch.float32, device=dev)
    _reduce_kernel[(C, OUTER, S)](
        xf,
        cmin,
        cmax,
        obs_ptr,
        C,
        INNER,
        S,
        nfull,
        ct,
        BLOCK=_BLOCK_R,
        TAIL=(INNER % _BLOCK_R) != 0,
        FLAG_PTR=flag_ptr,
        OBS=obs_c,
        num_warps=4,
    )

    np_ = OUTER * S
    if C == 1:
        blk = triton.next_power_of_2(max(1, np_))
        _fin_scalar_kernel[(1,)](
            cmin,
            cmax,
            running_min,
            running_max,
            scale,
            zero_point,
            obs_ptr,
            fq_ptr,
            np_,
            avg,
            qmin,
            qmax,
            chi,
            clo,
            chi_n,
            clo_n,
            chi_p,
            clo_p,
            FLAG_PTR=flag_ptr,
            OBS=obs_c,
            FQ=fq_c,
            SYM=sym,
            BLOCK=blk,
            num_warps=1,
        )
    else:
        _fin_vec_kernel[(triton.cdiv(C, _BLOCK_C),)](
            cmin,
            cmax,
            running_min,
            running_max,
            scale,
            zero_point,
            obs_ptr,
            fq_ptr,
            C,
            np_,
            avg,
            qmin,
            qmax,
            chi,
            clo,
            chi_n,
            clo_n,
            chi_p,
            clo_p,
            FLAG_PTR=flag_ptr,
            OBS=obs_c,
            FQ=fq_c,
            SYM=sym,
            BLOCK=_BLOCK_C,
            num_warps=4,
        )

    even_a = INNER % _BLOCK_A == 0
    _apply_kernel[(triton.cdiv(INNER, _BLOCK_A), C * OUTER)](
        xf,
        outf,
        maskf,
        scale,
        zero_point,
        fq_ptr,
        C,
        INNER,
        qmin,
        qmax,
        PC=pc,
        FLAG_PTR=flag_ptr,
        FQ=fq_c,
        BLOCK=_BLOCK_A,
        EVEN=even_a,
        num_warps=4,
    )

    return (out, mask)
