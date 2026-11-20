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
import triton.language.extra.libdevice as libdevice

logger = logging.getLogger(__name__)

# Per-tensor path tuning, mirroring the proven reduction layout in the generic
# _fused_moving_avg_obs_fq_helper.  _BLOCK is the per-lane DMA width for the
# reduction / finalize programs; _LARGE_BLOCK is the width for the elementwise
# fake-quant / copy passes.
_BLOCK = 16384
_LARGE_BLOCK = 65536
_PARTIAL_G = 256
_ELEM_G = 1024


@triton.jit
def _rint(x):
    # Round half to even, matching torch.round / libdevice.rint semantics.
    return libdevice.rint(x)


@triton.jit
def _reduce_partials(x_ptr, pmin_ptr, pmax_ptr, N, CHUNK, BLOCK: tl.constexpr):
    # Grid-strided partial min/max.  Each program owns a contiguous CHUNK span
    # and reduces it exactly (no atomics -- the platform has no reliable atomic
    # primitive, so the generic tl.atomic_min/max path silently loses updates).
    pid = tl.program_id(0)
    start = pid * CHUNK
    lidx = tl.arange(0, BLOCK)
    acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
    acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    for i in range(start, start + CHUNK, BLOCK):
        idx = i + lidx
        m = idx < N
        v = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
        acc_min = tl.minimum(acc_min, tl.where(m, v, float("inf")))
        acc_max = tl.maximum(acc_max, tl.where(m, v, float("-inf")))
    tl.store(pmin_ptr + pid, tl.min(acc_min, axis=0))
    tl.store(pmax_ptr + pid, tl.max(acc_max, axis=0))


@triton.jit
def _finalize(
    pmin_ptr,
    pmax_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    G,
    avg,
    qmin,
    qmax,
    dnegf,
    dposf,
    zsymf,
    OBS: tl.constexpr,
    SYM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single-program finalize: combine partials, update running stats in place
    # and recompute scale / zero_point.  scale/zp are recomputed unconditionally
    # (unlike the helper, the moving-avg observer always refreshes qparams) using
    # the generic operator's exact math -- no scale floor.
    one = tl.arange(0, 1)
    zf = tl.zeros((1,), tl.float32)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    rmin = tl.load(rmin_ptr + one).to(tl.float32)
    rmax = tl.load(rmax_ptr + one).to(tl.float32)
    if OBS:
        lidx = tl.arange(0, BLOCK)
        acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
        acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
        for i in range(0, G, BLOCK):
            idx = i + lidx
            m = idx < G
            pmn = tl.load(pmin_ptr + idx, mask=m, other=float("inf"))
            pmx = tl.load(pmax_ptr + idx, mask=m, other=float("-inf"))
            acc_min = tl.minimum(acc_min, pmn)
            acc_max = tl.maximum(acc_max, pmx)
        cmn = tl.min(acc_min, axis=0) + zf
        cmx = tl.max(acc_max, axis=0) + zf
        avgv = avg.to(tl.float32) + zf
        rmin = rmin + avgv * (cmn - rmin)
        rmax = rmax + avgv * (cmx - rmax)
        tl.store(rmin_ptr + one, rmin.to(rmin_ptr.dtype.element_ty))
        tl.store(rmax_ptr + one, rmax.to(rmax_ptr.dtype.element_ty))

    qmnv = qmnf + zf
    qmxv = qmxf + zf
    mn = tl.minimum(rmin, zf)
    mx = tl.maximum(rmax, zf)
    sc_a = (mx - mn) / (qmxv - qmnv)
    sc_a = tl.where(sc_a == 0.0, 0.1 + zf, sc_a)
    z_a = _rint(qmnv - (mn / sc_a))
    z_a = tl.minimum(tl.maximum(z_a, qmnv), qmxv)
    sc_v = sc_a
    z_v = z_a
    if SYM:
        dneg = dnegf + zf
        dpos = dposf + zf
        both = (mn < 0.0) & (mx > 0.0)
        sc_s = tl.maximum((-mn) / dneg, mx / dpos)
        sc_s = tl.where(sc_s == 0.0, 0.1 + zf, sc_s)
        z_s = zsymf + zf
        sc_v = tl.where(both, sc_s, sc_a)
        z_v = tl.where(both, z_s, z_a)
    tl.store(scale_ptr + one, sc_v.to(scale_ptr.dtype.element_ty))
    tl.store(zp_ptr + one, z_v.to(zp_ptr.dtype.element_ty))


@triton.jit
def _fq_elem(
    x_ptr,
    out_ptr,
    scale_ptr,
    zp_ptr,
    N,
    CHUNK,
    qmin,
    qmax,
    BLOCK: tl.constexpr,
):
    # Grid-strided elementwise fake-quant.  The launch grid is capped below the
    # device grid.x limit; each program walks its own CHUNK span so the whole
    # array is covered exactly once.
    pid = tl.program_id(0)
    start = pid * CHUNK
    lidx = tl.arange(0, BLOCK)
    sc = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    for i in range(start, start + CHUNK, BLOCK):
        off = i + lidx
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        q = _rint(v / sc) + z
        qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
        res = (qc - z) * sc
        tl.store(out_ptr + off, res.to(out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _copy_elem(x_ptr, out_ptr, N, CHUNK, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    start = pid * CHUNK
    lidx = tl.arange(0, BLOCK)
    for i in range(start, start + CHUNK, BLOCK):
        off = i + lidx
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0)
        tl.store(out_ptr + off, v, mask=m)


def fused_moving_avg_obs_fake_quant(
    input,
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
    """Fused moving-average observer + fake-quantize (per-tensor).

    Replaces the generic tl.atomic_min/max reduction (unreliable on this
    backend) with a two-pass partial-reduction + single-program finalize.  The
    observed min/max, moving-average update and scale/zero_point math match the
    generic operator exactly, so running_min/max/scale/zero_point are mutated in
    place with identical semantics.
    """
    logger.debug("GEMS_KUNLUNXIN FUSED_MOVING_AVG_OBS_FAKE_QUANT")

    obs = (
        bool(observer_on.item()) if torch.is_tensor(observer_on) else bool(observer_on)
    )
    fq = (
        bool(fake_quant_on.item())
        if torch.is_tensor(fake_quant_on)
        else bool(fake_quant_on)
    )
    sym = bool(symmetric_quant)

    x = input.contiguous().view(-1).float()
    N = x.numel()
    dev = x.device

    if N == 0:
        return torch.empty(input.shape, dtype=input.dtype, device=dev)

    out = torch.empty_like(x)

    qmin = int(quant_min)
    qmax = int(quant_max)
    dneg_h = float(-qmin)
    dpos_h = float(qmax)
    zsym_h = float(round((qmin + qmax) * 0.5))

    if obs:
        G = _PARTIAL_G
        chunk = (N + G - 1) // G
        pmin = torch.empty(G, dtype=torch.float32, device=dev)
        pmax = torch.empty(G, dtype=torch.float32, device=dev)
        _reduce_partials[(G,)](
            x,
            pmin,
            pmax,
            N,
            chunk,
            BLOCK=_BLOCK,
            num_warps=1,
        )
    else:
        G = 0
        pmin = pmax = scale

    _finalize[(1,)](
        pmin,
        pmax,
        running_min,
        running_max,
        scale,
        zero_point,
        G,
        float(averaging_const),
        qmin,
        qmax,
        dneg_h,
        dpos_h,
        zsym_h,
        OBS=obs,
        SYM=sym,
        BLOCK=_BLOCK,
        num_warps=4,
    )

    n_prog = min(triton.cdiv(N, _LARGE_BLOCK), _ELEM_G)
    echunk = (N + n_prog - 1) // n_prog
    grid = (n_prog,)
    if fq:
        _fq_elem[grid](
            x,
            out,
            scale,
            zero_point,
            N,
            echunk,
            qmin,
            qmax,
            BLOCK=_LARGE_BLOCK,
            num_warps=1,
        )
    else:
        _copy_elem[grid](
            x,
            out,
            N,
            echunk,
            BLOCK=_LARGE_BLOCK,
            num_warps=1,
        )

    return out.view(input.shape).to(input.dtype)
