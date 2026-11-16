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
"""Kunlunxin (XPU3) implementation of ``aten::_fused_moving_avg_obs_fq_helper``.

Two independent reasons for a backend override, both measured on XPU3
(``harness/run_logs/fused_moving_avg_obs_fq_helper_batch13``):

1. The stock generic file routes every per-channel case **and** every per-tensor
   input with ``numel <= 65536`` through its per-group ``_fused_kernel``, which
   the current XPU3 pipeline cannot compile: ``TritonXPULegalize`` aborts with
   ``reduction rowspercore is not consistent`` (the kernel mixes a BLOCK-wide
   min/max reduction with shape-(1,) reductions) and the fake-quant branch also
   trips ``'arith.addi' op requires the same type for all operands``.  On the
   stock tree 38 of the 64 real functional cells fail to build.

2. ``libdevice.rint`` in the elementwise fake-quant pass is compute bound and
   does not vectorize on XPU3 (37.2 ms vs 2.96 ms for the magic-number identity
   at 128M fp32 -- ``wiki/techniques/xpu3-magic-number-round-half-even.md``),
   and the two-comparison range check scalarises the byte (mask) store
   (``wiki/techniques/xpu3-product-form-range-mask.md``).

Layout: one partial-min/max kernel (one contiguous span per program, no
atomics -- ``tl.atomic_min/max`` silently loses whole blocks on this backend),
one single-program finalize (per-tensor: fold the partials and update
``running_min``/``running_max``/``scale``/``zero_point``; per-channel: the same
math on the channel vector, no reduction), and one grid-strided elementwise
pass.  Everything is 32-bit; ``torch.empty``/``torch.zeros`` only allocate.

``_rne`` (round-half-to-even) is ``(y + 1.5*2**23) - 1.5*2**23``: exact for
``|y| <= 2**22`` and, for any larger ``|y|``, a value that is still far outside
``[quant_min, quant_max]`` so it saturates to the same clamp bound as
``rint``.  Ties, ``NaN`` and ``+-Inf`` therefore round/clamp exactly like the
generic ``libdevice.rint`` path, but the pass reaches the memory floor instead
of being rounding bound.  The kernel requires the backend's
``enable_fp_fusion=False`` (it does not reassociate the ``(y + M) - M`` pair).
"""

import logging
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

_SCALAR_TYPE_NAMES = {
    torch.float16: "Half",
    torch.bfloat16: "BFloat16",
    torch.float64: "Double",
}

# ``1.5 * 2**23``: the magic constant of the round-half-to-even identity.
_RNE_M = 12582912.0
# Partial-reduction / elementwise tile width.  Measured on XPU3
# (``harness/perf_ir/probe_fmaofq_helper_{tile,reduce}.log``): BLOCK=8192 with
# one warp is the sweet spot for both the min/max pass and the fake-quant pass
# at every size tried (65536 gives 3.47 ms vs 1.88 ms for the elementwise body
# at 67M fp32); 2048 collapses (VRF/local-memory pressure), so 4096 is the
# floor and the width never exceeds the per-group span.
_BLOCK_CAP = 8192
_BLOCK_MIN = 4096
_PARTIAL_G = 256
_ELEM_G = 1024
# Channel-vector width for the per-channel finalize (no reductions there).
_FCH_BLOCK = 1024
# Native stores/uses a floored scale (6.1e-5) while zero_point is derived from
# the unfloored value; keep that quirk to stay output-identical.
_SCALE_FLOOR = 6.1e-5


def _pick_block(span):
    """Largest power-of-two tile <= min(span, cap), floored at _BLOCK_MIN."""
    b = _BLOCK_MIN
    while b < _BLOCK_CAP and b * 2 <= span:
        b *= 2
    return b

# XPU3 fuses the ``cmpf`` only when the compare is seen by the normalizer at
# compile time; the env var is set just around the first (compiling) launch of
# each kernel signature, exactly like ``fake_quantize_per_tensor_affine_
# cachemask`` does, so the steady-state host path is untouched.
_FUSION_ENV = (("TRITONXPU_COMPARE_FUSION", "1"),)
_WARMED = set()


def _launch(key, fn):
    if key in _WARMED:
        return fn()
    saved = {k: os.environ.get(k) for k, _ in _FUSION_ENV}
    for k, v in _FUSION_ENV:
        os.environ[k] = v
    try:
        res = fn()
    finally:
        for k, _ in _FUSION_ENV:
            if saved[k] is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = saved[k]
    _WARMED.add(key)
    return res


@triton.jit
def _rint(x):
    # Round half to even on a scalar/short vector.  Only used by the
    # single-program per-tensor finalize (one lane), where the extern call is
    # free; the hot elementwise passes use the magic-number identity.
    return libdevice.rint(x)


@triton.jit(do_not_specialize=["N", "CHUNK"])
def _reduce_partials(
    x_ptr, pmin_ptr, pmax_ptr, N, CHUNK, SPAN: tl.constexpr, BLOCK: tl.constexpr
):
    # One program per contiguous CHUNK-sized span.  No atomics: the platform's
    # atomic min/max lose whole blocks.
    #
    # SPAN=True is the per-channel mode: a span is exactly one channel's block
    # (CHUNK == INNER, C*INNER == N), so each program must stop at its own span
    # end -- ``idx < N`` would fold the next channels into this one.
    # SPAN=False is the per-tensor mode: spans tile [0, N) and a group whose
    # tile overruns into the next group only re-reads elements, which min/max
    # absorbs (every element still belongs to at least its own group), so the
    # global min/max is exact.
    #
    # One strict comparison against ONE runtime bound is deliberate: an
    # ``(a < end) & (a < N)`` mask silently drops byte stores on XPU3
    # (``pattern-xpu3-masked-store-constexpr-tail``).
    pid = tl.program_id(0)
    start = pid * CHUNK
    end = start + CHUNK
    lidx = tl.arange(0, BLOCK)
    acc_min = tl.full((BLOCK,), float("inf"), tl.float32)
    acc_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    for i in range(start, end, BLOCK):
        idx = i + lidx
        if SPAN:
            m = idx < end
        else:
            m = idx < N
        v = tl.load(x_ptr + idx, mask=m, other=0.0).to(tl.float32)
        acc_min = tl.minimum(acc_min, tl.where(m, v, float("inf")))
        acc_max = tl.maximum(acc_max, tl.where(m, v, float("-inf")))
    tl.store(pmin_ptr + pid, tl.min(acc_min, axis=0))
    tl.store(pmax_ptr + pid, tl.max(acc_max, axis=0))


@triton.jit(
    do_not_specialize=["G", "avg", "qmin", "qmax"],
)
def _finalize_pt(
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
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Single-program finalize for the per-tensor group: fold the G partials and
    # update running stats / qparams in place.  Kept in shape-(1,) vectors
    # (the backend cannot lower scalar x scalar divisions) and with exactly one
    # BLOCK-wide reduction, because mixing reductions of different shapes makes
    # TritonXPULegalize abort ("reduction rowspercore is not consistent").
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
    if FQ:
        qmnv = qmnf + zf
        qmxv = qmxf + zf
        mn = tl.minimum(rmin, zf)
        mx = tl.maximum(rmax, zf)
        sc_a = (mx - mn) / (qmxv - qmnv)
        deg = sc_a == 0.0
        sc_a = tl.where(deg, 0.1 + zf, sc_a)
        z_a = _rint(qmnv - (mn / sc_a))
        z_a = tl.minimum(tl.maximum(z_a, qmnv), qmxv)
        z_a = tl.where(deg, tl.where(qmnv < 0.0, qmxv, qmnv), z_a)
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
        sc_v = tl.maximum(sc_v, FLOOR + zf)
        tl.store(scale_ptr + one, sc_v.to(scale_ptr.dtype.element_ty))
        tl.store(zp_ptr + one, z_v.to(zp_ptr.dtype.element_ty))


@triton.jit(
    do_not_specialize=["C", "avg", "qmin", "qmax"],
)
def _finalize_ch(
    pmin_ptr,
    pmax_ptr,
    rmin_ptr,
    rmax_ptr,
    scale_ptr,
    zp_ptr,
    C,
    avg,
    qmin,
    qmax,
    dnegf,
    dposf,
    zsymf,
    OBS: tl.constexpr,
    FQ: tl.constexpr,
    SYM: tl.constexpr,
    FLOOR: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Per-channel finalize: every channel owns its observed min/max (already
    # final), so this pass is pure vector math over the channel axis -- no
    # reduction, hence the same kernel can host both branches safely.  ``_rne``
    # is used instead of ``libdevice.rint`` because a >256-lane extern
    # rounding feeding a min/max chain fails to legalize on this backend.
    lidx = tl.arange(0, BLOCK)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    avgv = avg.to(tl.float32)
    for g in range(0, C, BLOCK):
        idx = g + lidx
        m = idx < C
        rmin = tl.load(rmin_ptr + idx, mask=m, other=0.0).to(tl.float32)
        rmax = tl.load(rmax_ptr + idx, mask=m, other=0.0).to(tl.float32)
        if OBS:
            cmn = tl.load(pmin_ptr + idx, mask=m, other=0.0).to(tl.float32)
            cmx = tl.load(pmax_ptr + idx, mask=m, other=0.0).to(tl.float32)
            rmin = rmin + avgv * (cmn - rmin)
            rmax = rmax + avgv * (cmx - rmax)
            tl.store(rmin_ptr + idx, rmin.to(rmin_ptr.dtype.element_ty), mask=m)
            tl.store(rmax_ptr + idx, rmax.to(rmax_ptr.dtype.element_ty), mask=m)
        if FQ:
            mn = tl.minimum(rmin, 0.0)
            mx = tl.maximum(rmax, 0.0)
            sc_a = (mx - mn) / (qmxf - qmnf)
            deg = sc_a == 0.0
            sc_a = tl.where(deg, 0.1, sc_a)
            z_a = _rne(qmnf - (mn / sc_a))
            z_a = tl.minimum(tl.maximum(z_a, qmnf), qmxf)
            z_a = tl.where(deg, tl.where(qmnf < 0.0, qmxf, qmnf), z_a)
            sc_v = sc_a
            z_v = z_a
            if SYM:
                both = (mn < 0.0) & (mx > 0.0)
                sc_s = tl.maximum((-mn) / dnegf, mx / dposf)
                sc_s = tl.where(sc_s == 0.0, 0.1, sc_s)
                sc_v = tl.where(both, sc_s, sc_a)
                z_v = tl.where(both, zsymf, z_a)
            sc_v = tl.maximum(sc_v, FLOOR)
            tl.store(scale_ptr + idx, sc_v.to(scale_ptr.dtype.element_ty), mask=m)
            tl.store(zp_ptr + idx, z_v.to(zp_ptr.dtype.element_ty), mask=m)


@triton.jit
def _rne(x):
    # Round half to even; see the module docstring for the exactness argument.
    return (x + 12582912.0) - 12582912.0


@triton.jit(
    do_not_specialize=["N", "CHUNK", "qmin", "qmax"],
)
def _fq_elem(
    x_ptr,
    out_ptr,
    mask_ptr,
    scale_ptr,
    zp_ptr,
    N,
    CHUNK,
    qmin,
    qmax,
    FQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Grid-strided elementwise pass for the per-tensor group.  The grid is
    # capped well below the device limit and every program walks its own span,
    # so the array is covered exactly once.  A group whose last tile overruns
    # into the next group recomputes the same values (single scale / zero_point
    # for the whole tensor), so ``off < N`` -- one comparison -- is enough and
    # is the form that keeps the masked stores intact on XPU3.
    pid = tl.program_id(0)
    start = pid * CHUNK
    end = start + CHUNK
    lidx = tl.arange(0, BLOCK)
    ones = tl.full((BLOCK,), 1, tl.uint8)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    sc = tl.load(scale_ptr).to(tl.float32)
    z = tl.load(zp_ptr).to(tl.float32)
    for i in range(start, end, BLOCK):
        off = i + lidx
        m = off < N
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        if FQ:
            q = _rne(v / sc) + z
            # ``q`` is integer valued, so the pre-clamp range check
            # ``quant_min <= q <= quant_max`` is exactly "both factors of
            # (q - qmin) * (qmax - q) are non-negative" -- one comparison
            # instead of two, which keeps the byte store off the scalarised
            # path.  NaN / +-Inf classify out of range, like the two-compare
            # form.
            inr = (q - qmnf) * (qmxf - q) >= 0.0
            qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
            res = (qc - z) * sc
            tl.store(out_ptr + off, res.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + off, inr.to(tl.uint8), mask=m)
        else:
            tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + off, ones, mask=m)


@triton.jit(
    do_not_specialize=["N", "CHUNK", "qmin", "qmax"],
)
def _fq_groups(
    x_ptr,
    out_ptr,
    mask_ptr,
    scale_ptr,
    zp_ptr,
    N,
    CHUNK,
    qmin,
    qmax,
    FQ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Per-channel elementwise pass: one program per channel, loading that
    # channel's scale / zero_point and walking its contiguous span.  INNER can
    # be narrower than BLOCK, so the span end is the only correct bound, and it
    # is kept as a single strict comparison (see the reduction kernel).
    pid = tl.program_id(0)
    start = pid * CHUNK
    end = start + CHUNK
    lidx = tl.arange(0, BLOCK)
    ones = tl.full((BLOCK,), 1, tl.uint8)
    qmnf = qmin.to(tl.float32)
    qmxf = qmax.to(tl.float32)
    sc = tl.load(scale_ptr + pid).to(tl.float32)
    z = tl.load(zp_ptr + pid).to(tl.float32)
    for i in range(start, end, BLOCK):
        off = i + lidx
        m = off < end
        v = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        if FQ:
            q = _rne(v / sc) + z
            inr = (q - qmnf) * (qmxf - q) >= 0.0
            qc = tl.minimum(tl.maximum(q, qmnf), qmxf)
            res = (qc - z) * sc
            tl.store(out_ptr + off, res.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + off, inr.to(tl.uint8), mask=m)
        else:
            tl.store(out_ptr + off, v.to(out_ptr.dtype.element_ty), mask=m)
            tl.store(mask_ptr + off, ones, mask=m)


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
    """Fused moving-average observer + fake-quantize helper (QAT).

    Matches ``aten._fused_moving_avg_obs_fq_helper``: updates running_min/max in
    place by moving average, recomputes scale/zero_point, and fake-quantizes the
    input; returns ``(output, mask)``.  Supports per-tensor and per-channel
    (``per_row_fake_quant``, ``ch_axis == 0``) modes.
    """
    logger.debug("GEMS_KUNLUNXIN _FUSED_MOVING_AVG_OBS_FQ_HELPER")
    # The native ATen operator is intentionally restricted to float32 input.
    if self.dtype is not torch.float32:
        scalar_type = _SCALAR_TYPE_NAMES.get(self.dtype, str(self.dtype))
        raise RuntimeError(f"expected scalar type Float but found {scalar_type}")

    per_channel = bool(per_row_fake_quant)
    if per_channel and int(ch_axis) != 0:
        raise RuntimeError(
            "Per-channel FakeQuant in fused_moving_avg_obs_fake_quant is only "
            "supported on axis == 0"
        )

    obs = int(observer_on) != 0
    fq = int(fake_quant_on) != 0
    sym = bool(symmetric_quant)

    x = self
    shape = x.shape
    N = x.numel()
    dev = x.device
    if N == 0:
        return (
            torch.empty(shape, dtype=torch.float32, device=dev),
            torch.empty(shape, dtype=torch.bool, device=dev),
        )

    if x.is_contiguous():
        xf = x.view(-1)
    else:
        xf = x.contiguous().view(-1)

    if per_channel:
        C = shape[0] if x.dim() > 0 else 1
        if C == 0:
            return (
                torch.empty(shape, dtype=torch.float32, device=dev),
                torch.empty(shape, dtype=torch.bool, device=dev),
            )
        INNER = N // C
    else:
        C = 1
        INNER = N
    # Tile width for the reduce and elementwise passes (per-group span bound).
    blk = _pick_block(INNER)

    out = torch.empty(N, dtype=torch.float32, device=dev)
    mask = torch.empty(N, dtype=torch.bool, device=dev)
    mask_u8 = mask.view(torch.uint8)

    qmin = int(quant_min)
    qmax = int(quant_max)
    # Symmetric-mode constants, precomputed on the host.  Python's round() is
    # round-half-to-even, matching libdevice.rint.
    dneg_h = float((qmax - qmin + 1) // 2)
    dpos_h = float((qmax - qmin) // 2)
    zsym_h = float(round((qmin + qmax) * 0.5))
    avg = float(averaging_const)

    with torch_device_fn.device(dev):
        if per_channel:
            # One partial per channel, one program per channel.
            if obs:
                pmin = torch.empty(C, dtype=torch.float32, device=dev)
                pmax = torch.empty(C, dtype=torch.float32, device=dev)
                _launch(
                    ("reduce", blk, True),
                    lambda: _reduce_partials[(C,)](
                        xf,
                        pmin,
                        pmax,
                        N,
                        INNER,
                        SPAN=True,
                        BLOCK=blk,
                        num_warps=1,
                    ),
                )
            else:
                pmin = pmax = scale
            if obs or fq:
                _launch(
                    ("finalize_ch", obs, fq, sym),
                    lambda: _finalize_ch[(1,)](
                        pmin,
                        pmax,
                        running_min,
                        running_max,
                        scale,
                        zero_point,
                        C,
                        avg,
                        qmin,
                        qmax,
                        dneg_h,
                        dpos_h,
                        zsym_h,
                        OBS=obs,
                        FQ=fq,
                        SYM=sym,
                        FLOOR=_SCALE_FLOOR,
                        BLOCK=_FCH_BLOCK,
                        num_warps=1,
                    ),
                )
            _launch(
                ("fq_groups", fq, blk),
                lambda: _fq_groups[(C,)](
                    xf,
                    out,
                    mask_u8,
                    scale,
                    zero_point,
                    N,
                    INNER,
                    qmin,
                    qmax,
                    FQ=fq,
                    BLOCK=blk,
                    # One warp is pathological for one-program-per-channel
                    # (16.5 ms vs 1.94 ms at 4096x16384, 4.4 vs 0.80 ms at
                    # 4096x4096: many programs each holding one BLOCK-wide tile
                    # regress under this backend's one-thread-per-warp layout).
                    num_warps=4,
                ),
            )
        else:
            # Per-tensor: tile N into at most _PARTIAL_G contiguous spans, fold
            # them in one program, then run the grid-strided elementwise pass.
            if obs:
                G = max(1, min(_PARTIAL_G, triton.cdiv(N, blk)))
                chunk = triton.cdiv(N, G)
                pmin = torch.empty(G, dtype=torch.float32, device=dev)
                pmax = torch.empty(G, dtype=torch.float32, device=dev)
                _launch(
                    ("reduce", blk, False),
                    lambda: _reduce_partials[(G,)](
                        xf,
                        pmin,
                        pmax,
                        N,
                        chunk,
                        SPAN=False,
                        BLOCK=blk,
                        num_warps=1,
                    ),
                )
            else:
                G = 0
                pmin = pmax = scale
            if obs or fq:
                _launch(
                    ("finalize_pt", obs, fq, sym),
                    lambda: _finalize_pt[(1,)](
                        pmin,
                        pmax,
                        running_min,
                        running_max,
                        scale,
                        zero_point,
                        G,
                        avg,
                        qmin,
                        qmax,
                        dneg_h,
                        dpos_h,
                        zsym_h,
                        OBS=obs,
                        FQ=fq,
                        SYM=sym,
                        FLOOR=_SCALE_FLOOR,
                        # only the G partials are touched here (G <= 256)
                        BLOCK=_pick_block(max(G, 1)),
                        num_warps=4,
                    ),
                )
            n_prog = min(triton.cdiv(N, blk), _ELEM_G)
            echunk = (N + n_prog - 1) // n_prog
            _launch(
                ("fq_elem", fq, blk),
                lambda: _fq_elem[(n_prog,)](
                    xf,
                    out,
                    mask_u8,
                    scale,
                    zero_point,
                    N,
                    echunk,
                    qmin,
                    qmax,
                    FQ=fq,
                    BLOCK=blk,
                    num_warps=1,
                ),
            )

    return out.view(shape), mask.view(shape)


# Dedicated tests/benchmarks may resolve the op through ``flag_gems.ops``;
# ``SpecOpRegistrar`` only rewrites the top-level ``flag_gems`` namespace.
import sys  # noqa: E402

_generic_ops_module = sys.modules.get("flag_gems.ops")
if _generic_ops_module is not None:
    setattr(_generic_ops_module, "_fused_moving_avg_obs_fq_helper",
            _fused_moving_avg_obs_fq_helper)
