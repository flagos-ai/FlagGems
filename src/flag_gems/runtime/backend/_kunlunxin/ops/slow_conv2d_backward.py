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

"""Kunlunxin (XPU3) override for ``aten::_slow_conv2d_backward.output_mask``.

The generic module's ``_gw_kernel`` is a masked-load implicit-GEMM whose
``tl.load(..., mask=..., other=0.0)`` operands trip the XPU MLIR pass pipeline
(``TritonXPUOffsetAnalysis`` failure, and earlier ``uni_sram`` OOR), so the whole
operator failed to compile for the multi-channel shapes.

This override keeps the same deterministic, atomic-free algorithm (every output
is a gather / serial accumulation — never ``tl.atomic_add``, which is unreliable
on this backend), but rewrites every kernel mask-free following the proven
``slow_conv2d_forward`` pattern: all addresses are clamped into range and the
only masking is a register-side ``tl.where`` on loaded values. Because the tail
lanes of every tile are clamped to the last valid element, they recompute that
element's full result and write it back to the same clamped slot, so the stores
are idempotent and need no store predicate (masked stores are not reliably
honoured on XPU3).

Gradients produced (all deterministic):
  * grad_input : per (n, ci, ih, iw) gather over (co, kh, kw).
  * grad_weight: per (co, ci, kh, kw) serial reduction over (n, ho, wo).
  * grad_bias  : rides on the same (n, ho, wo) reduction as grad_weight.
"""

import logging
import sys as _sys

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

__all__ = ["_slow_conv2d_backward"]


# --------------------------------------------------------------------------- #
# grad_input: dgrad, gather over (co, kh, kw) with clamped, unmasked loads
# --------------------------------------------------------------------------- #
@triton.jit(
    do_not_specialize=[
        "CO",
        "CI",
        "HO",
        "WO",
        "H",
        "W",
        "SGO_N",
        "SGO_CO",
        "SGO_HO",
        "SGO_WO",
        "SW_CO",
        "SW_CI",
        "SW_KH",
        "SW_KW",
        "SO_N",
        "SO_CI",
        "SO_H",
        "SO_W",
    ]
)
def _sc2b_gi_kernel(
    go_ptr,
    w_ptr,
    out_ptr,
    CO,
    CI,
    HO,
    WO,
    H,
    W,
    SGO_N,
    SGO_CO,
    SGO_HO,
    SGO_WO,
    SW_CO,
    SW_CI,
    SW_KH,
    SW_KW,
    SO_N,
    SO_CI,
    SO_H,
    SO_W,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_ci = tl.program_id(1)
    n = tl.program_id(2)

    HW = H * W
    hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    # Tail lanes fold onto the last valid input position: they recompute its
    # exact result and store back to the same slot (idempotent, no store mask).
    hwc = tl.minimum(hw, HW - 1)
    ih = hwc // W
    iw = hwc - ih * W

    ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    cic = tl.minimum(ci, CI - 1)

    acc = tl.zeros((BLOCK_CI, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(KH):
        hh = ih + PH - kh
        oh = hh // SH
        okh = (hh >= 0) & (hh - oh * SH == 0) & (oh < HO)
        ohc = tl.minimum(tl.maximum(oh, 0), HO - 1)
        for kw in tl.static_range(KW):
            ww = iw + PW - kw
            ow = ww // SW
            okw = (ww >= 0) & (ww - ow * SW == 0) & (ow < WO)
            owc = tl.minimum(tl.maximum(ow, 0), WO - 1)
            ok = okh & okw
            go_base = go_ptr + n * SGO_N + ohc * SGO_HO + owc * SGO_WO
            w_base = w_ptr + cic * SW_CI + kh * SW_KH + kw * SW_KW
            for co_i in range(0, CO):
                g = tl.load(go_base + co_i * SGO_CO)
                wv = tl.load(w_base + co_i * SW_CO)
                # zero-pad the out-of-range taps BEFORE the multiply so they
                # contribute an exact 0 even for a NaN weight (reference rule)
                acc += tl.where(
                    ok[None, :],
                    wv[:, None].to(tl.float32) * g[None, :].to(tl.float32),
                    0.0,
                )

    o_off = n * SO_N + cic[:, None] * SO_CI + ih[None, :] * SO_H + iw[None, :] * SO_W
    tl.store(out_ptr + o_off, acc.to(out_ptr.dtype.element_ty))


# --------------------------------------------------------------------------- #
# grad_weight (+ grad_bias): serial reduction over (n, ho, wo)
# --------------------------------------------------------------------------- #
@triton.jit(
    do_not_specialize=[
        "CO",
        "CI",
        "N",
        "HO",
        "WO",
        "H",
        "W",
        "SGO_N",
        "SGO_CO",
        "SGO_HO",
        "SGO_WO",
        "SX_N",
        "SX_CI",
        "SX_H",
        "SX_W",
        "SP_CO",
        "SP_PID",
        "BP_PID",
    ]
)
def _sc2b_gw_kernel(
    go_ptr,
    x_ptr,
    part_ptr,
    bpart_ptr,
    CO,
    CI,
    N,
    HO,
    WO,
    H,
    W,
    SGO_N,
    SGO_CO,
    SGO_HO,
    SGO_WO,
    SX_N,
    SX_CI,
    SX_H,
    SX_W,
    SP_CO,
    SP_PID,
    BP_PID,
    KH: tl.constexpr,
    KW: tl.constexpr,
    SH: tl.constexpr,
    SW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    nsplit = tl.num_programs(2)

    KK = KH * KW
    CIKK = CI * KK
    K = N * HO * WO
    HW = HO * WO

    co = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    coc = tl.minimum(co, CO - 1)
    jj = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # Tail lanes fold onto the last valid (ci, kh, kw) column and store back to
    # it, so stores stay idempotent and need no predicate.
    jjc = tl.minimum(jj, CIKK - 1)
    cij = jjc // KK
    rem = jjc - cij * KK
    khj = rem // KW
    kwj = rem - khj * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if HAS_B:
        bid = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Each program owns a contiguous slice of the flattened (n, ho, wo) axis.
    k_begin = (K * pid_k) // nsplit
    k_end = (K * (pid_k + 1)) // nsplit
    for k in range(k_begin, k_end):
        n = k // HW
        r = k - n * HW
        ho = r // WO
        wo = r - ho * WO
        g = tl.load(go_ptr + n * SGO_N + coc * SGO_CO + ho * SGO_HO + wo * SGO_WO)
        ih = ho * SH - PH + khj
        iw = wo * SW - PW + kwj
        ok = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
        iwc = tl.minimum(tl.maximum(iw, 0), W - 1)
        xv = tl.load(x_ptr + n * SX_N + cij * SX_CI + ihc * SX_H + iwc * SX_W)
        xv = tl.where(ok, xv, 0.0).to(tl.float32)
        if HAS_B:
            # grad_bias shares this reduction axis; a NaN in grad_output must
            # propagate here too, so add the raw (unmasked) g.
            bid += g.to(tl.float32)
        # NOTE: product is NOT masked with `ok`: reference zero-pads the im2col
        # row and then multiplies, so 0 * NaN keeps propagating NaN.
        acc += g[:, None].to(tl.float32) * xv[None, :]

    tl.store(
        part_ptr + pid_k * SP_PID + coc[:, None] * SP_CO + jjc[None, :],
        acc.to(part_ptr.dtype.element_ty),
    )
    if HAS_B:
        # bid is identical in every pid_n program (it does not depend on jj), so
        # an unconditional idempotent store is safe and avoids a store predicate.
        tl.store(
            bpart_ptr + pid_k * BP_PID + coc,
            bid.to(bpart_ptr.dtype.element_ty),
        )


@triton.jit(
    do_not_specialize=[
        "CO",
        "CIKK",
        "SPLIT_K",
        "SP_PID",
        "SP_CO",
    ]
)
def _sc2b_gw_reduce_kernel(
    part_ptr,
    gw_ptr,
    CO,
    CIKK,
    SPLIT_K,
    SP_PID,
    SP_CO,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    co = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    coc = tl.minimum(co, CO - 1)
    jj = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    jjc = tl.minimum(jj, CIKK - 1)

    # Deterministic partial reduction over the split-K slabs (no atomics).
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        acc += tl.load(part_ptr + s * SP_PID + coc[:, None] * SP_CO + jjc[None, :])
    tl.store(
        gw_ptr + coc[:, None] * SP_CO + jjc[None, :],
        acc.to(gw_ptr.dtype.element_ty),
    )


@triton.jit(
    do_not_specialize=[
        "CO",
        "SPLIT_K",
        "BP_PID",
    ]
)
def _sc2b_gb_reduce_kernel(
    bpart_ptr,
    gb_ptr,
    CO,
    SPLIT_K,
    BP_PID,
    BLOCK_CO: tl.constexpr,
):
    pid = tl.program_id(0)
    co = pid * BLOCK_CO + tl.arange(0, BLOCK_CO)
    coc = tl.minimum(co, CO - 1)
    bacc = tl.zeros((BLOCK_CO,), dtype=tl.float32)
    for s in range(0, SPLIT_K):
        bacc += tl.load(bpart_ptr + s * BP_PID + coc)
    tl.store(gb_ptr + coc, bacc.to(gb_ptr.dtype.element_ty))


# --------------------------------------------------------------------------- #
# Affine (row-contiguous) fast path for grad_weight / grad_bias.
#
# The kernels above derive every address from a per-lane div/mod, which on XPU3
# lowers to one DMA descriptor per lane (see wiki
# ``pattern-xpu3-block-dma-descriptor-cost``): latency tracks descriptor count,
# not bytes.  The kernels below make every load an affine contiguous run (one
# descriptor per row), and accumulate the row segments element-wise into a
# register tile that is folded **once** after the loop.  No ``tl.dot`` (avoids
# the bf16 dot-operand miscompile and the SDNN event budget), no atomics, no
# masked stores, no in-loop reductions.
# --------------------------------------------------------------------------- #
@triton.jit(
    do_not_specialize=[
        "CO",
        "CI",
        "N",
        "HO",
        "WO",
        "H",
        "W",
        "SGO_N",
        "SGO_CO",
        "SGO_HO",
        "SX_N",
        "SX_CI",
        "SX_H",
        "SGW_CO",
    ]
)
def _sc2b_gw2_kernel(
    go_ptr,
    x_ptr,
    gw_ptr,
    CO,
    CI,
    N,
    HO,
    WO,
    H,
    W,
    SGO_N,
    SGO_CO,
    SGO_HO,
    SX_N,
    SX_CI,
    SX_H,
    SGW_CO,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PH: tl.constexpr,
    PW: tl.constexpr,
    BCO: tl.constexpr,
    BL: tl.constexpr,
):
    """grad_weight for one (ci, kh, kw) per program, vector accumulation.

    gw[co, ci, kh, kw] = sum_{n, oh, ow} go[n, co, oh, ow] * x[n, ci, oh+kh-PH, ow+kw-PW]

    Lane axes are (co, ow): one contiguous run per row for both operands.  The
    (n, oh) segments accumulate element-wise into a (BCO, BL) register tile and
    are folded **once** after the loop (no ``tl.sum`` inside the loop).
    """
    ci = tl.program_id(0)
    tap = tl.program_id(1)
    kh = tap // KW
    kw = tap - kh * KW

    co = tl.arange(0, BCO)
    coc = tl.minimum(co, CO - 1)
    okc = co < CO
    ow = tl.arange(0, BL)
    okw = ow < WO

    acc = tl.zeros((BCO, BL), dtype=tl.float32)
    for n in range(0, N):
        gob = go_ptr + n * SGO_N + coc[:, None] * SGO_CO + ow[None, :]
        xb = x_ptr + n * SX_N + ci * SX_CI + (kw - PW) + ow
        for oh in range(0, HO):
            ih = oh - PH + kh  # stride 1
            okh = (ih >= 0) & (ih < H)
            ihc = tl.minimum(tl.maximum(ih, 0), H - 1)
            g = tl.load(gob + oh * SGO_HO)
            xv = tl.load(xb + ihc * SX_H)
            g = tl.where(okw[None, :] & okh, g, 0.0).to(tl.float32)
            xv = tl.where(okw, xv, 0.0).to(tl.float32)
            acc += g * xv[None, :]

    s = tl.sum(acc, axis=1)  # single fold, after the loop
    tl.store(
        gw_ptr + coc[:, None] * SGW_CO + ci * (KH * KW) + tap,
        tl.where(okc, s, 0.0)[:, None].to(gw_ptr.dtype.element_ty),
    )


@triton.jit(
    do_not_specialize=[
        "CO",
        "N",
        "HW",
        "SGO_N",
        "SGO_CO",
    ]
)
def _sc2b_gb2_kernel(
    go_ptr,
    gb_ptr,
    CO,
    N,
    HW,
    SGO_N,
    SGO_CO,
    BLOCK: tl.constexpr,
):
    """grad_bias: one (n, co) plane is a flat contiguous run, so a whole plane
    is one descriptor; the plane is folded once per image."""
    co = tl.program_id(0)
    hw = tl.arange(0, BLOCK)
    ok = hw < HW
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for n in range(0, N):
        s += tl.where(ok, tl.load(go_ptr + n * SGO_N + co * SGO_CO + hw), 0.0).to(
            tl.float32
        )
    tot = tl.sum(s)
    tl.store(gb_ptr + co, tot.to(gb_ptr.dtype.element_ty))


# --------------------------------------------------------------------------- #
# host helpers
# --------------------------------------------------------------------------- #
def _pair(v, name):
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise RuntimeError(
                f"_slow_conv2d_backward: {name} must have exactly 2 elements"
            )
        return int(v[0]), int(v[1])
    raise RuntimeError(
        f"_slow_conv2d_backward: {name} must be a 2-element sequence, got {type(v)}"
    )


def _sc2b_gi_kernel_launch(go, w, out, CO, CI, HO, WO, H, W, kh, kw, sh, sw, ph, pw, N):
    """Launch the clamped-scalar grad_input kernel.

    16x64 keeps the accumulator small enough to avoid the XPU register /
    uni_sram cliff; grid parallelism comes from (hw, ci, n).
    """
    BLOCK_CI = 16
    BLOCK_HW = 64
    gi_warps = 4 if (H * W) * CI * N > 2 * 64 * BLOCK_HW else 2
    _sc2b_gi_kernel[(triton.cdiv(H * W, BLOCK_HW), triton.cdiv(CI, BLOCK_CI), N)](
        go,
        w,
        out,
        CO,
        CI,
        HO,
        WO,
        H,
        W,
        go.stride(0),
        go.stride(1),
        go.stride(2),
        go.stride(3),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        KH=kh,
        KW=kw,
        SH=sh,
        SW=sw,
        PH=ph,
        PW=pw,
        BLOCK_CI=BLOCK_CI,
        BLOCK_HW=BLOCK_HW,
        num_warps=gi_warps,
    )


def _slow_conv2d_backward(
    grad_output,
    self,
    weight,
    kernel_size,
    stride,
    padding,
    output_mask=None,
    *,
    grad_input=None,
    grad_weight=None,
    grad_bias=None,
    out0=None,
    out1=None,
    out2=None,
):
    """aten::_slow_conv2d_backward.output_mask(grad_output, self, weight, ...)."""
    logger.debug("GEMS_KUNLUNXIN _SLOW_CONV2D_BACKWARD")
    for t, nm in ((grad_output, "grad_output"), (self, "self"), (weight, "weight")):
        if not isinstance(t, torch.Tensor):
            raise RuntimeError(f"_slow_conv2d_backward: {nm} must be a Tensor")
        if t.dim() != 4:
            raise RuntimeError(f"_slow_conv2d_backward: {nm} must be 4-D")
        if not t.is_floating_point():
            raise RuntimeError(
                f"_slow_conv2d_backward: unsupported dtype {t.dtype} for {nm}"
            )
    if not (grad_output.dtype == self.dtype == weight.dtype):
        raise RuntimeError("_slow_conv2d_backward: dtype mismatch between inputs")

    kh, kw = _pair(kernel_size, "kernel_size")
    sh, sw = _pair(stride, "stride")
    ph, pw = _pair(padding, "padding")
    if sh <= 0 or sw <= 0 or ph < 0 or pw < 0 or kh <= 0 or kw <= 0:
        raise RuntimeError("_slow_conv2d_backward: invalid kernel/stride/padding")

    N, CI, H, W = self.shape
    CO, wci, wkh, wkw = weight.shape
    if wci != CI or wkh != kh or wkw != kw:
        raise RuntimeError(
            "_slow_conv2d_backward: weight shape incompatible with self/kernel_size"
        )
    HO = (H + 2 * ph - kh) // sh + 1
    WO = (W + 2 * pw - kw) // sw + 1
    if HO <= 0 or WO <= 0:
        raise RuntimeError("_slow_conv2d_backward: non-positive output size")
    if grad_output.shape != (N, CO, HO, WO):
        raise RuntimeError(
            "_slow_conv2d_backward: grad_output shape inconsistent with inputs"
        )

    if output_mask is None:
        mi = mw = mb = True
    else:
        if len(output_mask) != 3:
            raise RuntimeError("_slow_conv2d_backward: output_mask must have 3 entries")
        mi, mw, mb = bool(output_mask[0]), bool(output_mask[1]), bool(output_mask[2])

    oi = grad_input if grad_input is not None else out0
    ow = grad_weight if grad_weight is not None else out1
    ob = grad_bias if grad_bias is not None else out2

    dev = self.device
    dt = self.dtype

    gi = gw = gb = None
    if mi:
        gi = oi if oi is not None else torch.empty((N, CI, H, W), dtype=dt, device=dev)
        if not gi.is_contiguous():
            raise RuntimeError(
                "_slow_conv2d_backward: grad_input out must be contiguous"
            )
    if mw:
        gw = (
            ow
            if ow is not None
            else torch.empty((CO, CI, kh, kw), dtype=dt, device=dev)
        )
        if not gw.is_contiguous():
            raise RuntimeError(
                "_slow_conv2d_backward: grad_weight out must be contiguous"
            )
    if mb:
        gb = ob if ob is not None else torch.empty((CO,), dtype=dt, device=dev)
        if not gb.is_contiguous():
            raise RuntimeError(
                "_slow_conv2d_backward: grad_bias out must be contiguous"
            )

    # ------------------------------------------------------------------ #
    # Affine fast path: every load/store is a contiguous run (one DMA
    # descriptor per row) instead of a per-lane div/mod gather.
    # ------------------------------------------------------------------ #
    fast = (
        sh == 1
        and sw == 1
        and grad_output.is_contiguous()
        and self.is_contiguous()
        and weight.is_contiguous()
        and (W & (W - 1)) == 0
        and (CI & (CI - 1)) == 0
        and CI * W >= 64
        and HO * WO <= 8192
    )
    if fast:
        BL = 1 << ((WO - 1).bit_length())
        BCO = 1 << ((CO - 1).bit_length())

        if mw:
            _sc2b_gw2_kernel[(CI, kh * kw)](
                grad_output,
                self,
                gw,
                CO,
                CI,
                N,
                HO,
                WO,
                H,
                W,
                grad_output.stride(0),
                grad_output.stride(1),
                grad_output.stride(2),
                self.stride(0),
                self.stride(1),
                self.stride(2),
                CI * kh * kw,
                KH=kh,
                KW=kw,
                PH=ph,
                PW=pw,
                BCO=BCO,
                BL=BL,
            )
        if mb:
            _sc2b_gb2_kernel[(CO,)](
                grad_output,
                gb,
                CO,
                N,
                HO * WO,
                grad_output.stride(0),
                grad_output.stride(1),
                BLOCK=1 << ((HO * WO - 1).bit_length()),
            )

        # grad_weight / grad_bias are done; grad_input keeps using the clamped
        # scalar kernel (an affine grad_input store needs a >=64-element store
        # lane axis, which these rows do not have: the backend lane-widens
        # narrower stores).
        if mi:
            _sc2b_gi_kernel_launch(
                grad_output, weight, gi, CO, CI, HO, WO, H, W, kh, kw, sh, sw, ph, pw, N
            )
        return (gi, gw, gb)

    if mi:
        _sc2b_gi_kernel_launch(
            grad_output, weight, gi, CO, CI, HO, WO, H, W, kh, kw, sh, sw, ph, pw, N
        )

    if (mw or mb) and not fast:
        # grad_weight and grad_bias share the reduction over (n, ho, wo). When
        # only the bias is requested the weight result goes to a scratch buffer
        # (internal allocation only) so both stay on the single deterministic
        # kernel. grad_weight is contiguous, so SP_CO = CI*kh*kw addresses it
        # exactly as (co, ci, kh, kw).
        CIKK = CI * kh * kw
        gw_target = gw if gw is not None else torch.empty(
            (CO, CI, kh, kw), dtype=dt, device=dev
        )
        if CIKK >= 128:
            BLOCK_M = 32
            BLOCK_N = 64
        else:
            BLOCK_M = 16
            BLOCK_N = 32
        grid_m = triton.cdiv(CO, BLOCK_M)
        grid_n = triton.cdiv(CIKK, BLOCK_N)
        Ktot = N * HO * WO
        # Deterministic split-K: enough CTAs to fill the 64-core device while
        # keeping every slab a non-trivial reduction. Splitting is the dominant
        # win here because the (co, ci*kh*kw) tiling alone leaves only a handful
        # of programs each walking the full (n, ho, wo) axis serially.
        split_k = max(1, -(-256 // (grid_m * grid_n)))
        split_k = min(split_k, max(1, Ktot // 32), 128)
        if split_k > 1:
            part = torch.empty((split_k, CO, CIKK), dtype=torch.float32, device=dev)
            bpart = (
                torch.empty((split_k, CO), dtype=torch.float32, device=dev)
                if mb
                else part
            )
            sp_pid, sp_co, bp_pid = CO * CIKK, CIKK, CO
        else:
            # Single slab writes the final gradient directly.
            part = gw_target
            bpart = gb if gb is not None else gw_target
            sp_pid, sp_co, bp_pid = 0, CIKK, 0
        _sc2b_gw_kernel[(grid_m, grid_n, split_k)](
            grad_output,
            self,
            part,
            bpart,
            CO,
            CI,
            N,
            HO,
            WO,
            H,
            W,
            grad_output.stride(0),
            grad_output.stride(1),
            grad_output.stride(2),
            grad_output.stride(3),
            self.stride(0),
            self.stride(1),
            self.stride(2),
            self.stride(3),
            sp_co,
            sp_pid,
            bp_pid,
            KH=kh,
            KW=kw,
            SH=sh,
            SW=sw,
            PH=ph,
            PW=pw,
            HAS_B=bool(mb),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        if split_k > 1:
            _sc2b_gw_reduce_kernel[(grid_m, grid_n)](
                part,
                gw_target,
                CO,
                CIKK,
                split_k,
                sp_pid,
                sp_co,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                num_warps=4,
            )
            if mb:
                _sc2b_gb_reduce_kernel[(triton.cdiv(CO, 64),)](
                    bpart,
                    gb,
                    CO,
                    split_k,
                    bp_pid,
                    BLOCK_CO=64,
                    num_warps=4,
                )

    return (gi, gw, gb)


# --------------------------------------------------------------------------- #
# Make the *generic* module path resolve to this implementation.
#
# tests/test_slow_conv2d_backward.py and benchmark/test_slow_conv2d_backward.py
# both bind the candidate through the generic module
# (`flag_gems.ops._slow_conv2d_backward`) instead of `flag_gems._slow_conv2d_backward`.
# The generic ``_gw_kernel`` does not lower on XPU3, so that path can never run
# here even though the registrar already routes the top-level name to this
# module. Rebind the generic package attribute and submodule attribute to this
# vendor implementation so both call paths hit the kernel that actually runs on
# XPU (same pattern as ``_kunlunxin/ops/_native_batch_norm_legit.py``).
_generic_module = _sys.modules.get("flag_gems.ops._slow_conv2d_backward")
if _generic_module is not None:
    _generic_module._slow_conv2d_backward = _slow_conv2d_backward
_ops_package = _sys.modules.get("flag_gems.ops")
if _ops_package is not None:
    _ops_package._slow_conv2d_backward = _slow_conv2d_backward
