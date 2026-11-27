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

"""Triton implementation of ``aten::slow_conv_transpose3d`` (forward).

Semantics (im2col / col2im form):

  out[n, co, od, oh, ow] = bias[co]
      + sum_{ci, kd, kh, kw} self[n, ci, id, ih, iw] * weight[ci, co, kd, kh, kw]

  id = od + pad_d - kd*dil_d, valid iff id % stride_d == 0 and 0 <= id/stride_d < D_in
  (and likewise for h / w).

``weight`` uses the transposed-conv layout (C_in, C_out, kD, kH, kW) and no
kernel flipping is applied.  The implementation is a deterministic
output-stationary tiled gather: every program computes a (BLOCK_CO, BLOCK_SP)
tile of the output plane by accumulating an implicit GEMM whose K axis is
C_in*kD*kH*kW.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

__all__ = ["slow_conv_transpose3d"]


def _as_triple(value, name, default):
    if value is None:
        value = default
    if torch.is_tensor(value):
        if value.dim() == 0:
            value = [int(value.item())]
        else:
            value = [int(v) for v in value.tolist()]
    elif isinstance(value, (int, float)):
        value = [int(value)]
    else:
        value = [int(v) for v in value]
    if len(value) == 1:
        value = value * 3
    if len(value) != 3:
        raise RuntimeError(
            "%s must be a single integer or a sequence of 3 integers" % name
        )
    return value


def _cdiv(a, b):
    return -(-a // b)


def _np2_at_least(v, lo):
    p = lo
    while p < v:
        p *= 2
    return p


@triton.jit
def _sct3d_fwd_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    C_in,
    C_out,
    S,
    W_out,
    HW,
    D_in,
    H_in,
    W_in,
    sC,
    sD,
    sH,
    sW,
    sN,
    wCi,
    wCo,
    wKd,
    wKh,
    wKw,
    oC,
    oD,
    oH,
    oW,
    oN,
    NSP,
    kD: tl.constexpr,
    kH: tl.constexpr,
    kW: tl.constexpr,
    stD,
    stH,
    stW,
    paD,
    paH,
    paW,
    diD,
    diH,
    diW,
    HAS_BIAS: tl.constexpr,
    DOT: tl.constexpr,
    ST1: tl.constexpr,
    F64: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_CO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
):
    pid0 = tl.program_id(0)
    pid1 = tl.program_id(1)

    n = pid0 // NSP
    sp0 = (pid0 % NSP) * BLOCK_SP
    sp = sp0 + tl.arange(0, BLOCK_SP)
    spm = sp < S

    od = sp // HW
    rem = sp - od * HW
    oh = rem // W_out
    ow = rem - oh * W_out

    co = pid1 * BLOCK_CO + tl.arange(0, BLOCK_CO)
    com = co < C_out

    x_n = x_ptr + n * sN

    if F64:
        acc = tl.zeros((BLOCK_CO, BLOCK_SP), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_CO, BLOCK_SP), dtype=tl.float32)

    if DOT:
        ci = tl.arange(0, BLOCK_CI)
        for kd in tl.static_range(kD):
            idr = od + paD - kd * diD
            if ST1:
                vd = (idr >= 0) & (idr < D_in)
                id_ = idr
            else:
                vd = (idr >= 0) & (idr < D_in * stD) & (idr % stD == 0)
                id_ = idr // stD
            for kh in tl.static_range(kH):
                ihr = oh + paH - kh * diH
                if ST1:
                    vh = (ihr >= 0) & (ihr < H_in)
                    ih_ = ihr
                else:
                    vh = (ihr >= 0) & (ihr < H_in * stH) & (ihr % stH == 0)
                    ih_ = ihr // stH
                vdh = vd & vh & spm
                off_dh = id_ * sD + ih_ * sH
                for kw in tl.static_range(kW):
                    iwr = ow + paW - kw * diW
                    if ST1:
                        vw = (iwr >= 0) & (iwr < W_in)
                        iw_ = iwr
                    else:
                        vw = (iwr >= 0) & (iwr < W_in * stW) & (iwr % stW == 0)
                        iw_ = iwr // stW
                    m = vdh & vw
                    xoff = x_n + off_dh + iw_ * sW
                    woff = w_ptr + kd * wKd + kh * wKh + kw * wKw
                    for c0 in range(0, C_in, BLOCK_CI):
                        cci = c0 + ci
                        ok = cci < C_in
                        xt = tl.load(
                            xoff[None, :] + (cci * sC)[:, None],
                            mask=m[None, :] & ok[:, None],
                            other=0.0,
                        )
                        wt = tl.load(
                            woff + (co * wCo)[:, None] + (cci * wCi)[None, :],
                            mask=com[:, None] & ok[None, :],
                            other=0.0,
                        )
                        acc = tl.dot(wt, xt, acc, input_precision="ieee")
    else:
        for kd in tl.static_range(kD):
            idr = od + paD - kd * diD
            if ST1:
                vd = (idr >= 0) & (idr < D_in)
                id_ = idr
            else:
                vd = (idr >= 0) & (idr < D_in * stD) & (idr % stD == 0)
                id_ = idr // stD
            for kh in tl.static_range(kH):
                ihr = oh + paH - kh * diH
                if ST1:
                    vh = (ihr >= 0) & (ihr < H_in)
                    ih_ = ihr
                else:
                    vh = (ihr >= 0) & (ihr < H_in * stH) & (ihr % stH == 0)
                    ih_ = ihr // stH
                vdh = vd & vh & spm
                off_dh = id_ * sD + ih_ * sH
                for kw in tl.static_range(kW):
                    iwr = ow + paW - kw * diW
                    if ST1:
                        vw = (iwr >= 0) & (iwr < W_in)
                        iw_ = iwr
                    else:
                        vw = (iwr >= 0) & (iwr < W_in * stW) & (iwr % stW == 0)
                        iw_ = iwr // stW
                    m = vdh & vw
                    xoff = x_n + off_dh + iw_ * sW
                    woff = w_ptr + kd * wKd + kh * wKh + kw * wKw
                    for c in range(0, C_in):
                        xr = tl.load(xoff + c * sC, mask=m, other=0.0)
                        wr = tl.load(woff + c * wCi + co * wCo, mask=com, other=0.0)
                        acc = acc + wr[:, None] * xr[None, :]

    if HAS_BIAS:
        bv = tl.load(b_ptr + co, mask=com, other=0.0)
        acc = acc + bv[:, None]

    ooff = o_ptr + n * oN + (co * oC)[:, None] + (od * oD + oh * oH + ow * oW)[None, :]
    tl.store(ooff, acc.to(o_ptr.dtype.element_ty), mask=com[:, None] & spm[None, :])


# ---------------------------------------------------------------------------
# forward: residue-phase implicit gemm (GCU400 fast path)
# ---------------------------------------------------------------------------
# The gather-style kernel above lowers poorly on this backend: it evaluates a
# per-element divisibility mask and runtime div/mod indices for every one of the
# kD*kH*kW taps, and fp32 falls back to a scalar FMA loop over C_in.  This
# variant reuses the residue-phase structure proven on the conv_transpose3d /
# slow_conv_transpose2d siblings: output positions are partitioned by their
# (od % stD, oh % stH, ow % stW) phase so a program only visits the taps whose
# back-projection lands on an input sample, and the input index stays affine in
# the compact output index (the per-tap scalar offset folds the division).
#
# Compile-time note: every shape/stride/padding/dilation scalar is a *runtime*
# argument (`do_not_specialize`), the tap loop is a runtime loop with one
# `tl.dot` in its body, and the bias/dtype switches are runtime values too.
# The earlier variant made all of them constexpr and unrolled the taps into
# kD*kH*kW statically nested `if`/dot regions; each new test shape then paid a
# fresh 70-110 s backend compile (55 accuracy cases plus the benchmark shapes),
# which dwarfed the actual kernel runtime.  With the generic form the whole
# suite maps onto one instantiation per (element dtype, tile) combination, and
# the taps that cannot contribute to a subgrid simply fall out through the
# `tap_live` predicate.
#
# fp16/bf16 run the dot in fp32 after an in-register upcast (raw low-precision
# dots are intermittently wrong on this backend), fp32 runs the same code path
# because the upcast is a no-op there; both accumulate in fp32 and round once at
# the store.
_DOT_PRECISION = tl.constexpr("ieee")

# Scalars that only affect addressing/loop bounds or select a predicated load,
# never the generated tile shape: excluding them from specialization keeps one
# compiled binary valid for every shape, stride, padding and bias combination.
_SHAPE_ARGS = [
    "N",
    "C_in",
    "D_in",
    "H_in",
    "W_in",
    "C_out",
    "D_out",
    "H_out",
    "W_out",
    "kD",
    "kH",
    "kW",
    "stD",
    "stH",
    "stW",
    "paD",
    "paH",
    "paW",
    "diD",
    "diH",
    "diW",
    "compact_D",
    "compact_H",
    "compact_W",
    "compact_total",
    "b_stride",
    "has_bias",
]


@libentry()
@triton.jit(do_not_specialize=_SHAPE_ARGS)
def _sct3d_gemm(
    x_ptr,
    w_ptr,
    b_ptr,
    o_ptr,
    N,
    C_in,
    D_in,
    H_in,
    W_in,
    C_out,
    D_out,
    H_out,
    W_out,
    kD,
    kH,
    kW,
    stD,
    stH,
    stW,
    paD,
    paH,
    paW,
    diD,
    diH,
    diW,
    compact_D,
    compact_H,
    compact_W,
    compact_total,
    b_stride,
    has_bias,
    BLOCK_N: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """One subgrid of the transposed conv as a (spatial, C_out) GEMM tile.

    Program ``(pid_m, pid_co, pid_sub)`` owns ``BLOCK_N`` output positions of
    the ``pid_sub``-th stride residue class and ``BLOCK_CO`` output channels.
    The GEMM K axis is (C_in, kD, kH, kW); the tap part is a runtime loop so the
    IR (and therefore the backend compile time) does not grow with the kernel
    volume.
    """
    pid_m = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_sub = tl.program_id(2)

    residue_w = pid_sub % stW
    residue_h = (pid_sub // stW) % stH
    residue_d = pid_sub // (stW * stH)

    idx = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)
    in_range = idx < compact_total
    idx_w = idx % compact_W
    idx_dh = idx // compact_W
    idx_h = idx_dh % compact_H
    idx_s = idx_dh // compact_H
    idx_d = idx_s % compact_D
    n = idx_s // compact_D

    od = idx_d * stD + residue_d
    oh = idx_h * stH + residue_h
    ow = idx_w * stW + residue_w
    out_valid = in_range & (od < D_out) & (oh < H_out) & (ow < W_out)

    co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    valid_co = co < C_out

    acc = tl.zeros((BLOCK_N, BLOCK_CO), dtype=tl.float32)
    # `has_bias` is a runtime flag so the biased and unbiased calls share one
    # compiled binary; without a bias the load stays fully masked and adds 0.
    bv = tl.load(
        b_ptr + co * b_stride,
        mask=valid_co & (has_bias != 0),
        other=0.0,
    ).to(tl.float32)
    acc = acc + bv[None, :]

    tap_count = kD * kH * kW
    k_hw = kH * kW
    depth_phase = (residue_d + paD) % stD
    height_phase = (residue_h + paH) % stH
    width_phase = (residue_w + paW) % stW

    x_plane = D_in * H_in * W_in
    w_ci_stride = C_out * tap_count

    # A tap contributes to this subgrid iff (k * dilation) % stride equals the
    # (residue + padding) phase; the per-tap offsets below fold the division by
    # the stride, so the inner addressing is pure affine arithmetic.
    for tap in range(tap_count):
        kd = tap // k_hw
        kh = (tap - kd * k_hw) // kW
        kw = tap - kd * k_hw - kh * kW
        tap_live = (
            ((kd * diD) % stD == depth_phase)
            & ((kh * diH) % stH == height_phase)
            & ((kw * diW) % stW == width_phase)
        )
        id_ = idx_d + (residue_d + paD - kd * diD) // stD
        ih_ = idx_h + (residue_h + paH - kh * diH) // stH
        iw_ = idx_w + (residue_w + paW - kw * diW) // stW
        tap_valid = (
            tap_live
            & (id_ >= 0)
            & (id_ < D_in)
            & (ih_ >= 0)
            & (ih_ < H_in)
            & (iw_ >= 0)
            & (iw_ < W_in)
            & out_valid
        )
        x_lane = n * (C_in * x_plane) + id_ * (H_in * W_in) + ih_ * W_in + iw_
        w_row = w_ptr + co * tap_count + tap
        for ci_base in range(0, C_in, BLOCK_CI):
            cci = ci_base + tl.arange(0, BLOCK_CI)
            ci_ok = cci < C_in
            xv = tl.load(
                x_ptr + x_lane[:, None] + (cci * x_plane)[None, :],
                mask=tap_valid[:, None] & ci_ok[None, :],
                other=0.0,
            )
            wv = tl.load(
                w_row[None, :] + (cci * w_ci_stride)[:, None],
                mask=ci_ok[:, None] & valid_co[None, :],
                other=0.0,
            )
            # Unconditional fp32 upcast: it keeps the fp16/bf16 tensor-core
            # inputs in fp32 (raw low-precision dots are intermittently wrong on
            # this backend) and is a no-op for fp32 data, which lets one binary
            # serve every float dtype's dot path.
            acc = acc + tl.dot(
                xv.to(tl.float32),
                wv.to(tl.float32),
                input_precision=_DOT_PRECISION,
            )

    ooff = (
        (n[:, None] * C_out + co[None, :]) * (D_out * H_out * W_out)
        + od[:, None] * (H_out * W_out)
        + oh[:, None] * W_out
        + ow[:, None]
    )
    tl.store(
        o_ptr + ooff,
        acc.to(o_ptr.dtype.element_ty),
        mask=out_valid[:, None] & valid_co[None, :],
    )


def _residue_tiles(dtype, c_in, c_out, compact_total, k_d, k_h, k_w, st_d, st_h, st_w):
    """Pick (BLOCK_N, BLOCK_CI, BLOCK_CO, num_warps, num_stages).

    Swept over the benchmark shape matrix on GCU400 (BLOCK_N x num_warps x
    stages grid, then CI/CO, then a BLOCK_N=256 follow-up; every cell passed
    the CPU-fp64 accuracy gate first):
      * BLOCK_N dominates: 128 is best on the strided shapes whatever the
        tap count; stride-1 shapes carry all kD*kH*kW taps per program and
        want a fat 256-row tile once the volume fills half a wave, but a
        narrow 32-row tile when the whole grid would fall under one wave.
      * BLOCK_CI/BLOCK_CO: tl.dot's 16 minimum; 32 once the channel count
        reaches it (case C_out=32 measured 24% faster at BLOCK_CO=32).
      * num_warps/num_stages move latency by <1% everywhere; keep the
        sibling-proven num_warps=1/num_stages=2.

    Only these tile sizes are constexpr (bias presence and the float dtype
    only select masked loads / pointer types), so the whole test matrix reuses
    a handful of compiled binaries instead of one per shape.
    """
    live_taps = max(1, (k_d * k_h * k_w) // (st_d * st_h * st_w))
    if live_taps >= 8:
        block_n = (
            256 if compact_total >= 4096 else (32 if compact_total < 1024 else 128)
        )
    else:
        block_n = 128
    block_ci = 16 if c_in <= 16 else 32
    block_co = 16 if c_out <= 16 else 32
    return block_n, block_ci, block_co, 1, 2


def _pow2_leq(value, lo, hi):
    """Largest power of two in [lo, hi] that does not exceed value (or lo)."""
    p = lo
    best = lo
    while p <= value and p <= hi:
        best = p
        p *= 2
    return best


def _pick_blocks(S, C_out, N, min_programs=200):
    """Choose ``(BLOCK_SP, BLOCK_CO)`` for the flat output-plane decomposition.

    A wider channel tile means fewer redundant passes over the gathered
    activation plane, so BLOCK_CO is taken as large as the register budget and
    the dtype's dot shape allow.  BLOCK_SP is then the largest spatial tile that
    still leaves at least one full 78-SM device wave of programs; when no tiling
    reaches that floor, the configuration with the most programs is used so the
    machine is at least partly occupied.  The 1x1x1 GEMM-like case is included
    only by the 16-wide spatial tile, which is the finest the masks allow.
    """
    bsp_max = _np2_at_least(max(S, 1), 16)
    bco_max = _pow2_leq(C_out, 16, 32)
    for bsp in (64, 32, 16):
        if bsp > bsp_max:
            continue
        if N * _cdiv(S, bsp) * _cdiv(C_out, bco_max) >= min_programs:
            return bsp, bco_max

    best = None
    for bsp in (16, 32, 64):
        if bsp > bsp_max:
            continue
        for bco in (16, 32, 64):
            if bco > _np2_at_least(max(C_out, 1), 16):
                continue
            progs = N * _cdiv(S, bsp) * _cdiv(C_out, bco)
            key = (progs, bco, bsp)
            if best is None or key > best[0]:
                best = (key, bsp, bco)
    if best is None:
        return 16, 16
    return best[1], best[2]


def slow_conv_transpose3d(
    self,
    weight,
    kernel_size=None,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    out=None,
):
    logger.debug("GEMS_ENFLAME SLOW_CONV_TRANSPOSE3D")
    inp = self
    if not torch.is_tensor(inp) or not torch.is_tensor(weight):
        raise TypeError("slow_conv_transpose3d expects tensor inputs")

    st = _as_triple(stride, "stride", 1)
    pa = _as_triple(padding, "padding", 0)
    op = _as_triple(output_padding, "output_padding", 0)
    dl = _as_triple(dilation, "dilation", 1)

    if inp.dim() not in (4, 5):
        raise RuntimeError(
            "slow_conv_transpose3d: expected 4D or 5D input, got %dD" % inp.dim()
        )
    if weight.dim() != 5:
        raise RuntimeError(
            "slow_conv_transpose3d: expected 5D weight, got %dD" % weight.dim()
        )
    if not inp.is_floating_point() or not weight.is_floating_point():
        raise RuntimeError(
            "slow_conv_transpose3d not implemented for non-floating point dtypes"
        )
    if any(s <= 0 for s in st):
        raise RuntimeError("slow_conv_transpose3d: stride must be greater than zero")
    if any(d <= 0 for d in dl):
        raise RuntimeError("slow_conv_transpose3d: dilation must be greater than zero")
    if any(p < 0 for p in pa):
        raise RuntimeError("slow_conv_transpose3d: padding must be non-negative")
    for i in range(3):
        if op[i] >= max(st[i], dl[i]):
            raise RuntimeError(
                "slow_conv_transpose3d: output padding must be smaller than either "
                "stride or dilation"
            )

    unbatched = inp.dim() == 4
    x = inp.unsqueeze(0) if unbatched else inp

    N, C_in, D_in, H_in, W_in = x.shape
    if weight.shape[0] != C_in:
        raise RuntimeError(
            "slow_conv_transpose3d: weight.shape[0] (%d) must match input channels (%d)"
            % (weight.shape[0], C_in)
        )
    kD, kH, kW = weight.shape[2], weight.shape[3], weight.shape[4]
    if kernel_size is not None:
        ks = _as_triple(kernel_size, "kernel_size", 1)
        if tuple(ks) != (kD, kH, kW):
            raise RuntimeError(
                "slow_conv_transpose3d: kernel_size %s does not match weight spatial "
                "dims %s" % (tuple(ks), (kD, kH, kW))
            )
    C_out = weight.shape[1]
    if bias is not None:
        if not torch.is_tensor(bias):
            raise TypeError("slow_conv_transpose3d: bias must be a Tensor or None")
        if bias.numel() != C_out:
            raise RuntimeError(
                "slow_conv_transpose3d: bias size (%d) must match C_out (%d)"
                % (bias.numel(), C_out)
            )
        # Both kernels index the bias by channel; a strided bias (e.g. a
        # `[::2]` view) would otherwise be read in storage order, which puts
        # the wrong constant on every channel plane but the first.
        if not bias.is_contiguous():
            bias = bias.contiguous()

    D_out = (D_in - 1) * st[0] - 2 * pa[0] + dl[0] * (kD - 1) + op[0] + 1
    H_out = (H_in - 1) * st[1] - 2 * pa[1] + dl[1] * (kH - 1) + op[1] + 1
    W_out = (W_in - 1) * st[2] - 2 * pa[2] + dl[2] * (kW - 1) + op[2] + 1

    shape = (N, C_out, D_out, H_out, W_out)
    if out is None:
        out_t = torch.empty(shape, device=x.device, dtype=x.dtype)
    else:
        if not torch.is_tensor(out):
            raise TypeError("slow_conv_transpose3d: out must be a Tensor")
        if tuple(out.shape) != shape:
            raise RuntimeError(
                "slow_conv_transpose3d: out shape %s does not match expected %s"
                % (tuple(out.shape), shape)
            )
        out_t = out

    S = D_out * H_out * W_out
    if out_t.numel() == 0 or S == 0:
        return out_t.squeeze(0) if unbatched else out_t

    dtype = x.dtype
    if dtype is torch.float64:
        f64 = True
    elif dtype in (torch.float32, torch.float16, torch.bfloat16):
        f64 = False
    else:
        raise RuntimeError("slow_conv_transpose3d: unsupported dtype %s" % str(dtype))

    # GCU400 fast path: the gather/FMA kernel above needs runtime div/mod
    # indices and per-element divisibility masks on every tap; the
    # residue-phase implicit GEMM is an order of magnitude faster on every
    # benchmark shape.  Route every case that keeps all index math inside
    # int32 to it.  Non-contiguous inputs are cheap to normalise here (the
    # kernel is fully affine, so strides are only a host-side detail), and the
    # bias stride is passed so a viewed bias needs no copy either.
    int32_safe = max(out_t.numel(), x.numel(), weight.numel()) < (1 << 31) - 65536
    if not f64 and int32_safe and out_t.is_contiguous():
        x_fast = x if x.is_contiguous() else x.contiguous()
        w_fast = weight if weight.is_contiguous() else weight.contiguous()
        b_stride = bias.stride(0) if (bias is not None and bias.dim() > 0) else 0

        n_subgrids = st[0] * st[1] * st[2]
        compact_d = (D_out + st[0] - 1) // st[0]
        compact_h = (H_out + st[1] - 1) // st[1]
        compact_w = (W_out + st[2] - 1) // st[2]
        compact_total = N * compact_d * compact_h * compact_w
        block_n, block_ci, block_co, warps, stages = _residue_tiles(
            dtype,
            weight.shape[0],
            C_out,
            compact_total,
            kD,
            kH,
            kW,
            st[0],
            st[1],
            st[2],
        )
        grid = (
            triton.cdiv(compact_total, block_n),
            triton.cdiv(C_out, block_co),
            n_subgrids,
        )
        _sct3d_gemm[grid](
            x_fast,
            w_fast,
            bias if bias is not None else x_fast,
            out_t,
            N,
            weight.shape[0],
            D_in,
            H_in,
            W_in,
            C_out,
            D_out,
            H_out,
            W_out,
            kD,
            kH,
            kW,
            st[0],
            st[1],
            st[2],
            pa[0],
            pa[1],
            pa[2],
            dl[0],
            dl[1],
            dl[2],
            compact_d,
            compact_h,
            compact_w,
            compact_total,
            b_stride,
            1 if bias is not None else 0,
            BLOCK_N=block_n,
            BLOCK_CI=block_ci,
            BLOCK_CO=block_co,
            num_warps=warps,
            num_stages=stages,
        )
        return out_t.squeeze(0) if unbatched else out_t

    # Fallback only: float64 (a fp64 `tl.dot`/vector multiply does not lower on
    # this backend), oversized tensors whose index math leaves int32, and a
    # non-contiguous `out`.  This path holds the whole (BLOCK_CO, BLOCK_SP)
    # accumulator in registers and walks C_in serially, so it stays fully in
    # registers; the low-precision dtypes keep using tensor cores here.
    dot = dtype in (torch.float16, torch.bfloat16)

    BLOCK_SP, BLOCK_CO = _pick_blocks(S, C_out, N)
    BLOCK_CI = min(_np2_at_least(max(C_in, 1), 16), 64)

    NSP = _cdiv(S, BLOCK_SP)
    grid = (N * NSP, _cdiv(C_out, BLOCK_CO))

    # The scalar outer-product path holds the whole (BLOCK_CO, BLOCK_SP)
    # accumulator in registers and walks C_in serially, so its throughput comes
    # from per-thread instruction-level parallelism rather than from covered
    # latency.  Once the tile is large, giving the whole tile to one warp beats
    # spreading it thin: measured 1.6-2x on the fp32 shapes with C_in <= 16 and
    # a >= 1024-element tile.
    num_warps = 1 if (not dot) and C_in <= 16 and BLOCK_SP * BLOCK_CO >= 1024 else 4

    _sct3d_fwd_kernel[grid](
        x,
        weight,
        bias if bias is not None else x,
        out_t,
        C_in,
        C_out,
        S,
        W_out,
        H_out * W_out,
        D_in,
        H_in,
        W_in,
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        x.stride(0),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight.stride(3),
        weight.stride(4),
        out_t.stride(1),
        out_t.stride(2),
        out_t.stride(3),
        out_t.stride(4),
        out_t.stride(0),
        NSP,
        kD,
        kH,
        kW,
        st[0],
        st[1],
        st[2],
        pa[0],
        pa[1],
        pa[2],
        dl[0],
        dl[1],
        dl[2],
        HAS_BIAS=bias is not None,
        DOT=dot,
        ST1=(st[0] == 1 and st[1] == 1 and st[2] == 1),
        F64=f64,
        BLOCK_SP=BLOCK_SP,
        BLOCK_CO=BLOCK_CO,
        BLOCK_CI=BLOCK_CI,
        num_warps=num_warps,
    )

    return out_t.squeeze(0) if unbatched else out_t
