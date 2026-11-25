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

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Gather kernel (row-based): one program per (n, c, input-row) x BLOCK_W
# columns. Each input element gathers its (up to MAX_BINS_H x MAX_BINS_W)
# covering output bins from grad_output and accumulates grad / (kh * kw).
# Used for non-tile shapes with large element counts.
# ---------------------------------------------------------------------------
@triton.jit
def _aap2d_bwd_kernel(
    grad_output,
    out,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    C: tl.constexpr,
    sgo_b: tl.constexpr,
    sgo_c: tl.constexpr,
    sgo_h: tl.constexpr,
    sgo_w: tl.constexpr,
    so_b: tl.constexpr,
    so_c: tl.constexpr,
    so_h: tl.constexpr,
    so_w: tl.constexpr,
    BLOCK_W: tl.constexpr,
    EXACT_H: tl.constexpr,
    KH: tl.constexpr,
    EXACT_W: tl.constexpr,
    KW: tl.constexpr,
    MAX_BINS_H: tl.constexpr,
    MAX_BINS_W: tl.constexpr,
    OUT_F16: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_F64: tl.constexpr,
    ACC_F64: tl.constexpr,
):
    n = tl.program_id(0)
    pid_row = tl.program_id(1)
    pid_w = tl.program_id(2)

    c = pid_row // H_in
    ih = pid_row % H_in

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_in

    # ---- h dimension: bins covering input row ih are [bh0, bh0 + h_nb) ----
    if EXACT_H:
        bh0 = ih // KH
        wh0 = 1.0 / KH
    else:
        bh0 = (ih * H_out) // H_in
        h_nb = ((ih + 1) * H_out + H_in - 1) // H_in - bh0

    # ---- w dimension: bins covering input col iw are [bw0, bw0 + w_nb) ----
    if EXACT_W:
        bw0 = offs_w // KW
        ww0 = 1.0 / KW
    else:
        bw0 = (offs_w * W_out) // W_in
        w_nb = ((offs_w + 1) * W_out + W_in - 1) // W_in - bw0

    go_base = grad_output + n * sgo_b + c * sgo_c
    out_base = out + n * so_b + c * so_c

    acc = tl.zeros([BLOCK_W], dtype=tl.float64 if ACC_F64 else tl.float32)

    if EXACT_H:
        if EXACT_W:
            g = tl.load(go_base + bh0 * sgo_h + bw0 * sgo_w, mask=mask_w, other=0.0)
            acc = g * (wh0 * ww0)
        else:
            for k in tl.static_range(MAX_BINS_W):
                bw = bw0 + k
                bwc = tl.minimum(bw, W_out - 1)
                kwgt = ((bwc + 1) * W_in + W_out - 1) // W_out - (bwc * W_in) // W_out
                ww = 1.0 / kwgt.to(tl.float32)
                g = tl.load(
                    go_base + bh0 * sgo_h + bw * sgo_w,
                    mask=mask_w & (k < w_nb),
                    other=0.0,
                )
                acc += g * (wh0 * ww)
    else:
        if EXACT_W:
            for k in tl.static_range(MAX_BINS_H):
                bh = bh0 + k
                bhc = tl.minimum(bh, H_out - 1)
                khgt = ((bhc + 1) * H_in + H_out - 1) // H_out - (bhc * H_in) // H_out
                wh = 1.0 / khgt.to(tl.float32)
                g = tl.load(
                    go_base + bh * sgo_h + bw0 * sgo_w,
                    mask=mask_w & (k < h_nb),
                    other=0.0,
                )
                acc += g * (wh * ww0)
        else:
            for kh in tl.static_range(MAX_BINS_H):
                bh = bh0 + kh
                bhc = tl.minimum(bh, H_out - 1)
                khgt = ((bhc + 1) * H_in + H_out - 1) // H_out - (bhc * H_in) // H_out
                wh = 1.0 / khgt.to(tl.float32)
                for kw_ in tl.static_range(MAX_BINS_W):
                    bw = bw0 + kw_
                    bwc = tl.minimum(bw, W_out - 1)
                    kwgt = ((bwc + 1) * W_in + W_out - 1) // W_out - (
                        bwc * W_in
                    ) // W_out
                    ww = 1.0 / kwgt.to(tl.float32)
                    g = tl.load(
                        go_base + bh * sgo_h + bw * sgo_w,
                        mask=mask_w & (kh < h_nb) & (kw_ < w_nb),
                        other=0.0,
                    )
                    acc += g * (wh * ww)

    out_addr = out_base + ih * so_h + offs_w * so_w
    if OUT_F16:
        tl.store(out_addr, acc.to(tl.float16), mask=mask_w)
    elif OUT_BF16:
        tl.store(out_addr, acc.to(tl.bfloat16), mask=mask_w)
    else:
        tl.store(out_addr, acc, mask=mask_w)


# ---------------------------------------------------------------------------
# 2D-block gather kernel: BLOCK_H rows x BLOCK_W columns per program, used for
# small non-tile workloads. Keeps the h-bin math vectorized per row (cheap,
# unlike the flat kernel's per-element decomposition) while cutting CTA count
# and increasing per-CTA work versus the 1-warp row-based kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _aap2d_bwd_2d_kernel(
    grad_output,
    out,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    C: tl.constexpr,
    sgo_b: tl.constexpr,
    sgo_c: tl.constexpr,
    sgo_h: tl.constexpr,
    sgo_w: tl.constexpr,
    so_b: tl.constexpr,
    so_c: tl.constexpr,
    so_h: tl.constexpr,
    so_w: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    MAX_BINS_H: tl.constexpr,
    MAX_BINS_W: tl.constexpr,
    OUT_F16: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_F64: tl.constexpr,
    ACC_F64: tl.constexpr,
):
    n = tl.program_id(0)
    pid_rb = tl.program_id(1)
    pid_w = tl.program_id(2)

    rows = pid_rb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_r = rows < C * H_in
    c = rows // H_in
    ih = rows % H_in

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_in
    m2 = mask_r[:, None] & mask_w[None, :]

    bh0 = (ih * H_out) // H_in
    h_nb = ((ih + 1) * H_out + H_in - 1) // H_in - bh0
    bw0 = (offs_w * W_out) // W_in
    w_nb = ((offs_w + 1) * W_out + W_in - 1) // W_in - bw0

    go_base = grad_output + n * sgo_b + c[:, None] * sgo_c
    out_base = out + n * so_b + c[:, None] * so_c

    acc = tl.zeros([BLOCK_H, BLOCK_W], dtype=tl.float64 if ACC_F64 else tl.float32)
    for kh in tl.static_range(MAX_BINS_H):
        bh = bh0 + kh
        bhc = tl.minimum(bh, H_out - 1)
        khgt = ((bhc + 1) * H_in + H_out - 1) // H_out - (bhc * H_in) // H_out
        wh = 1.0 / khgt.to(tl.float32)
        mh = (kh < h_nb)[:, None]
        for kw_ in tl.static_range(MAX_BINS_W):
            bw = bw0 + kw_
            bwc = tl.minimum(bw, W_out - 1)
            kwgt = ((bwc + 1) * W_in + W_out - 1) // W_out - (bwc * W_in) // W_out
            ww = 1.0 / kwgt.to(tl.float32)
            g = tl.load(
                go_base + bh[:, None] * sgo_h + bw[None, :] * sgo_w,
                mask=m2 & mh & (kw_ < w_nb)[None, :],
                other=0.0,
            )
            acc += g * (wh[:, None] * ww[None, :])

    out_addr = out_base + ih[:, None] * so_h + offs_w[None, :] * so_w
    if OUT_F16:
        tl.store(out_addr, acc.to(tl.float16), mask=m2)
    elif OUT_BF16:
        tl.store(out_addr, acc.to(tl.bfloat16), mask=m2)
    else:
        tl.store(out_addr, acc, mask=m2)


# ---------------------------------------------------------------------------
# Tile-broadcast kernel: used when H_in % H_out == 0 and W_in % W_out == 0 and
# both ratios are powers of two. Then out[ih, iw] = go[ih//KH, iw//KW] / (KH*KW),
# a pure "expand" of the small grad_output tile. Load a small go tile with
# fully affine/coalesced addresses, scale it, and expand it to the output tile
# with broadcast + reshape. No per-lane gathers, no integer bin math.
# For fp16/bf16 outputs the scale is done in fp32 on the tiny tile and the
# result is cast to the output dtype BEFORE the broadcast-expand, so the
# expanded registers and stores are directly in the output dtype (halved
# register pressure, packed stores, no per-element store-time cast). For
# power-of-two KH*KW this is bit-identical to casting at the store.
# ---------------------------------------------------------------------------
@triton.jit
def _aap2d_bwd_exact_kernel(
    grad_output,
    out,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    C: tl.constexpr,
    NHT: tl.constexpr,
    sgo_b: tl.constexpr,
    sgo_c: tl.constexpr,
    sgo_h: tl.constexpr,
    sgo_w: tl.constexpr,
    so_b: tl.constexpr,
    so_c: tl.constexpr,
    so_h: tl.constexpr,
    so_w: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    OUT_F16: tl.constexpr,
    OUT_BF16: tl.constexpr,
    OUT_F64: tl.constexpr,
):
    n = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)
    c = pid_h // NHT
    ht = pid_h % NHT

    GH: tl.constexpr = BLOCK_H // KH
    GW: tl.constexpr = BLOCK_W // KW

    gh = ht * GH + tl.arange(0, GH)
    gw = pid_w * GW + tl.arange(0, GW)
    go_base = grad_output + n * sgo_b + c * sgo_c
    go_tile = tl.load(
        go_base + gh[:, None] * sgo_h + gw[None, :] * sgo_w,
        mask=(gh[:, None] < H_out) & (gw[None, :] < W_out),
        other=0.0,
    )
    if OUT_F64:
        go_tile = go_tile * (1.0 / (KH * KW))
    elif OUT_F16:
        go_tile = (go_tile.to(tl.float32) * (1.0 / (KH * KW))).to(tl.float16)
    elif OUT_BF16:
        go_tile = (go_tile.to(tl.float32) * (1.0 / (KH * KW))).to(tl.bfloat16)
    else:
        go_tile = go_tile.to(tl.float32) * (1.0 / (KH * KW))

    exp = tl.broadcast_to(go_tile[:, None, :, None], (GH, KH, GW, KW))
    exp = tl.reshape(exp, (BLOCK_H, BLOCK_W))

    rows = ht * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_addr = out + n * so_b + c * so_c + rows[:, None] * so_h + cols[None, :] * so_w
    mask_o = (rows[:, None] < H_in) & (cols[None, :] < W_in)
    tl.store(out_addr, exp, mask=mask_o)


def _is_pow2(x):
    return (x & (x - 1)) == 0


def _adaptive_avg_pool2d_backward(grad_output, self):
    dim = self.dim()
    if dim == 4:
        N, C, H_in, W_in = self.shape
        H_out, W_out = grad_output.shape[2], grad_output.shape[3]
        sgo_b, sgo_c = grad_output.stride(0), grad_output.stride(1)
        sgo_h, sgo_w = grad_output.stride(2), grad_output.stride(3)
        so_b, so_c = self.stride(0), self.stride(1)
        so_h, so_w = self.stride(2), self.stride(3)
    elif dim == 3:
        N, C, H_in, W_in = 1, self.shape[0], self.shape[1], self.shape[2]
        H_out, W_out = grad_output.shape[1], grad_output.shape[2]
        sgo_b, sgo_c = 0, grad_output.stride(0)
        sgo_h, sgo_w = grad_output.stride(1), grad_output.stride(2)
        so_b, so_c = 0, self.stride(0)
        so_h, so_w = self.stride(1), self.stride(2)
    elif dim == 2:
        N, C, H_in, W_in = 1, 1, self.shape[0], self.shape[1]
        H_out, W_out = grad_output.shape[0], grad_output.shape[1]
        sgo_b, sgo_c, sgo_h, sgo_w = 0, 0, grad_output.stride(0), grad_output.stride(1)
        so_b, so_c, so_h, so_w = 0, 0, self.stride(0), self.stride(1)
    else:
        raise ValueError("adaptive_avg_pool2d_backward expects 2D/3D/4D input")

    out = torch.empty_like(self)

    EXACT_H = H_in % H_out == 0
    EXACT_W = W_in % W_out == 0
    KH = H_in // H_out if EXACT_H else 1
    KW = W_in // W_out if EXACT_W else 1

    out_f16 = self.dtype == torch.float16
    out_bf16 = self.dtype == torch.bfloat16
    out_f64 = self.dtype == torch.float64

    use_tile = (
        EXACT_H
        and EXACT_W
        and _is_pow2(KH)
        and _is_pow2(KW)
        and KH <= 64
        and KW <= 128
        and 1 <= W_in
        and 1 <= H_in
    )

    if use_tile:
        kh_pow2 = 1 << (KH - 1).bit_length()
        small_h = H_in <= 64 and H_in % 8 == 0 and kh_pow2 <= 8
        if small_h and (out_f16 or out_bf16) and H_in >= 32:
            BH = 32
            BW = 128
            num_warps = 8
        else:
            if small_h:
                BH = 8
            else:
                BH = max(16, kh_pow2)
            BW = max(min(triton.next_power_of_2(W_in), 128), 1 << (KW - 1).bit_length())
            if out_f16 or out_bf16:
                num_warps = 1 if (BH * BW) <= 512 else min(8, max(1, (BH * BW) // 128))
            else:
                num_warps = 1
        NHT = triton.cdiv(H_in, BH)
        grid = (N, C * NHT, triton.cdiv(W_in, BW))
        _aap2d_bwd_exact_kernel[grid](
            grad_output,
            out,
            H_in,
            W_in,
            H_out,
            W_out,
            C,
            NHT,
            sgo_b,
            sgo_c,
            sgo_h,
            sgo_w,
            so_b,
            so_c,
            so_h,
            so_w,
            KH=KH,
            KW=KW,
            BLOCK_H=BH,
            BLOCK_W=BW,
            OUT_F16=out_f16,
            OUT_BF16=out_bf16,
            OUT_F64=out_f64,
            num_warps=num_warps,
        )
    else:
        total = N * C * H_in * W_in
        if total <= (1 << 20):
            BLOCK_H = 8
            BLOCK_W = 32
            nw = 4
            MAX_BINS_H = (H_out + H_in - 1) // H_in + 1
            MAX_BINS_W = (W_out + W_in - 1) // W_in + 1
            grid = (
                N,
                (C * H_in + BLOCK_H - 1) // BLOCK_H,
                (W_in + BLOCK_W - 1) // BLOCK_W,
            )
            _aap2d_bwd_2d_kernel[grid](
                grad_output,
                out,
                H_in,
                W_in,
                H_out,
                W_out,
                C,
                sgo_b,
                sgo_c,
                sgo_h,
                sgo_w,
                so_b,
                so_c,
                so_h,
                so_w,
                BLOCK_H=BLOCK_H,
                BLOCK_W=BLOCK_W,
                MAX_BINS_H=MAX_BINS_H,
                MAX_BINS_W=MAX_BINS_W,
                OUT_F16=out_f16,
                OUT_BF16=out_bf16,
                OUT_F64=out_f64,
                ACC_F64=out_f64,
                num_warps=nw,
            )
        else:
            BLOCK_W = max(32, min(1024, triton.next_power_of_2(W_in)))
            num_warps = max(1, min(8, BLOCK_W // 64))
            MAX_BINS_H = (H_out + H_in - 1) // H_in + 1
            MAX_BINS_W = (W_out + W_in - 1) // W_in + 1
            grid = (N, C * H_in, triton.cdiv(W_in, BLOCK_W))
            _aap2d_bwd_kernel[grid](
                grad_output,
                out,
                H_in,
                W_in,
                H_out,
                W_out,
                C,
                sgo_b,
                sgo_c,
                sgo_h,
                sgo_w,
                so_b,
                so_c,
                so_h,
                so_w,
                BLOCK_W=BLOCK_W,
                EXACT_H=EXACT_H,
                KH=KH,
                EXACT_W=EXACT_W,
                KW=KW,
                MAX_BINS_H=MAX_BINS_H,
                MAX_BINS_W=MAX_BINS_W,
                OUT_F16=out_f16,
                OUT_BF16=out_bf16,
                OUT_F64=out_f64,
                ACC_F64=out_f64,
                num_warps=num_warps,
            )
    return out
