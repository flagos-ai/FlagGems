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

from flag_gems.ops._adaptive_avg_pool2d_backward import (
    _adaptive_avg_pool2d_backward as default__adaptive_avg_pool2d_backward,
)

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


def _max_cover(in_size, out_size):
    """Max number of output bins covering any single input index (per axis)."""
    m = 0
    for i in range(in_size):
        c = ((i + 1) * out_size + in_size - 1) // in_size - (i * out_size) // in_size
        if c > m:
            m = c
    return m


@triton.jit
def _aap2d_bwd_div(
    g_ptr,
    out_ptr,
    plane_g,
    plane_o,
    HBLK,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    RH: tl.constexpr,
    RW: tl.constexpr,
    IS_FP64: tl.constexpr,
    BLOCK_W: tl.constexpr,
    R: tl.constexpr,
):
    # Divisible adaptive pooling backward (H_out | H_in, W_out | W_in):
    # out[nc, h, w] = grad_output[nc, h // RH, w // RW] / (RH * RW)
    # 2D tile of R rows x BLOCK_W columns per program: per-element work is one
    # constexpr shift for w0 plus a gather load and a store.
    pid_w = tl.program_id(0)
    pid_r = tl.program_id(1)
    nc = pid_r // HBLK
    hb = pid_r % HBLK
    r = tl.arange(0, R)[:, None]
    w = tl.arange(0, BLOCK_W)[None, :]
    h_in = hb * R + r
    w_in = pid_w * BLOCK_W + w
    m = (h_in < H_in) & (w_in < W_in)
    h0 = h_in // RH
    w0 = w_in // RW
    facc = tl.float64 if IS_FP64 else tl.float32
    scl = tl.full([1, 1], 1.0 / float(RH * RW), dtype=facc)
    gv = tl.load(g_ptr + nc * plane_g + h0 * W_out + w0, mask=m, other=0.0)
    res = gv.to(facc) * scl
    tl.store(
        out_ptr + nc * plane_o + h_in * W_in + w_in,
        res.to(out_ptr.dtype.element_ty),
        mask=m,
    )


@triton.jit
def _aap2d_bwd_gen(
    g_ptr,
    out_ptr,
    plane_g,
    plane_o,
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    IS_FP64: tl.constexpr,
    KMAX_H: tl.constexpr,
    KMAX_W: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # General adaptive pooling backward (any ratio): each program handles one
    # input row; h-coordinates and window sizes are scalars per program, so the
    # per-element ALU is limited to w0/w1 and the win_w gather math. Shapes are
    # constexpr so every integer division becomes a compile-time magic multiply.
    pid_w = tl.program_id(0)
    pid_r = tl.program_id(1)
    nc = pid_r // H_in
    h_in = pid_r % H_in
    h0 = (h_in * H_out) // H_in
    h1 = ((h_in + 1) * H_out + H_in - 1) // H_in
    w_in = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mw = w_in < W_in
    w0 = (w_in * W_out) // W_in
    w1 = ((w_in + 1) * W_out + W_in - 1) // W_in
    g_row = g_ptr + nc * plane_g
    facc = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros([BLOCK_W], dtype=facc)
    for i in tl.static_range(KMAX_H):
        ho = h0 + i
        vh = ho < h1
        sh = (ho * H_in) // H_out
        eh = ((ho + 1) * H_in + H_out - 1) // H_out
        winh = eh - sh
        for j in tl.static_range(KMAX_W):
            wo = w0 + j
            valid = vh & (wo < w1) & mw
            sw = (wo * W_in) // W_out
            ew = ((wo + 1) * W_in + W_out - 1) // W_out
            winw = ew - sw
            gv = tl.load(g_row + ho * W_out + wo, mask=valid, other=0.0)
            acc += gv.to(facc) / (winh * winw).to(facc)
    tl.store(
        out_ptr + nc * plane_o + h_in * W_in + w_in,
        acc.to(out_ptr.dtype.element_ty),
        mask=mw,
    )


@triton.jit
def _aap2d_bwd_fused(
    g_ptr,
    out_ptr,
    numel,
    plane_g,
    plane_o,
    H_in,
    W_in,
    H_out,
    W_out,
    IS_FP64: tl.constexpr,
    KMAX_H: tl.constexpr,
    KMAX_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Flat fallback for grids that exceed row-block limits; same arithmetic.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    nc = offs // plane_o
    pos = offs % plane_o
    h_in = pos // W_in
    w_in = pos % W_in
    h0 = (h_in * H_out) // H_in
    t = (h_in + 1) * H_out
    h1 = (t + H_in - 1) // H_in
    w0 = (w_in * W_out) // W_in
    t2 = (w_in + 1) * W_out
    w1 = (t2 + W_in - 1) // W_in
    g_base = g_ptr + nc * plane_g
    facc = tl.float64 if IS_FP64 else tl.float32
    acc = tl.zeros([BLOCK], dtype=facc)
    for i in tl.static_range(KMAX_H):
        ho = h0 + i
        vh = (ho < h1) & mask
        sh = (ho * H_in) // H_out
        eh = ((ho + 1) * H_in + H_out - 1) // H_out
        winh = eh - sh
        for j in tl.static_range(KMAX_W):
            wo = w0 + j
            valid = vh & (wo < w1)
            sw = (wo * W_in) // W_out
            ew = ((wo + 1) * W_in + W_out - 1) // W_out
            winw = ew - sw
            gv = tl.load(g_base + ho * W_out + wo, mask=valid, other=0.0)
            acc += gv.to(facc) / (winh * winw).to(facc)
    tl.store(out_ptr + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _specialized__adaptive_avg_pool2d_backward(grad_output, self):
    if self.dim() == 4:
        N, C, H_in, W_in = self.shape
    elif self.dim() == 3:
        C, H_in, W_in = self.shape
        N = 1
    else:
        raise RuntimeError(
            f"adaptive_avg_pool2d_backward: unsupported rank {self.dim()}"
        )

    if grad_output.dim() == 4:
        _, _, H_out, W_out = grad_output.shape
    else:
        _, H_out, W_out = grad_output.shape

    dev = self.device
    out = torch.empty(self.shape, dtype=self.dtype, device=dev)
    numel = out.numel()
    if numel == 0 or H_out == 0 or W_out == 0:
        return out

    NC = N * C
    plane_o = H_in * W_in
    plane_g = H_out * W_out
    IS_FP64 = self.dtype == torch.float64

    BLOCK_W = triton.next_power_of_2(W_in)
    rows = H_in * NC

    divisible = (H_in % H_out == 0) and (W_in % W_out == 0)

    if divisible and rows <= 65535:
        RH = H_in // H_out
        RW = W_in // W_out
        R = 128
        NW = 16 if W_in > 64 else 8
        HBLK = triton.cdiv(H_in, R)
        grid = (triton.cdiv(W_in, BLOCK_W), HBLK * NC)
        _aap2d_bwd_div[grid](
            grad_output,
            out,
            plane_g,
            plane_o,
            HBLK,
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            RH=RH,
            RW=RW,
            IS_FP64=IS_FP64,
            BLOCK_W=BLOCK_W,
            R=R,
            num_warps=NW,
        )
    elif rows <= 65535:
        KMAX_H = _max_cover(H_in, H_out)
        KMAX_W = _max_cover(W_in, W_out)
        num_warps = max(1, min(8, BLOCK_W // 32))
        grid = (triton.cdiv(W_in, BLOCK_W), rows)
        _aap2d_bwd_gen[grid](
            grad_output,
            out,
            plane_g,
            plane_o,
            H_in=H_in,
            W_in=W_in,
            H_out=H_out,
            W_out=W_out,
            IS_FP64=IS_FP64,
            KMAX_H=KMAX_H,
            KMAX_W=KMAX_W,
            BLOCK_W=BLOCK_W,
            num_warps=num_warps,
        )
    else:
        KMAX_H = _max_cover(H_in, H_out)
        KMAX_W = _max_cover(W_in, W_out)
        BLOCK = 1024
        grid = (triton.cdiv(numel, BLOCK),)
        _aap2d_bwd_fused[grid](
            grad_output,
            out,
            numel,
            plane_g,
            plane_o,
            H_in,
            W_in,
            H_out,
            W_out,
            IS_FP64=IS_FP64,
            KMAX_H=KMAX_H,
            KMAX_W=KMAX_W,
            BLOCK=BLOCK,
            num_warps=4,
        )
    return out


def _adaptive_avg_pool2d_backward(grad_output, self):
    logger.debug("GEMS_MTHREADS _ADAPTIVE_AVG_POOL2D_BACKWARD")
    if (
        isinstance(grad_output, torch.Tensor)
        and grad_output.device.type == "musa"
        and grad_output.dtype in _SUPPORTED_DTYPES
        and isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized__adaptive_avg_pool2d_backward(grad_output, self)
    return default__adaptive_avg_pool2d_backward(grad_output, self)
