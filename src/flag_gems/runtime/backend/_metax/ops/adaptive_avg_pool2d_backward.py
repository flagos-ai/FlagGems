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

"""Triton implementation of _adaptive_avg_pool2d_backward.

Every grad_input element is written exactly once by exactly one lane; no
atomics and no zero-fill pass are needed.

Adaptive window boundaries (ATen semantics):
    starts(i) = floor(i * I / O)
    ends(i)   = ceil((i + 1) * I / O)
Inverse membership (input index h belongs to output indices i with):
    i_lo(h) = floor(h * O / I)
    i_hi(h) = ceil((h + 1) * O / I) - 1

Three specializations, chosen on the host:
  * exact      : I % O == 0 on both axes. Every input element belongs to
                 exactly one window, so lane = (h, j) reads grad_output
                 (h//KH, j) once and writes the KW contiguous grad_input
                 elements of row h, columns [j*KW, (j+1)*KW). With a
                 whole-tensor flat lane index the destination is simply
                 lane*KW, i.e. one contiguous store stream.
  * downsample : I >= O on both axes -> at most 2x2 contributing windows.
  * generic    : upsampling on some axis -> K_i x K_j window loop.

All shape metadata is passed as tl.constexpr so integer divides lower to
multiply/shift.
"""

import torch
import triton
import triton.language as tl

_CONTIG = torch.contiguous_format


@triton.jit
def _aap2d_bwd_exact_kernel(
    grad_out_ptr,
    grad_in_ptr,
    TOTAL: tl.constexpr,
    LANES_PER_PLANE: tl.constexpr,
    OUT_PLANE: tl.constexpr,
    OUT_W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """I/O divides exactly on both axes and grad_output is contiguous.

    One lane per (input row h, output column j). Each lane performs a single
    scalar read and a run of KW contiguous stores; the flat lane index
    enumerates (plane, h, j) in exactly destination order, so the whole tile of
    BLOCK*KW destination elements is one contiguous stream.
    """
    pid = tl.program_id(0)
    lane = pid * BLOCK + tl.arange(0, BLOCK)

    plane = lane // LANES_PER_PLANE
    e = lane - plane * LANES_PER_PLANE
    h = e // OUT_W
    j = e - h * OUT_W
    src = plane.to(tl.int64) * OUT_PLANE + (h // KH) * OUT_W + j

    in_ptr = grad_in_ptr + lane.to(tl.int64) * KW
    r = tl.arange(0, PAD)

    if (pid + 1) * BLOCK <= TOTAL:
        g = tl.load(grad_out_ptr + src)
        val = (g.to(tl.float32) * SCALE).to(grad_in_ptr.dtype.element_ty)
        dst = in_ptr[:, None] + r[None, :]
        if KW == PAD:
            tl.store(dst, val[:, None])
        else:
            tl.store(dst, val[:, None], mask=(r < KW)[None, :])
    else:
        mask = lane < TOTAL
        g = tl.load(grad_out_ptr + src, mask=mask, other=0.0)
        val = (g.to(tl.float32) * SCALE).to(grad_in_ptr.dtype.element_ty)
        dst = in_ptr[:, None] + r[None, :]
        if KW == PAD:
            tl.store(dst, val[:, None], mask=mask[:, None])
        else:
            tl.store(dst, val[:, None], mask=mask[:, None] & (r < KW)[None, :])


@triton.jit
def _aap2d_bwd_downsample_kernel(
    grad_out_ptr,
    grad_in_ptr,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    IN_C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    PLANE_SIZE: tl.constexpr,
    NUM_W_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """I >= O on both axes: at most 2x2 contributing windows per element."""
    pid_hw = tl.program_id(0)
    plane = tl.program_id(1)

    pid_h = pid_hw // NUM_W_BLOCKS
    pid_w = pid_hw - pid_h * NUM_W_BLOCKS

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_h = h_offs < IN_H
    mask_w = w_offs < IN_W

    n_idx = plane // IN_C
    c_idx = plane - n_idx * IN_C
    out_base = n_idx.to(tl.int64) * out_stride_n + c_idx.to(tl.int64) * out_stride_c
    out_plane_ptr = grad_out_ptr + out_base

    i_lo = (h_offs * OUT_H) // IN_H
    i_hi = ((h_offs + 1) * OUT_H + IN_H - 1) // IN_H - 1
    j_lo = (w_offs * OUT_W) // IN_W
    j_hi = ((w_offs + 1) * OUT_W + IN_W - 1) // IN_W - 1

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for ii in tl.static_range(2):
        i = i_lo + ii
        valid_i = i <= i_hi
        kh = ((i + 1) * IN_H + OUT_H - 1) // OUT_H - (i * IN_H) // OUT_H
        inv_kh = 1.0 / kh.to(tl.float32)
        row_off = i * out_stride_h
        for jj in tl.static_range(2):
            j = j_lo + jj
            valid_j = j <= j_hi
            kw = ((j + 1) * IN_W + OUT_W - 1) // OUT_W - (j * IN_W) // OUT_W
            inv_kw = 1.0 / kw.to(tl.float32)
            m = mask_h[:, None] & mask_w[None, :] & valid_i[:, None] & valid_j[None, :]
            p = row_off[:, None] + j * out_stride_w
            g = tl.load(out_plane_ptr + p, mask=m, other=0.0)
            acc += g.to(tl.float32) * (inv_kh[:, None] * inv_kw[None, :])

    in_plane_ptr = grad_in_ptr + plane.to(tl.int64) * PLANE_SIZE
    p_in = h_offs[:, None] * IN_W + w_offs[None, :]
    tl.store(
        in_plane_ptr + p_in,
        acc.to(grad_in_ptr.dtype.element_ty),
        mask=mask_h[:, None] & mask_w[None, :],
    )


@triton.jit
def _aap2d_bwd_generic_kernel(
    grad_out_ptr,
    grad_in_ptr,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    IN_C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    PLANE_SIZE: tl.constexpr,
    NUM_W_BLOCKS: tl.constexpr,
    K_I: tl.constexpr,
    K_J: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Upsampling on some axis: one input may belong to many windows."""
    pid_hw = tl.program_id(0)
    plane = tl.program_id(1)

    pid_h = pid_hw // NUM_W_BLOCKS
    pid_w = pid_hw - pid_h * NUM_W_BLOCKS

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)

    mask_h = h_offs < IN_H
    mask_w = w_offs < IN_W

    n_idx = plane // IN_C
    c_idx = plane - n_idx * IN_C
    out_base = n_idx.to(tl.int64) * out_stride_n + c_idx.to(tl.int64) * out_stride_c
    out_plane_ptr = grad_out_ptr + out_base

    i_lo = (h_offs * OUT_H) // IN_H
    i_end = ((h_offs + 1) * OUT_H + IN_H - 1) // IN_H
    j_lo = (w_offs * OUT_W) // IN_W
    j_end = ((w_offs + 1) * OUT_W + IN_W - 1) // IN_W

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    for ii in range(K_I):
        i = i_lo + ii
        valid_i = i < i_end
        kh = ((i + 1) * IN_H + OUT_H - 1) // OUT_H - (i * IN_H) // OUT_H
        inv_kh = 1.0 / kh.to(tl.float32)
        row_off = i * out_stride_h
        for jj in range(K_J):
            j = j_lo + jj
            valid_j = j < j_end
            kw = ((j + 1) * IN_W + OUT_W - 1) // OUT_W - (j * IN_W) // OUT_W
            inv_kw = 1.0 / kw.to(tl.float32)
            m = mask_h[:, None] & mask_w[None, :] & valid_i[:, None] & valid_j[None, :]
            p = row_off[:, None] + j * out_stride_w
            g = tl.load(out_plane_ptr + p, mask=m, other=0.0)
            acc += g.to(tl.float32) * (inv_kh[:, None] * inv_kw[None, :])

    in_plane_ptr = grad_in_ptr + plane.to(tl.int64) * PLANE_SIZE
    p_in = h_offs[:, None] * IN_W + w_offs[None, :]
    tl.store(
        in_plane_ptr + p_in,
        acc.to(grad_in_ptr.dtype.element_ty),
        mask=mask_h[:, None] & mask_w[None, :],
    )


def _pow2(n):
    return 1 << (n - 1).bit_length()


def _adaptive_avg_pool2d_backward(grad_output, self):
    grad_input = torch.empty_like(self, memory_format=_CONTIG)

    shape = self.shape
    out_shape = grad_output.shape
    ndim = len(shape)
    if ndim == 3:
        in_n = 1
        in_c = shape[0]
        in_h = shape[1]
        in_w = shape[2]
        out_h = out_shape[1]
        out_w = out_shape[2]
    else:
        in_n = shape[0]
        in_c = shape[1]
        in_h = shape[2]
        in_w = shape[3]
        out_h = out_shape[2]
        out_w = out_shape[3]

    plane_size = in_h * in_w
    if grad_output.numel() == 0 or plane_size == 0 or in_c == 0:
        return grad_input

    if in_h % out_h == 0 and in_w % out_w == 0 and grad_output.is_contiguous():
        kh = in_h // out_h
        kw = in_w // out_w
        total = in_n * in_c * in_h * out_w
        blk = 256
        if total < blk:
            blk = 32 if total < 32 else _pow2(total)
        _aap2d_bwd_exact_kernel[(-(-total // blk),)](
            grad_output,
            grad_input,
            total,
            in_h * out_w,
            out_h * out_w,
            out_w,
            kh,
            kw,
            _pow2(kw),
            1.0 / (kh * kw),
            BLOCK=blk,
            num_warps=2 if blk > 64 else 1,
        )
        return grad_input

    if ndim == 3:
        s_n = 0
        s_c, s_h, s_w = grad_output.stride()
    else:
        s_n, s_c, s_h, s_w = grad_output.stride()

    bw = min(1024, _pow2(in_w))
    bh = max(1, min(1024 // bw, _pow2(in_h)))
    n_w_blocks = -(-in_w // bw)
    n_h_blocks = -(-in_h // bh)
    grid = (n_h_blocks * n_w_blocks, in_n * in_c)
    num_warps = max(1, min(8, (bh * bw) // 256))

    if in_h >= out_h and in_w >= out_w:
        _aap2d_bwd_downsample_kernel[grid](
            grad_output,
            grad_input,
            s_n,
            s_c,
            s_h,
            s_w,
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            plane_size,
            n_w_blocks,
            BLOCK_H=bh,
            BLOCK_W=bw,
            num_warps=num_warps,
        )
    else:
        _aap2d_bwd_generic_kernel[grid](
            grad_output,
            grad_input,
            s_n,
            s_c,
            s_h,
            s_w,
            in_c,
            in_h,
            in_w,
            out_h,
            out_w,
            plane_size,
            n_w_blocks,
            -(-out_h // in_h) + 1,
            -(-out_w // in_w) + 1,
            BLOCK_H=bh,
            BLOCK_W=bw,
            num_warps=num_warps,
        )

    return grad_input
