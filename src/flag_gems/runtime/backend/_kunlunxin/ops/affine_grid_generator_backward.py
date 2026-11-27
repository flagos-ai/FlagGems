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

# Kunlunxin (XPU) override of aten::affine_grid_generator_backward.
#
# Why a vendor override exists at all: the generic KernelGen implementation
# compiles pathologically on XPU.  Its kernels keep a `tl.sum` reduction tree
# inside the per-row loop and address the two (three) output channels with
# stride-2 (stride-3) gathers; for (2, 512, 512) a single kernel compile was
# observed to run > 25 min without producing a .so (measured 2026-11-25),
# and even the small test shapes dump a 35 MB IR log.
#
# Design (two-phase partial + merge, bounded tiles, single reduce tree per
# program):
#   * 2D partial kernel, grid = (N * HB,): one program owns R consecutive rows
#     of one batch element.  Both output channels of a row chunk are loaded as
#     ONE contiguous stride-1 vector (BLOCK_P lanes) and the two channels are
#     separated on the fly with a +1/-1 parity sign, so the XPU backend lowers
#     the load to block DMA instead of two stride-2 gathers.  Six per-lane
#     [BLOCK_P] accumulators defer every cross-lane reduction to one `tl.sum`
#     per output element at the end of the program; no `tt.reduce` is ever
#     placed inside a loop (XPU lowering constraint); the (t +/- u) * 0.5
#     folding recovers the per-channel sums exactly.
#   * 3D partial kernel, same decomposition over (d, h) row pairs; the three
#     channels are loaded with three affine strides (3D only covers small W
#     in the tests).
#   * Merge kernel, grid = (N,): scalar loop over the HB per-batch partials
#     (deterministic accumulation order, atomic-free).
#   * Single-pass path: small planes with enough batches (N >= 8 and
#     N * rows <= 1024) run one program per batch and skip the partial buffer
#     and the merge launch entirely.
#
# Measured XPU constraints (2026-11-26, container liuyu_fg_env, triton 3.6.0):
#   * Loop-carried tiled accumulators are rejected by TritonXPUUnrollControl
#     beyond a per-program tiled-register budget ("unroll control cannot
#     narrow hoisted operand", uni_sram OOM), so 2D uses BLOCK_P <= 1024 with
#     six accumulators and 3D uses BLOCK_W <= 128 with twelve accumulators.
#   * tl.reshape + tl.split (interleaved deinterleaving) fails to lower on
#     this backend (uni_sram OOM even at BLOCK_W = 128), which is why the 2D
#     kernel uses the parity-sign trick instead.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# Number of row-blocks the plane of one batch is split into, aimed at a total
# grid of ~_TARGET_PROGS programs.  The merge kernel is HB iterations long, so
# keeping HB modest also keeps the second launch cheap.
_TARGET_PROGS = 64
# N * rows below this go single-pass (one program per batch, no merge).
_SINGLE_PASS_WORK = 1024
# 2D pair-vector lane cap (2 lanes per spatial position); the six per-lane
# accumulators at this width keep the compile and uni_sram footprint within
# what the XPU backend accepts (measured 2026-11-26).
_BLOCK_P_CAP = 1024
_BLOCK_W_CAP_3D = 128
_VALS_2D = 6
_VALS_2D_PAD = 8
_VALS_3D = 12
_VALS_3D_PAD = 16


def _part_dtype(input_dtype):
    return torch.float64 if input_dtype == torch.float64 else torch.float32


def _block_w(width, cap):
    block_w = min(triton.next_power_of_2(width), cap)
    return max(32, block_w)


@libentry()
@triton.jit
def _affine_grid_bwd_2d_partial_kernel(
    grad_ptr,
    part_ptr,
    out_ptr,
    H,
    W,
    HB,
    R,
    inv_w,
    inv_h,
    inv_wm1,
    inv_hm1,
    align_corners: tl.constexpr,
    BLOCK_P: tl.constexpr,
    VALS_PAD: tl.constexpr,
    VALS: tl.constexpr,
    EXACT_W: tl.constexpr,
    FINAL: tl.constexpr,
):
    """One program owns R consecutive rows of one batch element (2D case).

    grad_output is [N, H, W, 2] and grad_theta is [N, 2, 3].

    Both output channels are loaded as one contiguous stride-1 vector of
    BLOCK_P lanes (BLOCK_P covers 2*W chunks of a row); the two channels are
    separated on the fly with a +-1 parity sign instead of two stride-2
    gathers, so the XPU backend lowers the load to block DMA.  Per-lane
    [BLOCK_P] accumulators defer every cross-lane reduction to a single tree
    per output element at the end of the program; the (t +/- u) * 0.5 folding
    recovers the per-channel sums exactly (power-of-two scaling).
    """
    acc_dtype = tl.float64 if grad_ptr.dtype.element_ty == tl.float64 else tl.float32
    pid = tle.program_id(0)
    batch = pid // HB
    blk = pid - batch * HB

    num_chunks = tl.cdiv(2 * W, BLOCK_P)
    rows = H * num_chunks
    r0 = blk * R
    r1 = tl.minimum(r0 + R, rows)

    offs = tl.arange(0, BLOCK_P)
    c = offs % 2
    sg = 1.0 - 2.0 * c.to(acc_dtype)
    wx_f = (offs // 2).to(acc_dtype)

    ax = tl.zeros([BLOCK_P], dtype=acc_dtype)
    ay = tl.zeros([BLOCK_P], dtype=acc_dtype)
    a1 = tl.zeros([BLOCK_P], dtype=acc_dtype)
    sx = tl.zeros([BLOCK_P], dtype=acc_dtype)
    sy = tl.zeros([BLOCK_P], dtype=acc_dtype)
    s1 = tl.zeros([BLOCK_P], dtype=acc_dtype)

    batch_base = batch * H * W * 2
    for r in range(r0, r1):
        h = r // num_chunks
        wt = r - h * num_chunks
        e0 = wt * BLOCK_P

        h_f = h.to(acc_dtype)
        if align_corners:
            y_val = tl.where(H > 1, (2.0 * h_f - (H - 1)) * inv_hm1, 0.0)
        else:
            y_val = (2.0 * h_f + 1.0) * inv_h - 1.0
        if align_corners:
            x2 = tl.where(W > 1, (2.0 * (e0 + wx_f) - (W - 1)) * inv_wm1, 0.0)
        else:
            x2 = (2.0 * (e0 + wx_f) + 1.0) * inv_w - 1.0

        row_base = batch_base + h * (W * 2) + e0
        if EXACT_W:
            v = tl.load(grad_ptr + row_base + offs)
        else:
            v = tl.load(
                grad_ptr + row_base + offs,
                mask=(e0 + offs) < 2 * W,
                other=0.0,
            )
        v = v.to(acc_dtype)
        ax += v * x2
        ay += v * y_val
        a1 += v
        sx += v * x2 * sg
        sy += v * y_val * sg
        s1 += v * sg

    tx = tl.sum(ax, axis=0)
    ty = tl.sum(ay, axis=0)
    t1 = tl.sum(a1, axis=0)
    ux = tl.sum(sx, axis=0)
    uy = tl.sum(sy, axis=0)
    u1 = tl.sum(s1, axis=0)
    s_g0x = (tx + ux) * 0.5
    s_g1x = (tx - ux) * 0.5
    s_g0y = (ty + uy) * 0.5
    s_g1y = (ty - uy) * 0.5
    s_g0 = (t1 + u1) * 0.5
    s_g1 = (t1 - u1) * 0.5

    if FINAL:
        base = out_ptr + batch * VALS
        out_ty = out_ptr.dtype.element_ty
        tl.store(base + 0, s_g0x.to(out_ty))
        tl.store(base + 1, s_g0y.to(out_ty))
        tl.store(base + 2, s_g0.to(out_ty))
        tl.store(base + 3, s_g1x.to(out_ty))
        tl.store(base + 4, s_g1y.to(out_ty))
        tl.store(base + 5, s_g1.to(out_ty))
    else:
        base = part_ptr + (batch * HB + blk) * VALS_PAD
        part_ty = base.dtype.element_ty
        tl.store(base + 0, s_g0x.to(part_ty))
        tl.store(base + 1, s_g0y.to(part_ty))
        tl.store(base + 2, s_g0.to(part_ty))
        tl.store(base + 3, s_g1x.to(part_ty))
        tl.store(base + 4, s_g1y.to(part_ty))
        tl.store(base + 5, s_g1.to(part_ty))


@libentry()
@triton.jit
def _affine_grid_bwd_3d_partial_kernel(
    grad_ptr,
    part_ptr,
    out_ptr,
    D,
    H,
    W,
    HB,
    R,
    align_corners: tl.constexpr,
    BLOCK_W: tl.constexpr,
    VALS_PAD: tl.constexpr,
    VALS: tl.constexpr,
    EXACT_W: tl.constexpr,
    FINAL: tl.constexpr,
):
    """3D variant: rows are (d, h) pairs of one batch element.

    grad_output is [N, D, H, W, 3] and grad_theta is [N, 3, 4].
    """
    acc_dtype = tl.float64 if grad_ptr.dtype.element_ty == tl.float64 else tl.float32
    pid = tle.program_id(0)
    batch = pid // HB
    blk = pid - batch * HB
    r0 = blk * R
    r1 = tl.minimum(r0 + R, D * H * tl.cdiv(W, BLOCK_W))

    w_offs = tl.arange(0, BLOCK_W)

    acc_g0 = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g0x = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g0y = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g0z = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g1 = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g1x = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g1y = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g1z = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g2 = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g2x = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g2y = tl.zeros([BLOCK_W], dtype=acc_dtype)
    acc_g2z = tl.zeros([BLOCK_W], dtype=acc_dtype)

    batch_base = batch * D * H * W * 3
    num_w_tiles = tl.cdiv(W, BLOCK_W)

    for r in range(r0, r1):
        d = r // H
        h = r - d * H
        d_f = d.to(acc_dtype)
        h_f = h.to(acc_dtype)
        if align_corners:
            y_val = tl.where(H > 1, 2.0 * h_f / (H - 1) - 1.0, 0.0)
            z_val = tl.where(D > 1, 2.0 * d_f / (D - 1) - 1.0, 0.0)
        else:
            y_val = (2.0 * h_f + 1.0) / H - 1.0
            z_val = (2.0 * d_f + 1.0) / D - 1.0

        row_base = batch_base + r * (W * 3)

        for wt in range(num_w_tiles):
            w0 = wt * BLOCK_W
            if align_corners:
                x_vec = tl.where(
                    W > 1, 2.0 * (w0 + w_offs).to(acc_dtype) / (W - 1) - 1.0, 0.0
                )
            else:
                x_vec = (2.0 * (w0 + w_offs).to(acc_dtype) + 1.0) / W - 1.0

            ptrs = grad_ptr + row_base + (w0 + w_offs) * 3
            if EXACT_W:
                g0 = tl.load(ptrs)
                g1 = tl.load(ptrs + 1)
                g2 = tl.load(ptrs + 2)
            else:
                w_mask = (w0 + w_offs) < W
                g0 = tl.load(ptrs, mask=w_mask, other=0.0)
                g1 = tl.load(ptrs + 1, mask=w_mask, other=0.0)
                g2 = tl.load(ptrs + 2, mask=w_mask, other=0.0)
            g0 = g0.to(acc_dtype)
            g1 = g1.to(acc_dtype)
            g2 = g2.to(acc_dtype)

            acc_g0 += g0
            acc_g0x += g0 * x_vec
            acc_g0y += g0 * y_val
            acc_g0z += g0 * z_val
            acc_g1 += g1
            acc_g1x += g1 * x_vec
            acc_g1y += g1 * y_val
            acc_g1z += g1 * z_val
            acc_g2 += g2
            acc_g2x += g2 * x_vec
            acc_g2y += g2 * y_val
            acc_g2z += g2 * z_val

    s_g0x = tl.sum(acc_g0x, axis=0)
    s_g0y = tl.sum(acc_g0y, axis=0)
    s_g0z = tl.sum(acc_g0z, axis=0)
    s_g0 = tl.sum(acc_g0, axis=0)
    s_g1x = tl.sum(acc_g1x, axis=0)
    s_g1y = tl.sum(acc_g1y, axis=0)
    s_g1z = tl.sum(acc_g1z, axis=0)
    s_g1 = tl.sum(acc_g1, axis=0)
    s_g2x = tl.sum(acc_g2x, axis=0)
    s_g2y = tl.sum(acc_g2y, axis=0)
    s_g2z = tl.sum(acc_g2z, axis=0)
    s_g2 = tl.sum(acc_g2, axis=0)

    if FINAL:
        base = out_ptr + batch * VALS
        out_ty = out_ptr.dtype.element_ty
        tl.store(base + 0, s_g0x.to(out_ty))
        tl.store(base + 1, s_g0y.to(out_ty))
        tl.store(base + 2, s_g0z.to(out_ty))
        tl.store(base + 3, s_g0.to(out_ty))
        tl.store(base + 4, s_g1x.to(out_ty))
        tl.store(base + 5, s_g1y.to(out_ty))
        tl.store(base + 6, s_g1z.to(out_ty))
        tl.store(base + 7, s_g1.to(out_ty))
        tl.store(base + 8, s_g2x.to(out_ty))
        tl.store(base + 9, s_g2y.to(out_ty))
        tl.store(base + 10, s_g2z.to(out_ty))
        tl.store(base + 11, s_g2.to(out_ty))
    else:
        base = part_ptr + (batch * HB + blk) * VALS_PAD
        tl.store(base + 0, s_g0x)
        tl.store(base + 1, s_g0y)
        tl.store(base + 2, s_g0z)
        tl.store(base + 3, s_g0)
        tl.store(base + 4, s_g1x)
        tl.store(base + 5, s_g1y)
        tl.store(base + 6, s_g1z)
        tl.store(base + 7, s_g1)
        tl.store(base + 8, s_g2x)
        tl.store(base + 9, s_g2y)
        tl.store(base + 10, s_g2z)
        tl.store(base + 11, s_g2)


@libentry()
@triton.jit
def _affine_grid_bwd_merge_kernel(
    part_ptr,
    out_ptr,
    HB,
    VALS: tl.constexpr,
    VALS_PAD: tl.constexpr,
):
    """Fold the [HB, VALS] per-batch partials of one batch element.

    A scalar loop over HB keeps the accumulation order deterministic and the
    masked lanes of the padded partial buffer are never stored to the output.
    """
    batch = tle.program_id(0)
    offs = tl.arange(0, VALS_PAD)
    acc = tl.zeros([VALS_PAD], dtype=part_ptr.dtype.element_ty)
    base = part_ptr + batch * HB * VALS_PAD
    for i in range(HB):
        acc += tl.load(base + i * VALS_PAD + offs, mask=offs < VALS, other=0.0)
    tl.store(
        out_ptr + batch * VALS + offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=offs < VALS,
    )


def affine_grid_generator_backward(grad_output, size, align_corners):
    """
    Compute the backward pass of affine_grid_generator using Triton.

    Args:
        grad_output: Gradient w.r.t. the grid output.
            2D: [N, H, W, 2]
            3D: [N, D, H, W, 3]
        size: The size of the input tensor [N, C, H, W] or [N, C, D, H, W]
        align_corners: Whether to align corners.

    Note: Only floating-point dtypes are supported (float16, bfloat16,
          float32, float64).  Accumulation follows the input dtype: float64
          inputs accumulate in float64, others accumulate in float32.

    Returns:
        grad_theta: Gradient w.r.t. theta.
            2D: [N, 2, 3]
            3D: [N, 3, 4]
    """
    logger.debug("GEMS_KUNLUNXIN AFFINE_GRID_GENERATOR_BACKWARD")
    assert grad_output.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"affine_grid_generator_backward requires floating-point dtype, got {grad_output.dtype}"
    grad_output = grad_output.contiguous()
    N = size[0]

    if len(size) == 4:
        # 2D case: size = [N, C, H, W], grad_output = [N, H, W, 2]
        H, W = size[2], size[3]
        grad_theta = torch.empty(
            N, 2, 3, device=grad_output.device, dtype=grad_output.dtype
        )
        if N == 0 or H == 0 or W == 0:
            return grad_theta

        block_p = max(32, min(triton.next_power_of_2(2 * W), _BLOCK_P_CAP))
        num_chunks = triton.cdiv(2 * W, block_p)
        rows = H * num_chunks
        # Small planes are launch/merge-overhead bound: if one batch has few
        # rows and the batch count is large enough to fill the device, run one
        # program per batch and skip the partial buffer and merge launch.
        if N >= 8 and N * rows <= _SINGLE_PASS_WORK:
            hb = 1
        else:
            hb = max(1, min(rows, _TARGET_PROGS // N))
        rows_per_block = triton.cdiv(rows, hb)
        exact_w = (2 * W) % block_p == 0
        inv_w = 1.0 / W
        inv_h = 1.0 / H
        inv_wm1 = 1.0 / max(W - 1, 1)
        inv_hm1 = 1.0 / max(H - 1, 1)

        if hb == 1:
            _affine_grid_bwd_2d_partial_kernel[(N,)](
                grad_output,
                grad_theta,
                grad_theta,
                H,
                W,
                1,
                rows,
                inv_w,
                inv_h,
                inv_wm1,
                inv_hm1,
                align_corners,
                BLOCK_P=block_p,
                VALS_PAD=_VALS_2D_PAD,
                VALS=_VALS_2D,
                EXACT_W=exact_w,
                FINAL=True,
            )
        else:
            part = torch.empty(
                (N, hb, _VALS_2D_PAD),
                device=grad_output.device,
                dtype=_part_dtype(grad_output.dtype),
            )
            _affine_grid_bwd_2d_partial_kernel[(N * hb,)](
                grad_output,
                part,
                grad_theta,
                H,
                W,
                hb,
                rows_per_block,
                inv_w,
                inv_h,
                inv_wm1,
                inv_hm1,
                align_corners,
                BLOCK_P=block_p,
                VALS_PAD=_VALS_2D_PAD,
                VALS=_VALS_2D,
                EXACT_W=exact_w,
                FINAL=False,
            )
            _affine_grid_bwd_merge_kernel[(N,)](
                part,
                grad_theta,
                hb,
                VALS=_VALS_2D,
                VALS_PAD=_VALS_2D_PAD,
            )
        return grad_theta

    # 3D case: size = [N, C, D, H, W], grad_output = [N, D, H, W, 3]
    D, H, W = size[2], size[3], size[4]
    grad_theta = torch.empty(
        N, 3, 4, device=grad_output.device, dtype=grad_output.dtype
    )
    if N == 0 or D == 0 or H == 0 or W == 0:
        return grad_theta

    block_w = _block_w(W, _BLOCK_W_CAP_3D)
    num_w_tiles = triton.cdiv(W, block_w)
    rows = D * H * num_w_tiles
    if N >= 8 and N * rows <= _SINGLE_PASS_WORK:
        hb = 1
    else:
        hb = max(1, min(rows, _TARGET_PROGS // N))
    rows_per_block = triton.cdiv(rows, hb)
    exact_w = W % block_w == 0

    if hb == 1:
        _affine_grid_bwd_3d_partial_kernel[(N,)](
            grad_output,
            grad_theta,
            grad_theta,
            D,
            H,
            W,
            1,
            rows,
            align_corners,
            BLOCK_W=block_w,
            VALS_PAD=_VALS_3D_PAD,
            VALS=_VALS_3D,
            EXACT_W=exact_w,
            FINAL=True,
        )
    else:
        part = torch.empty(
            (N, hb, _VALS_3D_PAD),
            device=grad_output.device,
            dtype=_part_dtype(grad_output.dtype),
        )
        _affine_grid_bwd_3d_partial_kernel[(N * hb,)](
            grad_output,
            part,
            grad_theta,
            D,
            H,
            W,
            hb,
            rows_per_block,
            align_corners,
            BLOCK_W=block_w,
            VALS_PAD=_VALS_3D_PAD,
            VALS=_VALS_3D,
            EXACT_W=exact_w,
            FINAL=False,
        )
        _affine_grid_bwd_merge_kernel[(N,)](
            part,
            grad_theta,
            hb,
            VALS=_VALS_3D,
            VALS_PAD=_VALS_3D_PAD,
        )
    return grad_theta
