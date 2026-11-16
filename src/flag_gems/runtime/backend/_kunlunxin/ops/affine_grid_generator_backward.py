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
#
# Kunlunxin XPU3 override for affine_grid_generator_backward.
#
# The generic implementation launches one (or N*6) program(s) per batch and
# serially walks every spatial row, so the K = D*H*W reduction is exposed to a
# handful of programs (N in 1..32 for the benchmark matrix).  On XPU3 that is
# 0.02–0.34x of ATen.  This override reformulates the contraction as a flat,
# massively parallel two-stage reduction:
#
#   * pass 1 tiles the (contiguous) [N, ..., coordinate_count] grad_output into
#     BLOCK-wide flat chunks.  Each chunk computes its six (2-D) / twelve (3-D)
#     partial sums with a single tuple ``tl.reduce`` (one tree traversal for all
#     outputs, ~2x faster than six separate ``tl.sum``).
#   * pass 2 (only when a batch spans more than one chunk) sums the per-chunk
#     partials.  When a whole batch fits in one chunk, pass 1 writes grad_theta
#     directly and the second launch is skipped.
#
# The chunk is contiguous, so the load is a single block DMA (no stride-2 /
# stride-3 gapped access), and per-lane coordinates only need cheap integer
# arithmetic.  Measured on XPU3 (P800): dtype-equal speedup ~2.2x vs ATen across
# the benchmark matrix (see harness/solution/affine_grid_generator_backward).

import torch
import triton
import triton.language as tl

# Widest flat chunk.  Measured on XPU3 for (8,256,256) fp32: the pass-1 kernel
# drops 530 -> 345 -> 253 -> 206 us as BLOCK grows 1024 -> 2048 -> 4096 -> 8192
# (the per-program floor flattens only above ~1 KB lanes per program).
_BLOCK = 8192


@triton.jit
def _add6(a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5):
    return a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4, a5 + b5


@triton.jit
def _add12(
    a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11,
    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11,
):
    return (
        a0 + b0, a1 + b1, a2 + b2, a3 + b3, a4 + b4, a5 + b5,
        a6 + b6, a7 + b7, a8 + b8, a9 + b9, a10 + b10, a11 + b11,
    )


@triton.jit
def _agb_pass1_2d(
    grad_ptr,
    part_ptr,
    out_ptr,
    W,
    H,
    ELEMS,
    TILES,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
    FULL: tl.constexpr,
    WRITE_OUT: tl.constexpr,
):
    """2-D: grad_output [N, H, W, 2] -> grad_theta [N, 2, 3].

    Each program owns one flat chunk of a single batch.  Column order matches
    the flattened output: (i0*x, i0*y, i0*1, i1*x, i1*y, i1*1).
    """
    acc_dtype = tl.float64 if grad_ptr.type.element_ty == tl.float64 else tl.float32
    n = tl.program_id(0)
    t = tl.program_id(1)
    off = t * BLOCK + tl.arange(0, BLOCK)
    if FULL:
        f = tl.load(grad_ptr + n * ELEMS + off).to(acc_dtype)
    else:
        m = off < ELEMS
        f = tl.load(grad_ptr + n * ELEMS + off, mask=m, other=0.0).to(acc_dtype)

    s = off // 2
    c = off % 2
    w = s % W
    h = s // W
    wf = w.to(acc_dtype)
    hf = h.to(acc_dtype)
    if ALIGN_CORNERS:
        x = tl.where(W > 1, 2.0 * wf / (W - 1) - 1.0, tl.zeros([BLOCK], acc_dtype))
        y = tl.where(H > 1, 2.0 * hf / (H - 1) - 1.0, tl.zeros([BLOCK], acc_dtype))
    else:
        x = (2.0 * wf + 1.0) / W - 1.0
        y = (2.0 * hf + 1.0) / H - 1.0

    f0 = tl.where(c == 0, f, 0.0)
    f1 = tl.where(c == 1, f, 0.0)
    r = tl.reduce((f0 * x, f0 * y, f0, f1 * x, f1 * y, f1), 0, _add6)

    if WRITE_OUT:
        tl.store(out_ptr + n * 6 + 0, r[0])
        tl.store(out_ptr + n * 6 + 1, r[1])
        tl.store(out_ptr + n * 6 + 2, r[2])
        tl.store(out_ptr + n * 6 + 3, r[3])
        tl.store(out_ptr + n * 6 + 4, r[4])
        tl.store(out_ptr + n * 6 + 5, r[5])
    else:
        pb = (n * 6) * TILES + t
        tl.store(part_ptr + pb + 0 * TILES, r[0])
        tl.store(part_ptr + pb + 1 * TILES, r[1])
        tl.store(part_ptr + pb + 2 * TILES, r[2])
        tl.store(part_ptr + pb + 3 * TILES, r[3])
        tl.store(part_ptr + pb + 4 * TILES, r[4])
        tl.store(part_ptr + pb + 5 * TILES, r[5])


@triton.jit
def _agb_pass2(part_ptr, out_ptr, TILES, NCOLS: tl.constexpr, BLOCK_T: tl.constexpr):
    """Sum the per-chunk partials for every (batch, output-column)."""
    pid = tl.program_id(0)
    n = pid // NCOLS
    j = pid % NCOLS
    o = tl.arange(0, BLOCK_T)
    m = o < TILES
    v = tl.load(part_ptr + (n * NCOLS + j) * TILES + o, mask=m, other=0.0)
    tl.store(out_ptr + n * NCOLS + j, tl.sum(v))


@triton.jit
def _agb_pass1_3d(
    grad_ptr,
    part_ptr,
    out_ptr,
    W,
    H,
    HW,
    ELEMS,
    TILES,
    ALIGN_CORNERS: tl.constexpr,
    BLOCK: tl.constexpr,
    FULL: tl.constexpr,
    WRITE_OUT: tl.constexpr,
):
    """3-D: grad_output [N, D, H, W, 3] -> grad_theta [N, 3, 4].

    Column order matches the flattened output:
    (i0*x, i0*y, i0*z, i0*1, i1*x, i1*y, i1*z, i1*1, i2*x, i2*y, i2*z, i2*1).
    """
    acc_dtype = tl.float64 if grad_ptr.type.element_ty == tl.float64 else tl.float32
    n = tl.program_id(0)
    t = tl.program_id(1)
    off = t * BLOCK + tl.arange(0, BLOCK)
    if FULL:
        f = tl.load(grad_ptr + n * ELEMS + off).to(acc_dtype)
    else:
        m = off < ELEMS
        f = tl.load(grad_ptr + n * ELEMS + off, mask=m, other=0.0).to(acc_dtype)

    D = HW // H
    s = off // 3
    c = off % 3
    w = s % W
    hw = s // W
    h = hw % H
    d = hw // H
    wf = w.to(acc_dtype)
    hf = h.to(acc_dtype)
    df = d.to(acc_dtype)
    if ALIGN_CORNERS:
        x = tl.where(W > 1, 2.0 * wf / (W - 1) - 1.0, tl.zeros([BLOCK], acc_dtype))
        y = tl.where(H > 1, 2.0 * hf / (H - 1) - 1.0, tl.zeros([BLOCK], acc_dtype))
        z = tl.where(D > 1, 2.0 * df / (D - 1) - 1.0, tl.zeros([BLOCK], acc_dtype))
    else:
        x = (2.0 * wf + 1.0) / W - 1.0
        y = (2.0 * hf + 1.0) / H - 1.0
        z = (2.0 * df + 1.0) / D - 1.0

    f0 = tl.where(c == 0, f, 0.0)
    f1 = tl.where(c == 1, f, 0.0)
    f2 = tl.where(c == 2, f, 0.0)
    r = tl.reduce(
        (
            f0 * x, f0 * y, f0 * z, f0,
            f1 * x, f1 * y, f1 * z, f1,
            f2 * x, f2 * y, f2 * z, f2,
        ),
        0,
        _add12,
    )

    if WRITE_OUT:
        for j in tl.static_range(12):
            tl.store(out_ptr + n * 12 + j, r[j])
    else:
        pb = (n * 12) * TILES + t
        for j in tl.static_range(12):
            tl.store(part_ptr + pb + j * TILES, r[j])


def affine_grid_generator_backward(grad_output, size, align_corners):
    assert grad_output.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ), f"affine_grid_generator_backward requires floating-point dtype, got {grad_output.dtype}"
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()

    n = size[0]
    device = grad_output.device
    dtype = grad_output.dtype

    if len(size) == 4:
        _, _, h, w = size
        grad_theta = torch.empty(n, 2, 3, device=device, dtype=dtype)
        elems = h * w * 2
        if n == 0 or elems == 0:
            return grad_theta

        block = min(_BLOCK, max(32, triton.next_power_of_2(elems)))
        tiles = triton.cdiv(elems, block)
        full = tiles * block == elems
        if tiles == 1:
            _agb_pass1_2d[(n, tiles)](
                grad_output,
                grad_theta,
                grad_theta,
                w,
                h,
                elems,
                tiles,
                ALIGN_CORNERS=align_corners,
                BLOCK=block,
                FULL=full,
                WRITE_OUT=True,
            )
        else:
            part = torch.empty(n * 6 * tiles, device=device, dtype=dtype)
            _agb_pass1_2d[(n, tiles)](
                grad_output,
                part,
                grad_theta,
                w,
                h,
                elems,
                tiles,
                ALIGN_CORNERS=align_corners,
                BLOCK=block,
                FULL=full,
                WRITE_OUT=False,
            )
            block_t = max(32, triton.next_power_of_2(tiles))
            _agb_pass2[(n * 6,)](part, grad_theta, tiles, NCOLS=6, BLOCK_T=block_t)
        return grad_theta

    if len(size) == 5:
        _, _, d, h, w = size
        grad_theta = torch.empty(n, 3, 4, device=device, dtype=dtype)
        elems = d * h * w * 3
        if n == 0 or elems == 0:
            return grad_theta

        hw = d * h
        block = min(_BLOCK, max(32, triton.next_power_of_2(elems)))
        tiles = triton.cdiv(elems, block)
        full = tiles * block == elems
        if tiles == 1:
            _agb_pass1_3d[(n, tiles)](
                grad_output,
                grad_theta,
                grad_theta,
                w,
                h,
                hw,
                elems,
                tiles,
                ALIGN_CORNERS=align_corners,
                BLOCK=block,
                FULL=full,
                WRITE_OUT=True,
            )
        else:
            part = torch.empty(n * 12 * tiles, device=device, dtype=dtype)
            _agb_pass1_3d[(n, tiles)](
                grad_output,
                part,
                grad_theta,
                w,
                h,
                hw,
                elems,
                tiles,
                ALIGN_CORNERS=align_corners,
                BLOCK=block,
                FULL=full,
                WRITE_OUT=False,
            )
            block_t = max(32, triton.next_power_of_2(tiles))
            _agb_pass2[(n * 12,)](part, grad_theta, tiles, NCOLS=12, BLOCK_T=block_t)
        return grad_theta

    raise RuntimeError(
        f"affine_grid_generator_backward: expected size with 4 or 5 dimensions, got {len(size)}"
    )
