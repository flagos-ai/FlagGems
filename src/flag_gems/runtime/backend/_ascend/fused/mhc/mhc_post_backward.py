"""Modified Triton mhc_post_backward matching NPU semantics.

This is a copy of the original since mhc_post_backward already produces
outputs identical to the NPU implementation for D>=128.

Gradients:
    grad_x[j,d]     = sum_i h_res[j,i] * grad_y[i,d]
    grad_h_res[j,i]  = sum_d x[j,d]    * grad_y[i,d]
    grad_h_out[d]    = sum_i h_post[i]  * grad_y[i,d]
    grad_h_post[i]   = sum_d h_out[d]   * grad_y[i,d]
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 256}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 1024}, num_warps=8, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_dxdho_hc4(
    grad_y_ptr,
    h_res_ptr,
    h_post_ptr,
    grad_x_ptr,
    grad_hout_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    gy_base = pid_t * 4 * D
    gx_base = pid_t * 4 * D
    gho_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    gy0 = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0).to(
        tl.float32
    )
    gy1 = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0).to(
        tl.float32
    )
    gy2 = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0).to(
        tl.float32
    )
    gy3 = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0).to(
        tl.float32
    )

    gx0 = r00 * gy0 + r01 * gy1 + r02 * gy2 + r03 * gy3
    gx1 = r10 * gy0 + r11 * gy1 + r12 * gy2 + r13 * gy3
    gx2 = r20 * gy0 + r21 * gy1 + r22 * gy2 + r23 * gy3
    gx3 = r30 * gy0 + r31 * gy1 + r32 * gy2 + r33 * gy3

    gho = hp0 * gy0 + hp1 * gy1 + hp2 * gy2 + hp3 * gy3

    dt = grad_x_ptr.dtype.element_ty
    tl.store(grad_x_ptr + gx_base + 0 * D + d_off, gx0.to(dt), mask=d_mask)
    tl.store(grad_x_ptr + gx_base + 1 * D + d_off, gx1.to(dt), mask=d_mask)
    tl.store(grad_x_ptr + gx_base + 2 * D + d_off, gx2.to(dt), mask=d_mask)
    tl.store(grad_x_ptr + gx_base + 3 * D + d_off, gx3.to(dt), mask=d_mask)
    tl.store(grad_hout_ptr + gho_base + d_off, gho.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# LoopD + manual 2-stage pipeline for dxdho (same structure as forward v3).
# grid=(T,) -> safe for any T (unlike the (T, cdiv) grid which exceeds
# coreDim=65535 at T>=16384 with BLOCK_D<3584).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 896}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 1792}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=1),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_dxdho_hc4_loopD_pipe(
    grad_y_ptr,
    h_res_ptr,
    h_post_ptr,
    grad_x_ptr,
    grad_hout_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    NUM_D_BLOCKS: tl.constexpr = (D + BLOCK_D - 1) // BLOCK_D

    gy_base = pid_t * 4 * D
    gx_base = pid_t * 4 * D
    gho_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    # ---- Prologue: load d-block 0 ----
    d_off = tl.arange(0, BLOCK_D)
    d_mask = d_off < D
    gy0_cur = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0)
    gy1_cur = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0)
    gy2_cur = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0)
    gy3_cur = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0)

    for db in range(NUM_D_BLOCKS):
        nxt = (db + 1) % NUM_D_BLOCKS
        d_off_nxt = nxt * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask_nxt = d_off_nxt < D
        gy0_nxt = tl.load(
            grad_y_ptr + gy_base + 0 * D + d_off_nxt, mask=d_mask_nxt, other=0.0
        )
        gy1_nxt = tl.load(
            grad_y_ptr + gy_base + 1 * D + d_off_nxt, mask=d_mask_nxt, other=0.0
        )
        gy2_nxt = tl.load(
            grad_y_ptr + gy_base + 2 * D + d_off_nxt, mask=d_mask_nxt, other=0.0
        )
        gy3_nxt = tl.load(
            grad_y_ptr + gy_base + 3 * D + d_off_nxt, mask=d_mask_nxt, other=0.0
        )

        gy0_f = gy0_cur.to(tl.float32)
        gy1_f = gy1_cur.to(tl.float32)
        gy2_f = gy2_cur.to(tl.float32)
        gy3_f = gy3_cur.to(tl.float32)

        gx0 = r00 * gy0_f + r01 * gy1_f + r02 * gy2_f + r03 * gy3_f
        gx1 = r10 * gy0_f + r11 * gy1_f + r12 * gy2_f + r13 * gy3_f
        gx2 = r20 * gy0_f + r21 * gy1_f + r22 * gy2_f + r23 * gy3_f
        gx3 = r30 * gy0_f + r31 * gy1_f + r32 * gy2_f + r33 * gy3_f
        gho = hp0 * gy0_f + hp1 * gy1_f + hp2 * gy2_f + hp3 * gy3_f

        d_off_cur = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask_cur = d_off_cur < D
        dt = grad_x_ptr.dtype.element_ty
        tl.store(grad_x_ptr + gx_base + 0 * D + d_off_cur, gx0.to(dt), mask=d_mask_cur)
        tl.store(grad_x_ptr + gx_base + 1 * D + d_off_cur, gx1.to(dt), mask=d_mask_cur)
        tl.store(grad_x_ptr + gx_base + 2 * D + d_off_cur, gx2.to(dt), mask=d_mask_cur)
        tl.store(grad_x_ptr + gx_base + 3 * D + d_off_cur, gx3.to(dt), mask=d_mask_cur)
        tl.store(grad_hout_ptr + gho_base + d_off_cur, gho.to(dt), mask=d_mask_cur)

        gy0_cur = gy0_nxt
        gy1_cur = gy1_nxt
        gy2_cur = gy2_nxt
        gy3_cur = gy3_nxt


@triton.autotune(
    configs=[
        # Forced-config sweep on chip 13 (T in {1024..8192}, D=3584) shows a
        # monotonic win for larger BLOCK_D; 3584 (single chunk, no D-loop) is
        # ~1.9x faster than 1024 at every T. Keep only the top few to bound
        # autotune overhead.
        triton.Config({"BLOCK_D": 1792}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_dresdpost_hc4(
    grad_y_ptr,
    x_ptr,
    h_out_ptr,
    grad_hres_ptr,
    grad_hpost_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)

    gy_base = pid_t * 4 * D
    x_base = pid_t * 4 * D
    hout_base = pid_t * D

    p0 = tl.zeros([], dtype=tl.float32)
    p1 = tl.zeros([], dtype=tl.float32)
    p2 = tl.zeros([], dtype=tl.float32)
    p3 = tl.zeros([], dtype=tl.float32)
    r00 = tl.zeros([], dtype=tl.float32)
    r01 = tl.zeros([], dtype=tl.float32)
    r02 = tl.zeros([], dtype=tl.float32)
    r03 = tl.zeros([], dtype=tl.float32)
    r10 = tl.zeros([], dtype=tl.float32)
    r11 = tl.zeros([], dtype=tl.float32)
    r12 = tl.zeros([], dtype=tl.float32)
    r13 = tl.zeros([], dtype=tl.float32)
    r20 = tl.zeros([], dtype=tl.float32)
    r21 = tl.zeros([], dtype=tl.float32)
    r22 = tl.zeros([], dtype=tl.float32)
    r23 = tl.zeros([], dtype=tl.float32)
    r30 = tl.zeros([], dtype=tl.float32)
    r31 = tl.zeros([], dtype=tl.float32)
    r32 = tl.zeros([], dtype=tl.float32)
    r33 = tl.zeros([], dtype=tl.float32)

    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy0 = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy1 = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy2 = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy3 = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        p0 += tl.sum(ho * gy0, axis=0)
        p1 += tl.sum(ho * gy1, axis=0)
        p2 += tl.sum(ho * gy2, axis=0)
        p3 += tl.sum(ho * gy3, axis=0)

        r00 += tl.sum(x0 * gy0, axis=0)
        r01 += tl.sum(x0 * gy1, axis=0)
        r02 += tl.sum(x0 * gy2, axis=0)
        r03 += tl.sum(x0 * gy3, axis=0)
        r10 += tl.sum(x1 * gy0, axis=0)
        r11 += tl.sum(x1 * gy1, axis=0)
        r12 += tl.sum(x1 * gy2, axis=0)
        r13 += tl.sum(x1 * gy3, axis=0)
        r20 += tl.sum(x2 * gy0, axis=0)
        r21 += tl.sum(x2 * gy1, axis=0)
        r22 += tl.sum(x2 * gy2, axis=0)
        r23 += tl.sum(x2 * gy3, axis=0)
        r30 += tl.sum(x3 * gy0, axis=0)
        r31 += tl.sum(x3 * gy1, axis=0)
        r32 += tl.sum(x3 * gy2, axis=0)
        r33 += tl.sum(x3 * gy3, axis=0)

    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    tl.store(grad_hpost_ptr + hpost_base + 0, p0)
    tl.store(grad_hpost_ptr + hpost_base + 1, p1)
    tl.store(grad_hpost_ptr + hpost_base + 2, p2)
    tl.store(grad_hpost_ptr + hpost_base + 3, p3)

    tl.store(grad_hres_ptr + hres_base + 0, r00)
    tl.store(grad_hres_ptr + hres_base + 1, r01)
    tl.store(grad_hres_ptr + hres_base + 2, r02)
    tl.store(grad_hres_ptr + hres_base + 3, r03)
    tl.store(grad_hres_ptr + hres_base + 4, r10)
    tl.store(grad_hres_ptr + hres_base + 5, r11)
    tl.store(grad_hres_ptr + hres_base + 6, r12)
    tl.store(grad_hres_ptr + hres_base + 7, r13)
    tl.store(grad_hres_ptr + hres_base + 8, r20)
    tl.store(grad_hres_ptr + hres_base + 9, r21)
    tl.store(grad_hres_ptr + hres_base + 10, r22)
    tl.store(grad_hres_ptr + hres_base + 11, r23)
    tl.store(grad_hres_ptr + hres_base + 12, r30)
    tl.store(grad_hres_ptr + hres_base + 13, r31)
    tl.store(grad_hres_ptr + hres_base + 14, r32)
    tl.store(grad_hres_ptr + hres_base + 15, r33)


# ---------------------------------------------------------------------------
# FUSED kernel: compute grad_x, grad_hout, grad_hres, grad_hpost in a single
# grid=(T,) launch. The two split kernels above each read the full grad_y
# (T,4,D) tensor; since this op is memory-bound, reading grad_y ONCE here cuts
# total DRAM traffic by ~22% (grad_y is 4/13 of all reads). The reduction part
# keeps the proven serial 20-scalar accumulators (tl.join batching miscompiles
# on this Ascend backend -- see note below).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 1792}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_fused_hc4(
    grad_y_ptr,
    x_ptr,
    h_res_ptr,
    h_out_ptr,
    h_post_ptr,
    grad_x_ptr,
    grad_hout_ptr,
    grad_hres_ptr,
    grad_hpost_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)

    gy_base = pid_t * 4 * D
    x_base = pid_t * 4 * D
    gx_base = pid_t * 4 * D
    hout_base = pid_t * D
    gho_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    # h_res / h_post scalars (for grad_x / grad_hout).
    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    # Reduction accumulators (grad_hpost p*, grad_hres g*).
    p0 = tl.zeros([], dtype=tl.float32)
    p1 = tl.zeros([], dtype=tl.float32)
    p2 = tl.zeros([], dtype=tl.float32)
    p3 = tl.zeros([], dtype=tl.float32)
    g00 = tl.zeros([], dtype=tl.float32)
    g01 = tl.zeros([], dtype=tl.float32)
    g02 = tl.zeros([], dtype=tl.float32)
    g03 = tl.zeros([], dtype=tl.float32)
    g10 = tl.zeros([], dtype=tl.float32)
    g11 = tl.zeros([], dtype=tl.float32)
    g12 = tl.zeros([], dtype=tl.float32)
    g13 = tl.zeros([], dtype=tl.float32)
    g20 = tl.zeros([], dtype=tl.float32)
    g21 = tl.zeros([], dtype=tl.float32)
    g22 = tl.zeros([], dtype=tl.float32)
    g23 = tl.zeros([], dtype=tl.float32)
    g30 = tl.zeros([], dtype=tl.float32)
    g31 = tl.zeros([], dtype=tl.float32)
    g32 = tl.zeros([], dtype=tl.float32)
    g33 = tl.zeros([], dtype=tl.float32)

    dt = grad_x_ptr.dtype.element_ty

    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy0 = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy1 = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy2 = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy3 = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        # ---- grad_x + grad_hout (uses grad_y only) ----
        gx0 = r00 * gy0 + r01 * gy1 + r02 * gy2 + r03 * gy3
        gx1 = r10 * gy0 + r11 * gy1 + r12 * gy2 + r13 * gy3
        gx2 = r20 * gy0 + r21 * gy1 + r22 * gy2 + r23 * gy3
        gx3 = r30 * gy0 + r31 * gy1 + r32 * gy2 + r33 * gy3
        gho = hp0 * gy0 + hp1 * gy1 + hp2 * gy2 + hp3 * gy3

        tl.store(grad_x_ptr + gx_base + 0 * D + d_off, gx0.to(dt), mask=d_mask)
        tl.store(grad_x_ptr + gx_base + 1 * D + d_off, gx1.to(dt), mask=d_mask)
        tl.store(grad_x_ptr + gx_base + 2 * D + d_off, gx2.to(dt), mask=d_mask)
        tl.store(grad_x_ptr + gx_base + 3 * D + d_off, gx3.to(dt), mask=d_mask)
        tl.store(grad_hout_ptr + gho_base + d_off, gho.to(dt), mask=d_mask)

        # ---- grad_hres + grad_hpost reductions (needs x, h_out) ----
        x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        p0 += tl.sum(ho * gy0, axis=0)
        p1 += tl.sum(ho * gy1, axis=0)
        p2 += tl.sum(ho * gy2, axis=0)
        p3 += tl.sum(ho * gy3, axis=0)

        g00 += tl.sum(x0 * gy0, axis=0)
        g01 += tl.sum(x0 * gy1, axis=0)
        g02 += tl.sum(x0 * gy2, axis=0)
        g03 += tl.sum(x0 * gy3, axis=0)
        g10 += tl.sum(x1 * gy0, axis=0)
        g11 += tl.sum(x1 * gy1, axis=0)
        g12 += tl.sum(x1 * gy2, axis=0)
        g13 += tl.sum(x1 * gy3, axis=0)
        g20 += tl.sum(x2 * gy0, axis=0)
        g21 += tl.sum(x2 * gy1, axis=0)
        g22 += tl.sum(x2 * gy2, axis=0)
        g23 += tl.sum(x2 * gy3, axis=0)
        g30 += tl.sum(x3 * gy0, axis=0)
        g31 += tl.sum(x3 * gy1, axis=0)
        g32 += tl.sum(x3 * gy2, axis=0)
        g33 += tl.sum(x3 * gy3, axis=0)

    tl.store(grad_hpost_ptr + hpost_base + 0, p0)
    tl.store(grad_hpost_ptr + hpost_base + 1, p1)
    tl.store(grad_hpost_ptr + hpost_base + 2, p2)
    tl.store(grad_hpost_ptr + hpost_base + 3, p3)

    tl.store(grad_hres_ptr + hres_base + 0, g00)
    tl.store(grad_hres_ptr + hres_base + 1, g01)
    tl.store(grad_hres_ptr + hres_base + 2, g02)
    tl.store(grad_hres_ptr + hres_base + 3, g03)
    tl.store(grad_hres_ptr + hres_base + 4, g10)
    tl.store(grad_hres_ptr + hres_base + 5, g11)
    tl.store(grad_hres_ptr + hres_base + 6, g12)
    tl.store(grad_hres_ptr + hres_base + 7, g13)
    tl.store(grad_hres_ptr + hres_base + 8, g20)
    tl.store(grad_hres_ptr + hres_base + 9, g21)
    tl.store(grad_hres_ptr + hres_base + 10, g22)
    tl.store(grad_hres_ptr + hres_base + 11, g23)
    tl.store(grad_hres_ptr + hres_base + 12, g30)
    tl.store(grad_hres_ptr + hres_base + 13, g31)
    tl.store(grad_hres_ptr + hres_base + 14, g32)
    tl.store(grad_hres_ptr + hres_base + 15, g33)


# ---------------------------------------------------------------------------
# FUSED + tl.dot variant: same single-launch fusion, but the 16 grad_hres
# reductions (grad_hres[j,i] = sum_d x[j,d]*gy[i,d] = X @ GY^T) are done with
# ONE tl.dot on the cube/matmul unit instead of 16 serial tl.sum vector
# reductions. The kernel is reduction-op-bound (measured ~314 GB/s, well under
# HBM limit), so moving the 4x4 contraction onto the cube unit is the main win.
# gy is loaded transposed via make_block_ptr (tl.trans miscompiles on this
# Ascend backend). grad_hpost keeps 4 cheap tl.sum's; grad_x/grad_hout stay
# elementwise.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 1792}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_D": 3584}, num_warps=4, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _mhc_post_bwd_fused_dot_hc4(
    grad_y_ptr,
    x_ptr,
    h_res_ptr,
    h_out_ptr,
    h_post_ptr,
    grad_x_ptr,
    grad_hout_ptr,
    grad_hres_ptr,
    grad_hpost_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)

    gy_base = pid_t * 4 * D
    x_base = pid_t * 4 * D
    gx_base = pid_t * 4 * D
    hout_base = pid_t * D
    gho_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    r00 = tl.load(h_res_ptr + hres_base + 0)
    r01 = tl.load(h_res_ptr + hres_base + 1)
    r02 = tl.load(h_res_ptr + hres_base + 2)
    r03 = tl.load(h_res_ptr + hres_base + 3)
    r10 = tl.load(h_res_ptr + hres_base + 4)
    r11 = tl.load(h_res_ptr + hres_base + 5)
    r12 = tl.load(h_res_ptr + hres_base + 6)
    r13 = tl.load(h_res_ptr + hres_base + 7)
    r20 = tl.load(h_res_ptr + hres_base + 8)
    r21 = tl.load(h_res_ptr + hres_base + 9)
    r22 = tl.load(h_res_ptr + hres_base + 10)
    r23 = tl.load(h_res_ptr + hres_base + 11)
    r30 = tl.load(h_res_ptr + hres_base + 12)
    r31 = tl.load(h_res_ptr + hres_base + 13)
    r32 = tl.load(h_res_ptr + hres_base + 14)
    r33 = tl.load(h_res_ptr + hres_base + 15)

    hp0 = tl.load(h_post_ptr + hpost_base + 0)
    hp1 = tl.load(h_post_ptr + hpost_base + 1)
    hp2 = tl.load(h_post_ptr + hpost_base + 2)
    hp3 = tl.load(h_post_ptr + hpost_base + 3)

    p0 = tl.zeros([], dtype=tl.float32)
    p1 = tl.zeros([], dtype=tl.float32)
    p2 = tl.zeros([], dtype=tl.float32)
    p3 = tl.zeros([], dtype=tl.float32)
    acc = tl.zeros([4, 4], dtype=tl.float32)

    dt = grad_x_ptr.dtype.element_ty

    # x tile viewed as (4, D) row-major -> (4, BLOCK_D) natural block.
    x_bp = tl.make_block_ptr(
        base=x_ptr + x_base,
        shape=(4, D),
        strides=(D, 1),
        offsets=(0, 0),
        block_shape=(4, BLOCK_D),
        order=(1, 0),
    )
    # gy tile viewed transposed as (D, 4): element [d, i] = gy[i, d] at i*D+d,
    # so strides over (d, i) are (1, D). Gives the (BLOCK_D, 4) rhs for tl.dot.
    gyt_bp = tl.make_block_ptr(
        base=grad_y_ptr + gy_base,
        shape=(D, 4),
        strides=(1, D),
        offsets=(0, 0),
        block_shape=(BLOCK_D, 4),
        order=(0, 1),
    )

    for d_start in range(0, D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy0 = tl.load(grad_y_ptr + gy_base + 0 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy1 = tl.load(grad_y_ptr + gy_base + 1 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy2 = tl.load(grad_y_ptr + gy_base + 2 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        gy3 = tl.load(grad_y_ptr + gy_base + 3 * D + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        gx0 = r00 * gy0 + r01 * gy1 + r02 * gy2 + r03 * gy3
        gx1 = r10 * gy0 + r11 * gy1 + r12 * gy2 + r13 * gy3
        gx2 = r20 * gy0 + r21 * gy1 + r22 * gy2 + r23 * gy3
        gx3 = r30 * gy0 + r31 * gy1 + r32 * gy2 + r33 * gy3
        gho = hp0 * gy0 + hp1 * gy1 + hp2 * gy2 + hp3 * gy3

        tl.store(grad_x_ptr + gx_base + 0 * D + d_off, gx0.to(dt), mask=d_mask)
        tl.store(grad_x_ptr + gx_base + 1 * D + d_off, gx1.to(dt), mask=d_mask)
        tl.store(grad_x_ptr + gx_base + 2 * D + d_off, gx2.to(dt), mask=d_mask)
        tl.store(grad_x_ptr + gx_base + 3 * D + d_off, gx3.to(dt), mask=d_mask)
        tl.store(grad_hout_ptr + gho_base + d_off, gho.to(dt), mask=d_mask)

        ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        p0 += tl.sum(ho * gy0, axis=0)
        p1 += tl.sum(ho * gy1, axis=0)
        p2 += tl.sum(ho * gy2, axis=0)
        p3 += tl.sum(ho * gy3, axis=0)

        # grad_hres 4x4 via one matmul: X(4,BLOCK_D) @ GY^T(BLOCK_D,4).
        x_tile = tl.load(x_bp, boundary_check=(1,), padding_option="zero").to(
            tl.float32
        )
        gyt_tile = tl.load(gyt_bp, boundary_check=(0,), padding_option="zero").to(
            tl.float32
        )
        acc += tl.dot(x_tile, gyt_tile)

        x_bp = tl.advance(x_bp, (0, BLOCK_D))
        gyt_bp = tl.advance(gyt_bp, (BLOCK_D, 0))

    tl.store(grad_hpost_ptr + hpost_base + 0, p0)
    tl.store(grad_hpost_ptr + hpost_base + 1, p1)
    tl.store(grad_hpost_ptr + hpost_base + 2, p2)
    tl.store(grad_hpost_ptr + hpost_base + 3, p3)

    hres_idx = tl.arange(0, 4)[:, None] * 4 + tl.arange(0, 4)[None, :]
    tl.store(grad_hres_ptr + hres_base + hres_idx, acc)


# ---------------------------------------------------------------------------
# tl.join batched-accumulator experiment REMOVED (dead end, probe-verified):
# batching the 4 gy rows via join -> (BLOCK_D,2,2) then 5 axis-0 sums instead
# of 20 scalar sums MISCOMPILES on Ascend. Isolation probes showed a single
# loop-carried (2,2) accumulator off a joined tensor is correct, but TWO
# accumulators off the same joined tensor already return garbage (backend
# register-allocation bug), and wider joins overflow UB. tl.trans / tl.permute
# also produce garbage on this backend, so there is no layout workaround.
# The serial 20-scalar kernel below is the fastest correct option.
# ---------------------------------------------------------------------------
# Split-D partials experiment REMOVED: proven 2.7x SLOWER than the serial
# kernel (T=2048: 3.06ms vs 1.20ms; T=8192: 12.04ms vs 4.75ms). The kernel is
# reduction-op-bound, so 7x more programs only multiplies fixed overhead and
# GM traffic without reducing per-program work enough. Also re-introduced the
# coreDim=65535 crash at T=16384 (grid=(16384, 7)).
# ---------------------------------------------------------------------------


def _flatten(grad_y, x, h_res, h_out, h_post):
    if grad_y.dim() == 4:
        B, S, N, D = grad_y.shape
        T = B * S
        shape4 = (B, S, N, D)
        shape_out = (B, S, D)
        return (
            grad_y.reshape(T, N, D).contiguous(),
            x.reshape(T, N, D).contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            shape4,
            shape_out,
        )
    if grad_y.dim() == 3:
        T, N, D = grad_y.shape
        return (
            grad_y.contiguous(),
            x.contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            (T, N, D),
            (T, D),
        )
    raise ValueError(f"unsupported grad_y.dim()={grad_y.dim()}")


def mhc_post_backward(grad_y, x, h_res, h_out, h_post):
    gy, xf, hres, hout, hpost, shape4, shape_out = _flatten(
        grad_y, x, h_res, h_out, h_post
    )
    T, N, D = gy.shape

    grad_x = torch.empty_like(xf)
    grad_hout = torch.empty_like(hout)
    grad_hres = torch.empty_like(hres)
    grad_hpost = torch.empty_like(hpost)

    if N == 4:
        # Single fused kernel: reads grad_y once (vs twice for the split
        # dxdho+dresdpost path) to cut DRAM traffic on this memory-bound op.
        # (A tl.dot variant for grad_hres was ~20x SLOWER: a 4x4 output with
        # K=3584 wastes the cube unit; see _mhc_post_bwd_fused_dot_hc4 note.)
        _mhc_post_bwd_fused_hc4[(T,)](
            gy,
            xf,
            hres,
            hout,
            hpost,
            grad_x,
            grad_hout,
            grad_hres,
            grad_hpost,
            D=D,
        )
    else:
        raise NotImplementedError("Only N=4 supported")

    return (
        grad_x.reshape(shape4),
        grad_hres.reshape(h_res.shape),
        grad_hout.reshape(shape_out),
        grad_hpost.reshape(h_post.shape),
    )


def mhc_post_backward_ref(grad_y, x, h_res, h_out, h_post):
    orig_dtype = x.dtype
    gy, xf, hres, hout, hpost, shape4, shape_out = _flatten(
        grad_y, x, h_res, h_out, h_post
    )
    gy_f = gy.float()
    x_f = xf.float()
    ho_f = hout.float()

    grad_x = torch.einsum("tji,tid->tjd", hres, gy_f).to(orig_dtype)
    grad_hres = torch.einsum("tjd,tid->tji", x_f, gy_f)
    grad_hout = torch.einsum("ti,tid->td", hpost, gy_f).to(orig_dtype)
    grad_hpost = torch.einsum("td,tid->ti", ho_f, gy_f)

    return (
        grad_x.reshape(shape4),
        grad_hres.reshape(h_res.shape),
        grad_hout.reshape(shape_out),
        grad_hpost.reshape(h_post.shape),
    )
