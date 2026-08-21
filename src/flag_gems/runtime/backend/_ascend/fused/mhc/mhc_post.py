"""TLE (Triton Language Extensions) implementation of mhc_post.

1:1 port of the AscendC operator at
``/data/liuhy/ops-transformer/mhc/mhc_post`` (``op_kernel/mhc_post.h``)
written in Triton using the TLE DSA surface documented under
``/data/liuhy/flir/flagtree/documents/tle``.

Semantics (both AscendC and this file)::

    output[t, i, d] = sum_j h_res[t, j, i] * x[t, j, d]
                    + h_post[t, i]     * h_out[t, d]

AscendC layout -> TLE mapping
-----------------------------
    ``CopyInHOut``  (DataCopyPad 1x dNum*sizeof(T))
        ``hout_ub = tle.dsa.alloc([BLOCK_D], x_dtype, UB)``  +
        ``tle.dsa.copy(h_out_ptr + off, hout_ub, [tail_d])``.

    ``CopyInX`` with ``USE_PERMANENT_X == 1``
        One ``DataCopyPad`` with ``blockCount = n`` /
        ``srcStride = (D - dNum) * sizeof(T)`` -- the whole
        (n, dNum) x-tile is staged in UB once per d-chunk.
        TLE: a single 2D ``tle.dsa.alloc([4, BLOCK_D], ..., UB)`` +
        ``tle.dsa.copy(src_2d, x_ub, [4, tail_d])``.

    ``ComputeCopyOutAllX``
        ``Cast``        -> ``tle.dsa.to_tensor(x_ub).to(tl.float32)``
        ``Muls``        -> ``out_f32 = hout_f32 * h_post[i]``
        ``Axpy``        -> ``out_f32 += x_f32[j] * h_res[j, i]``
                           (one fused multiply-add in tensor land)
        ``Cast RINT``   -> ``out_f32.to(x_dtype)``
        ``CopyOutTile`` -> ``tle.dsa.copy(out_ub, out_ptr + off, [tail_d])``
                           via ``tle.dsa.to_buffer(out_tile, UB)``.

    Scalar ``h_res`` / ``h_post`` GM reads (``GetValue`` inside the
    AscendC inner loop) become per-token scalar ``tl.load`` hoisted
    outside the D loop -- the AscendC arch35 "permanent-x" fast path
    keeps x in UB precisely so these 20 scalars amortize across all
    d-chunks; we keep the same data flow.

Grid
----
``(T, cdiv(D, BLOCK_D))``: each program owns one token's D-chunk, same
work distribution as ``tilingData_->normalCoreProcessNum`` slicing
``(bs, d-chunk)`` pairs across cores.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

try:
    import triton.experimental.tle as tle

    _HAS_DSA = hasattr(tle, "dsa") and hasattr(tle.dsa, "alloc")
except (ImportError, AttributeError):
    tle = None
    _HAS_DSA = False


# ===========================================================================
# HC_MULT=4 kernel -- maps MhcPostKernel<..., USE_PERMANENT_X=1>::Process.
# Pipeline version: 1D grid over T, iterates D-chunks internally with
# tle.dsa.pipeline(num_stages=2) for MTE2/Vector overlap via double-buffer.
# Uses tle.dsa.to_buffer + tle.dsa.copy for output (matching AscendC CopyOut).
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32   layout: h_res[t, j, i]
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: load coefficients (constant across D-chunks) ---------------
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors (reused across all D-chunks) -------------
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32

    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    # ---- Allocate double-buffered UB for x[4,BLOCK_D] and h_out[BLOCK_D] ------
    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined D-chunk loop --------------------------------------
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        # -- DMA stage: copy x[4,BLOCK_D] and h_out[BLOCK_D] from GM to UB ---
        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        # -- Compute stage: parallel FMA across 4 heads -----------------------
        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        with tle.dsa.hint(inter_no_alias=True):
            Y = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
            Y += tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)
            Y += tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)
            Y += tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)
            Y += tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)

        # -- Store: to_buffer + dsa.copy (matching AscendC CopyOutTile) -------
        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        y0b = tle.dsa.to_buffer(y0.to(x_dt), tle.dsa.ascend.UB)
        y1b = tle.dsa.to_buffer(y1.to(x_dt), tle.dsa.ascend.UB)
        y2b = tle.dsa.to_buffer(y2.to(x_dt), tle.dsa.ascend.UB)
        y3b = tle.dsa.to_buffer(y3.to(x_dt), tle.dsa.ascend.UB)
        with tle.dsa.hint(inter_no_alias=True):
            tle.dsa.copy(y0b, out_ptr + x_base + 0 * D + d_off, [tail_d])
            tle.dsa.copy(y1b, out_ptr + x_base + 1 * D + d_off, [tail_d])
            tle.dsa.copy(y2b, out_ptr + x_base + 2 * D + d_off, [tail_d])
            tle.dsa.copy(y3b, out_ptr + x_base + 3 * D + d_off, [tail_d])


# ===========================================================================
# Row-major-safe variant: builds the (4, BLOCK_D) output tile with
# tl.join on the *leading* axis so memory order matches (n, D).
# Kept separate so the compiler only sees one layout path per kernel.
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle_rows(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_off < D

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: merge all tl.load into large tensor DSA copies -----------
    # h_post[t, :] -> UB [4] fp32 (exact size, no tail issue)
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    # h_res[t, :, :] -> UB [16] fp32 (exact size, no tail issue)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors from UB ----------------------------------
    # h_post: [4] f32 — broadcast coeff for h_out
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32

    # h_res flat [16] layout: [r00,r01,r02,r03, r10,r11,r12,r13, r20,..., r30,...]
    # Row j of h_res[j,i] contains the coefficients of x[j] for all 4 output heads.
    # Extract contiguous [4] slices — one per input x row:
    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4]
    )  # [r00,r01,r02,r03]
    hr1 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4]
    )  # [r10,r11,r12,r13]
    hr2 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4]
    )  # [r20,r21,r22,r23]
    hr3 = tl.reshape(
        tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4]
    )  # [r30,r31,r32,r33]

    # ---- Load x, h_out with mask (handles non-power-of-2 D correctly) ---------
    ho = tl.load(h_out_ptr + hout_base + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + x_base + 0 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + x_base + 1 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x2 = tl.load(x_ptr + x_base + 2 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)
    x3 = tl.load(x_ptr + x_base + 3 * D + d_off, mask=d_mask, other=0.0).to(tl.float32)

    # ---- Parallel vector multiply + accumulate --------------------------------
    # Broadcast coeff[4,1] * data[1,BLOCK_D] → [4,BLOCK_D], accumulate 5 terms.
    # Each term computes contributions of one input vector to ALL 4 output heads
    # simultaneously, instead of computing each head independently.
    #   Y[i, d] = hp[i] * ho[d] + hr0[i] * x0[d] + hr1[i] * x1[d]
    #           + hr2[i] * x2[d] + hr3[i] * x3[d]
    Y = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
    Y += tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)  # + [4, BLOCK_D]
    Y += tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)  # + [4, BLOCK_D]
    Y += tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)  # + [4, BLOCK_D]
    Y += tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)  # + [4, BLOCK_D]

    # Extract output rows
    y0 = tl.reshape(tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    y1 = tl.reshape(tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    y2 = tl.reshape(tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])
    y3 = tl.reshape(tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D])

    # ---- Cast f32 -> T + store with mask ---------------------------------------
    tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
    tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
    tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
    tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


# ===========================================================================
# Pipeline variant: iterates over D-chunks inside the kernel with
# tle.dsa.pipeline for software-pipelined double-buffer overlap between
# MTE2 (load) and Vector (compute).  Grid is 1D: (T,).
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle_rows_pipeline(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: load coefficients (constant across D-chunks) ---------------
    # h_post[t, :] -> UB [4] fp32 (exact size, no tail issue)
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    # h_res[t, :, :] -> UB [16] fp32 (exact size, no tail issue)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors (reused across all D-chunks) -------------
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32

    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    # ---- Allocate double-buffered UB for x[4,BLOCK_D] and h_out[BLOCK_D] ------
    # tle.dsa.pipeline with num_stages=2 enables MTE2/V overlap via double buffer.
    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined D-chunk loop --------------------------------------
    # tle.dsa.pipeline(start, end, step, num_stages=2) generates double-buffered
    # DMA/compute overlap: while computing on chunk N, loads chunk N+1.
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        # -- DMA stage: copy x[4,BLOCK_D] and h_out[BLOCK_D] from GM to UB ---
        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        # -- Compute stage: parallel FMA across 4 heads -----------------------
        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        # Compute all 5 terms separately, then sum once
        term0 = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
        term1 = tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)  # [4, BLOCK_D]
        term2 = tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)  # [4, BLOCK_D]
        term3 = tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)  # [4, BLOCK_D]
        term4 = tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)  # [4, BLOCK_D]
        Y = term0 + term1 + term2 + term3 + term4  # [4, BLOCK_D]

        # -- Extract rows and store back to GM --------------------------------
        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


# ===========================================================================
# Concat + reduce variant: stack terms and use explicit summation.
# ===========================================================================
@triton.jit
def _mhc_post_kernel_tle_concat_reduce(
    x_ptr,  # (T, 4, D) bf16/fp16
    h_res_ptr,  # (T, 4, 4) fp32
    h_out_ptr,  # (T, D)    bf16/fp16
    h_post_ptr,  # (T, 4)    fp32
    out_ptr,  # (T, 4, D) bf16/fp16
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
):
    """Alternative implementation that computes all 5 terms separately then sums.

    Compute in FP32 for intermediate results, cast to output dtype once at the end.
    This matches the original pipeline version's precision but with better scheduling.
    """
    pid_t = tl.program_id(0)

    x_base = pid_t * 4 * D
    hout_base = pid_t * D
    hres_base = pid_t * 16
    hpost_base = pid_t * 4

    x_dt = x_ptr.dtype.element_ty

    # ---- Bulk DMA: load coefficients (constant across D-chunks) ---------------
    hpost_ub = tle.dsa.alloc([4], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)
    hres_ub = tle.dsa.alloc([16], dtype=tl.float32, mem_addr_space=tle.dsa.ascend.UB)

    with tle.dsa.hint(inter_no_alias=True):
        tle.dsa.copy(h_post_ptr + hpost_base + tl.arange(0, 4), hpost_ub, [4])
        tle.dsa.copy(h_res_ptr + hres_base + tl.arange(0, 16), hres_ub, [16])

    # ---- Extract coefficient vectors (keep in FP32) --------------------------------------
    hp = tle.dsa.to_tensor(hpost_ub)  # [4] f32
    hr_vec = tle.dsa.to_tensor(hres_ub)  # [16] f32
    hr0 = tl.reshape(tle.dsa.extract_slice(hr_vec, (0,), (4,), (1,)), [4])
    hr1 = tl.reshape(tle.dsa.extract_slice(hr_vec, (4,), (4,), (1,)), [4])
    hr2 = tl.reshape(tle.dsa.extract_slice(hr_vec, (8,), (4,), (1,)), [4])
    hr3 = tl.reshape(tle.dsa.extract_slice(hr_vec, (12,), (4,), (1,)), [4])

    # ---- Allocate double-buffered UB ----------------------------------------
    ho_ub = tle.dsa.alloc([BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)
    x_ub = tle.dsa.alloc([4, BLOCK_D], dtype=x_dt, mem_addr_space=tle.dsa.ascend.UB)

    # ---- Software-pipelined D-chunk loop --------------------------------------
    for d_chunk in tle.dsa.pipeline(0, NUM_D_BLOCKS, 1, num_stages=2):
        d_start = d_chunk * BLOCK_D
        d_off = d_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D
        tail_d = tl.minimum(BLOCK_D, D - d_start)

        # -- DMA stage: load data ---
        n_idx = tl.arange(0, 4)
        src_2d = x_ptr + x_base + n_idx[:, None] * D + d_off[None, :]
        tle.dsa.copy(src_2d, x_ub, [4, tail_d])
        tle.dsa.copy(h_out_ptr + hout_base + d_off, ho_ub, [tail_d])

        # -- Compute: compute 5 separate outer products [4, BLOCK_D] each in FP32 ---
        ho = tle.dsa.to_tensor(ho_ub).to(tl.float32)  # [BLOCK_D]
        x2d = tle.dsa.to_tensor(x_ub).to(tl.float32)  # [4, BLOCK_D]
        x0 = tl.reshape(
            tle.dsa.extract_slice(x2d, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x1 = tl.reshape(
            tle.dsa.extract_slice(x2d, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x2 = tl.reshape(
            tle.dsa.extract_slice(x2d, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        x3 = tl.reshape(
            tle.dsa.extract_slice(x2d, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        # Compute 5 terms, each is [4, BLOCK_D] in FP32
        term0 = tl.expand_dims(hp, 1) * tl.expand_dims(ho, 0)  # [4, BLOCK_D]
        term1 = tl.expand_dims(hr0, 1) * tl.expand_dims(x0, 0)  # [4, BLOCK_D]
        term2 = tl.expand_dims(hr1, 1) * tl.expand_dims(x1, 0)  # [4, BLOCK_D]
        term3 = tl.expand_dims(hr2, 1) * tl.expand_dims(x2, 0)  # [4, BLOCK_D]
        term4 = tl.expand_dims(hr3, 1) * tl.expand_dims(x3, 0)  # [4, BLOCK_D]

        # Sum all terms in FP32
        Y = term0 + term1 + term2 + term3 + term4  # [4, BLOCK_D]

        # -- Extract rows, cast once, and store ---
        y0 = tl.reshape(
            tle.dsa.extract_slice(Y, (0, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y1 = tl.reshape(
            tle.dsa.extract_slice(Y, (1, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y2 = tl.reshape(
            tle.dsa.extract_slice(Y, (2, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )
        y3 = tl.reshape(
            tle.dsa.extract_slice(Y, (3, 0), (1, BLOCK_D), (1, 1)), [BLOCK_D]
        )

        tl.store(out_ptr + x_base + 0 * D + d_off, y0.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 1 * D + d_off, y1.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 2 * D + d_off, y2.to(x_dt), mask=d_mask)
        tl.store(out_ptr + x_base + 3 * D + d_off, y3.to(x_dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _flatten_bsn(x, h_res, h_out, h_post):
    """Return (x, h_res, h_out, h_post, restore_shape) in (T, n, D) layout."""
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        x = x.reshape(T, N, D).contiguous()
        h_res = h_res.reshape(T, N, N).contiguous()
        h_out = h_out.reshape(T, D).contiguous()
        h_post = h_post.reshape(T, N).contiguous()
        return x, h_res, h_out, h_post, (B, S, N, D)
    if x.dim() == 3:
        T, N, D = x.shape
        return (
            x.contiguous(),
            h_res.reshape(T, N, N).contiguous(),
            h_out.reshape(T, D).contiguous(),
            h_post.reshape(T, N).contiguous(),
            (T, N, D),
        )
    raise ValueError(f"unsupported x.dim()={x.dim()}, expect 3 or 4")


def mhc_post(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
    use_pipeline: bool = True,
    use_concat_reduce: bool = False,
) -> torch.Tensor:
    """Fused mhc_post forward. See module docstring for semantics.

    Args:
        use_pipeline: If True (default), use the software-pipelined kernel
            that overlaps MTE2 DMA with Vector compute via double-buffer.
            If False, use the non-pipelined 2D-grid variant.
        use_concat_reduce: If True, use the concat+reduce variant that
            stacks coefficients and data into larger tensors, then uses
            element-wise multiply + tl.sum instead of separate outer products.
            Takes precedence over use_pipeline if both are True.
    """
    if not _HAS_DSA:
        raise RuntimeError(
            "This mhc_post implementation requires the TLE DSA surface "
            "(triton.experimental.tle.dsa.*)."
        )

    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    T, N, D = xf.shape
    assert N == 4, "hc_mult=4 fast path only (matches AscendC hcMult=4)."
    out = torch.empty_like(xf)

    # AscendC picks dInner from the tiling table; we mirror with a fixed
    # BLOCK_D that keeps T * cdiv(D, BLOCK_D) under the 65535 coreDim cap.
    BLOCK_D = min(1024, triton.next_power_of_2(D))
    while T * triton.cdiv(D, BLOCK_D) > 65535 and BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_D = min(BLOCK_D, triton.next_power_of_2(D))
    NUM_D_BLOCKS = triton.cdiv(D, BLOCK_D)

    if use_concat_reduce:
        # Concat+reduce variant: 1D grid over T
        grid = (T,)
        _mhc_post_kernel_tle_concat_reduce[grid](
            xf,
            hres.to(torch.float32),
            hout,
            hpost.to(torch.float32),
            out,
            D=D,
            BLOCK_D=BLOCK_D,
            NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    elif use_pipeline:
        # Pipeline kernel: 1D grid over T, iterates D-chunks internally
        # with tle.dsa.pipeline(num_stages=2) for MTE2/Vector overlap.
        grid = (T,)
        _mhc_post_kernel_tle_rows_pipeline[grid](
            xf,
            hres.to(torch.float32),
            hout,
            hpost.to(torch.float32),
            out,
            D=D,
            BLOCK_D=BLOCK_D,
            NUM_D_BLOCKS=NUM_D_BLOCKS,
        )
    else:
        # Non-pipelined: 2D grid (T, cdiv(D, BLOCK_D))
        grid = (T, NUM_D_BLOCKS)
        _mhc_post_kernel_tle_rows[grid](
            xf,
            hres.to(torch.float32),
            hout,
            hpost.to(torch.float32),
            out,
            D=D,
            BLOCK_D=BLOCK_D,
        )
    return out.reshape(shape)


def mhc_post_ref(
    x: torch.Tensor,
    h_res: torch.Tensor,
    h_out: torch.Tensor,
    h_post: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference (aclnn semantic).

    output[t,i,d] = sum_j h_res[t,j,i] * x[t,j,d]
                  + h_post[t,i] * h_out[t,d]
    """
    orig_dtype = x.dtype
    xf, hres, hout, hpost, shape = _flatten_bsn(x, h_res, h_out, h_post)
    x_f = xf.float()
    hout_f = hout.float()
    mix = torch.einsum("tji,tjd->tid", hres.float(), x_f)
    outer = hpost.float().unsqueeze(-1) * hout_f.unsqueeze(1)
    y = (mix + outer).to(orig_dtype)
    return y.reshape(shape)


__all__ = [
    "mhc_post",
    "mhc_post_ref",
    "_mhc_post_kernel_tle",
    "_mhc_post_kernel_tle_rows",
    "_mhc_post_kernel_tle_rows_pipeline",
    "_mhc_post_kernel_tle_concat_reduce",
]
