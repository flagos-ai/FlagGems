"""FlagGems-style Triton backward for mhc_pre_clamp_sinkhorn.

Design (hybrid Triton + torch.autograd)
---------------------------------------
The backward of the four-stage forward is naturally split into "big"
(D-dimensional / (T,hcMult,D) sized) and "small" ((T, 24)/(T,4,4) sized)
work. We put the D-sized work in Triton kernels for Ascend efficiency,
and let torch.autograd handle the small Sinkhorn/softmax/clamp/head-affine
backward because:

  * the small stage is a tiny fp32 tensor per token (T * 24 elements),
    so backward-through-autograd is negligible vs the D-sized work;
  * autograd removes the risk of hand-rolled Sinkhorn VJP bugs;
  * the forward already had to save `h_res_logits`, `mixes`, `pre` — reusing
    those under a `torch.enable_grad` re-invocation gives an exact
    numerical match with the FlagGems forward at zero extra memory cost.

Stages and where they run:

  Y-backward (Triton `_y_bwd_kernel_hc4`)
      grad_x_from_y[i,d] = grad_y[i,d] * pre[i]
      grad_pre_from_y[i] = sum_d grad_y[i,d] * x_orig[i,d]

  Small stage backward (torch.autograd)
      inputs   : mixes (T,24), alpha (3,), base (24,), h_res_logits (T,4,4)
      outputs  : post_out (T,4), comb_frag (T,4,4), pre (T,4)
      recompute from `mixes` (fp32), get vjp against grad_post_out /
      grad_comb_frag / grad_pre_from_y -> produces grad_mixes (T,24),
      grad_alpha (3,), grad_base (24,).

  GEMM backward (torch.mm)
      grad_x_scaled = grad_mixes @ phi
      grad_phi      = grad_mixes.T @ x_scaled

  RMSNorm backward (Triton `_rms_bwd_kernel`)
      Given grad_x_scaled (T,HC_D) fp32, inv_rms (T,) fp32,
            x_orig (T,HC_D) input dtype:
      grad_x_from_rms = inv_rms * (grad_x_scaled
                        - (1/HC_D) * inv_rms^2 * x_orig * dot(x_orig, grad_x_scaled))

  Combine:
      grad_x = grad_x_from_y + grad_x_from_rms   (in input dtype)
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    HAS_TLE = True
except ImportError:
    HAS_TLE = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Kernel: hin-backward (modified for reduced hin)
#   grad_hin is (T, D) — broadcast across N=4 heads
#   grad_x_from_hin[t, i, d] = grad_hin[t, d] * pre[t, i]
#   grad_pre_from_hin[t, i]  = sum_d grad_hin[t, d] * x[t, i, d]
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 2048}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_D": 4096}, num_warps=4, num_stages=2),
    ],
    key=["D"],
)
@triton.heuristics({"EVEN_D": lambda args: args["D"] % args["BLOCK_D"] == 0})
@triton.jit
def _y_bwd_kernel_hc4(
    grad_y_ptr,  # (T, D)    — same dtype as x (bf16/fp16)
    x_ptr,  # (T, 4, D) — input dtype
    pre_ptr,  # (T, 4)    — fp32  (raw sigmoid, NO eps)
    grad_x_from_y_ptr,  # (T, 4, D) fp32  OUTPUT
    grad_pre_from_y_ptr,  # (T, 4) fp32  OUTPUT
    # strides (element counts, not bytes)
    stride_gy_t,  # grad_y stride along T
    stride_x_t,  # x stride along T  (= 4*D if contiguous)
    stride_x_n,  # x stride along N  (= D if contiguous)
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    pid_t = tl.program_id(0)

    # Stride-based addressing: supports non-contiguous x and grad_y
    gy_base = pid_t * stride_gy_t
    x_row = pid_t * stride_x_t
    ox_base = pid_t * 4 * D  # output is always contiguous (we allocate it)

    p0 = tl.load(pre_ptr + pid_t * 4 + 0)
    p1 = tl.load(pre_ptr + pid_t * 4 + 1)
    p2 = tl.load(pre_ptr + pid_t * 4 + 2)
    p3 = tl.load(pre_ptr + pid_t * 4 + 3)

    acc0 = tl.zeros([], dtype=tl.float32)
    acc1 = tl.zeros([], dtype=tl.float32)
    acc2 = tl.zeros([], dtype=tl.float32)
    acc3 = tl.zeros([], dtype=tl.float32)

    n_full = D // BLOCK_D

    # --- Reverse order: tail block first so d_mask reg dies before main loop ---
    if not EVEN_D:
        d_tail_start = n_full * BLOCK_D
        d_off = d_tail_start + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy = tl.load(grad_y_ptr + gy_base + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x0 = tl.load(x_ptr + x_row + 0 * stride_x_n + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x1 = tl.load(x_ptr + x_row + 1 * stride_x_n + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x2 = tl.load(x_ptr + x_row + 2 * stride_x_n + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x3 = tl.load(x_ptr + x_row + 3 * stride_x_n + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        tl.store(grad_x_from_y_ptr + ox_base + 0 * D + d_off, gy * p0, mask=d_mask)
        tl.store(grad_x_from_y_ptr + ox_base + 1 * D + d_off, gy * p1, mask=d_mask)
        tl.store(grad_x_from_y_ptr + ox_base + 2 * D + d_off, gy * p2, mask=d_mask)
        tl.store(grad_x_from_y_ptr + ox_base + 3 * D + d_off, gy * p3, mask=d_mask)

        acc0 += tl.sum(gy * x0, axis=0)
        acc1 += tl.sum(gy * x1, axis=0)
        acc2 += tl.sum(gy * x2, axis=0)
        acc3 += tl.sum(gy * x3, axis=0)

    # --- Full blocks: no mask, no mask register alive ---
    for d_start in range(0, n_full * BLOCK_D, BLOCK_D):
        d_off = d_start + tl.arange(0, BLOCK_D)

        gy = tl.load(grad_y_ptr + gy_base + d_off).to(tl.float32)
        x0 = tl.load(x_ptr + x_row + 0 * stride_x_n + d_off).to(tl.float32)
        x1 = tl.load(x_ptr + x_row + 1 * stride_x_n + d_off).to(tl.float32)
        x2 = tl.load(x_ptr + x_row + 2 * stride_x_n + d_off).to(tl.float32)
        x3 = tl.load(x_ptr + x_row + 3 * stride_x_n + d_off).to(tl.float32)

        tl.store(grad_x_from_y_ptr + ox_base + 0 * D + d_off, gy * p0)
        tl.store(grad_x_from_y_ptr + ox_base + 1 * D + d_off, gy * p1)
        tl.store(grad_x_from_y_ptr + ox_base + 2 * D + d_off, gy * p2)
        tl.store(grad_x_from_y_ptr + ox_base + 3 * D + d_off, gy * p3)

        acc0 += tl.sum(gy * x0, axis=0)
        acc1 += tl.sum(gy * x1, axis=0)
        acc2 += tl.sum(gy * x2, axis=0)
        acc3 += tl.sum(gy * x3, axis=0)

    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 0, acc0)
    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 1, acc1)
    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 2, acc2)
    tl.store(grad_pre_from_y_ptr + pid_t * 4 + 3, acc3)


# ---------------------------------------------------------------------------
# Kernel: rms-backward + combine (produces grad_x in input dtype)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 512}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_H": 1024}, num_warps=8, num_stages=1),
    ],
    key=["HC_D"],
)
@triton.jit
def _rms_bwd_combine_kernel(
    grad_x_scaled_ptr,  # (T, HC_D) fp32
    grad_x_from_y_ptr,  # (T, HC_D) fp32
    inv_rms_ptr,  # (T,) fp32
    x_orig_ptr,  # (T, HC_D) input dtype
    grad_x_ptr,  # (T, HC_D) input dtype  OUTPUT
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,  # 1 / HC_D
    BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    inv = tl.load(inv_rms_ptr + pid)
    base = pid * HC_D

    # dot(x_orig, grad_x_scaled)  (fp32 accumulator)
    dot = 0.0
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        xv = tl.load(x_orig_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        gv = tl.load(grad_x_scaled_ptr + base + offs, mask=mask, other=0.0)
        dot += tl.sum(xv * gv, axis=0)

    # scale = -D_INV * inv_rms^3 * dot
    scale = -D_INV * inv * inv * inv * dot

    dt = grad_x_ptr.dtype.element_ty
    for h_start in range(0, HC_D, BLOCK_H):
        offs = h_start + tl.arange(0, BLOCK_H)
        mask = offs < HC_D
        xv = tl.load(x_orig_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        gs = tl.load(grad_x_scaled_ptr + base + offs, mask=mask, other=0.0)
        gy = tl.load(grad_x_from_y_ptr + base + offs, mask=mask, other=0.0)
        # d(x_scaled)/dx = inv * I + x_orig * d(inv)/dx    (d(inv)/dx = -D_INV*inv^3*x_orig)
        # so grad_x_from_rms = inv * grad_x_scaled + scale * x_orig
        gx = gy + inv * gs + scale * xv
        tl.store(grad_x_ptr + base + offs, gx.to(dt), mask=mask)


# ---------------------------------------------------------------------------
# Fused kernel: y-backward + rms-backward + combine
#   Eliminates the ~80MB intermediate grad_x_from_y write+read.
#   One program per token. Two passes over HC_D:
#     Pass 1: accumulate dot(x_orig, grad_x_scaled) and grad_pre[n]
#     Pass 2: compute final grad_x = grad_x_from_y + inv*grad_x_scaled + scale*x_orig
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_D": 4096}, num_warps=32, num_stages=2),
    ],
    key=["D"],
)
@triton.jit
def _fused_ybwd_rmsbwd_kernel(
    grad_y_ptr,  # (T, D) input dtype  -- grad of hin
    x_orig_ptr,  # (T, 4, D) input dtype  -- original x (may be non-contiguous)
    grad_x_scaled_ptr,  # (T, 4*D) fp32  -- grad from GEMM bwd (always contiguous)
    inv_rms_ptr,  # (T,) fp32
    pre_ptr,  # (T, 4) fp32
    grad_x_ptr,  # (T, 4*D) input dtype  OUTPUT (always contiguous)
    grad_pre_ptr,  # (T, 4) fp32  OUTPUT
    stride_gy_t,  # grad_y stride along T (elements) — supports non-contiguous
    stride_x_t,  # x stride along T (elements)
    stride_x_n,  # x stride along N (elements)
    D: tl.constexpr,
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused y-backward + RMS-backward kernel with TLE async loads.

    Iterates over 4 heads x D-blocks explicitly to avoid expensive
    modulo/division/where operations that dominated the previous version.
    Two passes over the data:
      Pass 1: dot = sum(x * grad_x_scaled), grad_pre[n] = sum(gy * x[n])
      Pass 2: grad_x = gy*pre[n] + inv*gs + scale*x

    Input x and grad_y are accessed via strides to avoid any .contiguous() copy
    even when the caller passes a (B,S,N,D) or (B,S,D) tensor directly.
    Output grad_x is always freshly-allocated contiguous (T, 4*D).

    TLE async loads are applied to all tensor loads to enable prefetching.
    BLOCK_D is autotuned for optimal performance.
    """
    pid = tl.program_id(0)

    # Scalar loads with async hint (TLE required)
    inv = tle.load(inv_rms_ptr + pid, is_async=True)

    ix_base = pid * stride_x_t  # input x base: stride-based (non-contiguous ok)
    ox_base = pid * 4 * D  # output grad_x base: always contiguous
    base_gs = pid * HC_D  # grad_x_scaled base: always contiguous
    igy_base = pid * stride_gy_t  # input grad_y base: stride-based

    # Load pre values once with async hint
    p0 = tle.load(pre_ptr + pid * 4 + 0, is_async=True)
    p1 = tle.load(pre_ptr + pid * 4 + 1, is_async=True)
    p2 = tle.load(pre_ptr + pid * 4 + 2, is_async=True)
    p3 = tle.load(pre_ptr + pid * 4 + 3, is_async=True)

    # Pass 1: accumulate dot and grad_pre
    dot = 0.0
    gp0 = tl.zeros([], dtype=tl.float32)
    gp1 = tl.zeros([], dtype=tl.float32)
    gp2 = tl.zeros([], dtype=tl.float32)
    gp3 = tl.zeros([], dtype=tl.float32)

    NUM_D_BLOCKS = tl.cdiv(D, BLOCK_D)
    for db in range(NUM_D_BLOCKS):
        d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        # stride-based load with async hint: supports non-contiguous grad_y and x
        gy = tle.load(
            grad_y_ptr + igy_base + d_off, mask=d_mask, other=0.0, is_async=True
        ).to(tl.float32)

        x0 = tle.load(
            x_orig_ptr + ix_base + 0 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs0 = tle.load(
            grad_x_scaled_ptr + base_gs + 0 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        dot += tl.sum(x0 * gs0, axis=0)
        gp0 += tl.sum(gy * x0, axis=0)

        x1 = tle.load(
            x_orig_ptr + ix_base + 1 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs1 = tle.load(
            grad_x_scaled_ptr + base_gs + 1 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        dot += tl.sum(x1 * gs1, axis=0)
        gp1 += tl.sum(gy * x1, axis=0)

        x2 = tle.load(
            x_orig_ptr + ix_base + 2 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs2 = tle.load(
            grad_x_scaled_ptr + base_gs + 2 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        dot += tl.sum(x2 * gs2, axis=0)
        gp2 += tl.sum(gy * x2, axis=0)

        x3 = tle.load(
            x_orig_ptr + ix_base + 3 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs3 = tle.load(
            grad_x_scaled_ptr + base_gs + 3 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        dot += tl.sum(x3 * gs3, axis=0)
        gp3 += tl.sum(gy * x3, axis=0)

    # Store grad_pre
    tl.store(grad_pre_ptr + pid * 4 + 0, gp0)
    tl.store(grad_pre_ptr + pid * 4 + 1, gp1)
    tl.store(grad_pre_ptr + pid * 4 + 2, gp2)
    tl.store(grad_pre_ptr + pid * 4 + 3, gp3)

    # Compute scale for RMS backward
    scale = -D_INV * inv * inv * inv * dot

    # Pass 2: compute final grad_x (output is always contiguous → ox_base)
    dt = grad_x_ptr.dtype.element_ty
    for db in range(NUM_D_BLOCKS):
        d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy = tle.load(
            grad_y_ptr + igy_base + d_off, mask=d_mask, other=0.0, is_async=True
        ).to(tl.float32)

        x0 = tle.load(
            x_orig_ptr + ix_base + 0 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs0 = tle.load(
            grad_x_scaled_ptr + base_gs + 0 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        gx0 = gy * p0 + inv * gs0 + scale * x0
        tl.store(grad_x_ptr + ox_base + 0 * D + d_off, gx0.to(dt), mask=d_mask)

        x1 = tle.load(
            x_orig_ptr + ix_base + 1 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs1 = tle.load(
            grad_x_scaled_ptr + base_gs + 1 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        gx1 = gy * p1 + inv * gs1 + scale * x1
        tl.store(grad_x_ptr + ox_base + 1 * D + d_off, gx1.to(dt), mask=d_mask)

        x2 = tle.load(
            x_orig_ptr + ix_base + 2 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs2 = tle.load(
            grad_x_scaled_ptr + base_gs + 2 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        gx2 = gy * p2 + inv * gs2 + scale * x2
        tl.store(grad_x_ptr + ox_base + 2 * D + d_off, gx2.to(dt), mask=d_mask)

        x3 = tle.load(
            x_orig_ptr + ix_base + 3 * stride_x_n + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        ).to(tl.float32)
        gs3 = tle.load(
            grad_x_scaled_ptr + base_gs + 3 * D + d_off,
            mask=d_mask,
            other=0.0,
            is_async=True,
        )
        gx3 = gy * p3 + inv * gs3 + scale * x3
        tl.store(grad_x_ptr + ox_base + 3 * D + d_off, gx3.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Custom skinny GEMM kernel for (T, 24) @ (24, HC_D)
#   K=24 is too small for Cube core. This kernel loads the full K=24 in
#   registers and does the dot product per (BLOCK_T, BLOCK_D) output tile.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 4, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 4, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 4, "BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_T": 8, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 8, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 8, "BLOCK_D": 512}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 256}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_T": 16, "BLOCK_D": 512}, num_warps=8, num_stages=1),
    ],
    key=["T", "HC_D", "K"],
)
@triton.jit
def _skinny_gemm_kernel(
    a_ptr,  # (T, K) fp32  -- grad_mixes
    b_ptr,  # (K, HC_D) fp32  -- phi
    c_ptr,  # (T, HC_D) fp32  -- output grad_x_scaled
    T: tl.constexpr,
    HC_D: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_d = tl.program_id(1)

    t_off = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_off = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    t_mask = t_off < T
    d_mask = d_off < HC_D

    # Accumulator: (BLOCK_T, BLOCK_D) fp32
    acc = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for k in range(K):
        # Load a column: a[t_off, k] -> (BLOCK_T,)
        a_val = tl.load(a_ptr + t_off * K + k, mask=t_mask, other=0.0)  # (BLOCK_T,)
        # Load b row: b[k, d_off] -> (BLOCK_D,)
        b_val = tl.load(b_ptr + k * HC_D + d_off, mask=d_mask, other=0.0)  # (BLOCK_D,)
        # Outer product
        acc += a_val[:, None] * b_val[None, :]

    # Store result
    out_offs = t_off[:, None] * HC_D + d_off[None, :]
    out_mask = t_mask[:, None] & d_mask[None, :]
    tl.store(c_ptr + out_offs, acc, mask=out_mask)


# ---------------------------------------------------------------------------
# Fused kernel: skinny-GEMM + RMS-backward + Y-backward in one pass.
#   Eliminates the grad_x_scaled intermediate tensor entirely.
#   For each D-block, computes grad_x_scaled on-the-fly from grad_mixes @ phi,
#   then immediately uses it for RMS backward.
#   Two passes over D: Pass1 computes dot, Pass2 computes grad_x.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gemm_rmsbwd_ybwd_kernel(
    grad_y_ptr,  # (T, D) input dtype
    x_orig_ptr,  # (T, 4, D) input dtype
    grad_mixes_ptr,  # (T, 24) fp32  — from small-stage bwd
    phi_ptr,  # (24, HC_D) fp32
    inv_rms_ptr,  # (T,) fp32
    pre_ptr,  # (T, 4) fp32
    grad_x_ptr,  # (T, 4*D) input dtype  OUTPUT
    grad_pre_ptr,  # (T, 4) fp32  OUTPUT (grad_pre_from_y)
    stride_gy_t,
    stride_x_t,
    stride_x_n,
    D: tl.constexpr,
    HC_D: tl.constexpr,
    D_INV: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NUM_D_BLOCKS: tl.constexpr,
    K: tl.constexpr,  # =24
):
    """Fused skinny-GEMM + Y-bwd + RMS-bwd kernel.

    Per-token: computes grad_x_scaled = grad_mixes @ phi on-the-fly,
    fuses with RMS-backward and Y-backward to produce final grad_x.
    Eliminates the T*HC_D*4B grad_x_scaled intermediate write+read.

    Two passes over D (same structure as _fused_ybwd_rmsbwd_kernel):
      Pass 1: for each D-block, compute grad_x_scaled chunk via K=24 outer-product,
              accumulate dot(x, grad_x_scaled) and grad_pre[n] = sum(gy * x[n])
      Pass 2: recompute grad_x_scaled chunk, compute final grad_x
    """
    pid = tl.program_id(0)
    inv = tl.load(inv_rms_ptr + pid)
    ix_base = pid * stride_x_t
    ox_base = pid * 4 * D
    igy_base = pid * stride_gy_t
    gm_base = pid * K  # grad_mixes base

    # Load pre values
    p0 = tl.load(pre_ptr + pid * 4 + 0)
    p1 = tl.load(pre_ptr + pid * 4 + 1)
    p2 = tl.load(pre_ptr + pid * 4 + 2)
    p3 = tl.load(pre_ptr + pid * 4 + 3)

    # Pass 1: accumulate dot(x, grad_x_scaled) and grad_pre
    dot = 0.0
    gp0 = tl.zeros([], dtype=tl.float32)
    gp1 = tl.zeros([], dtype=tl.float32)
    gp2 = tl.zeros([], dtype=tl.float32)
    gp3 = tl.zeros([], dtype=tl.float32)

    for db in range(NUM_D_BLOCKS):
        d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy = tl.load(grad_y_ptr + igy_base + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        # Compute grad_x_scaled for 4 heads at this D-block via skinny GEMM
        # grad_x_scaled[n*D+d] = sum_k grad_mixes[k] * phi[k, n*D+d]
        # Unroll k=0..23 manually for each head
        for head in tl.static_range(4):
            hd_off = head * D + d_off  # offset into HC_D dimension
            # Compute skinny GEMM for this head's D-block
            gs = tl.zeros([BLOCK_D], dtype=tl.float32)
            for k in range(K):
                phi_val = tl.load(phi_ptr + k * HC_D + hd_off, mask=d_mask, other=0.0)
                gm_k = tl.load(grad_mixes_ptr + gm_base + k)
                gs += gm_k * phi_val
            # Load x for this head
            x_val = tl.load(
                x_orig_ptr + ix_base + head * stride_x_n + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            dot += tl.sum(x_val * gs, axis=0)
            # grad_pre accumulation
            if head == 0:
                gp0 += tl.sum(gy * x_val, axis=0)
            elif head == 1:
                gp1 += tl.sum(gy * x_val, axis=0)
            elif head == 2:
                gp2 += tl.sum(gy * x_val, axis=0)
            else:
                gp3 += tl.sum(gy * x_val, axis=0)

    # Store grad_pre
    tl.store(grad_pre_ptr + pid * 4 + 0, gp0)
    tl.store(grad_pre_ptr + pid * 4 + 1, gp1)
    tl.store(grad_pre_ptr + pid * 4 + 2, gp2)
    tl.store(grad_pre_ptr + pid * 4 + 3, gp3)

    # Compute RMS scale
    scale = -D_INV * inv * inv * inv * dot

    # Pass 2: recompute grad_x_scaled, combine with y-bwd, produce grad_x
    dt = grad_x_ptr.dtype.element_ty
    for db in range(NUM_D_BLOCKS):
        d_off = db * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_off < D

        gy = tl.load(grad_y_ptr + igy_base + d_off, mask=d_mask, other=0.0).to(
            tl.float32
        )

        for head in tl.static_range(4):
            hd_off = head * D + d_off
            # Recompute skinny GEMM
            gs = tl.zeros([BLOCK_D], dtype=tl.float32)
            for k in range(K):
                phi_val = tl.load(phi_ptr + k * HC_D + hd_off, mask=d_mask, other=0.0)
                gm_k = tl.load(grad_mixes_ptr + gm_base + k)
                gs += gm_k * phi_val
            # Load x
            x_val = tl.load(
                x_orig_ptr + ix_base + head * stride_x_n + d_off, mask=d_mask, other=0.0
            ).to(tl.float32)
            # pre value for this head
            if head == 0:
                pre_h = p0
            elif head == 1:
                pre_h = p1
            elif head == 2:
                pre_h = p2
            else:
                pre_h = p3
            # grad_x = gy * pre + inv * grad_x_scaled + scale * x
            gx = gy * pre_h + inv * gs + scale * x_val
            tl.store(grad_x_ptr + ox_base + head * D + d_off, gx.to(dt), mask=d_mask)


# ---------------------------------------------------------------------------
# Fused small-stage backward Triton kernel.
#
# One program per token. Recomputes the sinkhorn forward, saving each
# intermediate M state (16 fp32) to a scratch buffer, then walks backward
# through row/col-norm iterations, softmax, clamp, pre/post sigmoids.
#
# Emits:
#   grad_mixes       (T, 24)  fp32
#   grad_alpha_part  (T, 3)   fp32  -- reduced to (3,) outside
#   grad_base_part   (T, 24)  fp32  -- reduced to (24,) outside
#
# scratch layout: fp32 tensor of shape (T, ITERS, 16), holds M at end of
#   each iter (before the +hc_eps that starts the next iter's col-norm add).
# ---------------------------------------------------------------------------
@triton.jit
def _small_stage_bwd_kernel_hc4(
    mixes_ptr,  # (T, 24) fp32
    alpha_ptr,  # (3,)    fp32
    base_ptr,  # (24,)   fp32
    h_res_logits_ptr,  # (T, 4, 4) fp32   pre-clamp logits from fwd (unused here, recomputed)
    grad_pre_ptr,  # (T, 4)   fp32
    grad_post_ptr,  # (T, 4)   fp32
    grad_comb_ptr,  # (T, 4, 4) fp32
    grad_mixes_ptr,  # (T, 24)  fp32     OUT
    grad_alpha_part_ptr,  # (T, 3)   fp32     OUT (per-token partial)
    grad_base_part_ptr,  # (T, 24)  fp32     OUT (per-token partial)
    scratch_ptr,  # (T, ITERS, 16) fp32  (M state at end of each iter)
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    a0 = tl.load(alpha_ptr + 0)
    a1 = tl.load(alpha_ptr + 1)
    a2 = tl.load(alpha_ptr + 2)
    mb = pid * 24
    pb = pid * 4
    cb = pid * 16
    sb = pid * ITERS * 16

    # ---- FORWARD RECOMPUTE (pre / post / logits / sinkhorn) ----
    # Load per-token mixes/base slices
    mix_pre0 = tl.load(mixes_ptr + mb + 0)
    mix_pre1 = tl.load(mixes_ptr + mb + 1)
    mix_pre2 = tl.load(mixes_ptr + mb + 2)
    mix_pre3 = tl.load(mixes_ptr + mb + 3)
    b_pre0 = tl.load(base_ptr + 0)
    b_pre1 = tl.load(base_ptr + 1)
    b_pre2 = tl.load(base_ptr + 2)
    b_pre3 = tl.load(base_ptr + 3)
    # z_pre = m*a0 + b   -> pre_out = sigmoid(z_pre) + eps
    z_pre0 = mix_pre0 * a0 + b_pre0
    z_pre1 = mix_pre1 * a0 + b_pre1
    z_pre2 = mix_pre2 * a0 + b_pre2
    z_pre3 = mix_pre3 * a0 + b_pre3
    sp0 = tl.sigmoid(z_pre0)
    sp1 = tl.sigmoid(z_pre1)
    sp2 = tl.sigmoid(z_pre2)
    sp3 = tl.sigmoid(z_pre3)

    mix_po0 = tl.load(mixes_ptr + mb + 4)
    mix_po1 = tl.load(mixes_ptr + mb + 5)
    mix_po2 = tl.load(mixes_ptr + mb + 6)
    mix_po3 = tl.load(mixes_ptr + mb + 7)
    b_po0 = tl.load(base_ptr + 4)
    b_po1 = tl.load(base_ptr + 5)
    b_po2 = tl.load(base_ptr + 6)
    b_po3 = tl.load(base_ptr + 7)
    z_po0 = mix_po0 * a1 + b_po0
    z_po1 = mix_po1 * a1 + b_po1
    z_po2 = mix_po2 * a1 + b_po2
    z_po3 = mix_po3 * a1 + b_po3
    spo0 = tl.sigmoid(z_po0)
    spo1 = tl.sigmoid(z_po1)
    spo2 = tl.sigmoid(z_po2)
    spo3 = tl.sigmoid(z_po3)

    # logits: 16 values
    ml00 = tl.load(mixes_ptr + mb + 8)
    ml01 = tl.load(mixes_ptr + mb + 9)
    ml02 = tl.load(mixes_ptr + mb + 10)
    ml03 = tl.load(mixes_ptr + mb + 11)
    ml10 = tl.load(mixes_ptr + mb + 12)
    ml11 = tl.load(mixes_ptr + mb + 13)
    ml12 = tl.load(mixes_ptr + mb + 14)
    ml13 = tl.load(mixes_ptr + mb + 15)
    ml20 = tl.load(mixes_ptr + mb + 16)
    ml21 = tl.load(mixes_ptr + mb + 17)
    ml22 = tl.load(mixes_ptr + mb + 18)
    ml23 = tl.load(mixes_ptr + mb + 19)
    ml30 = tl.load(mixes_ptr + mb + 20)
    ml31 = tl.load(mixes_ptr + mb + 21)
    ml32 = tl.load(mixes_ptr + mb + 22)
    ml33 = tl.load(mixes_ptr + mb + 23)
    bl00 = tl.load(base_ptr + 8)
    bl01 = tl.load(base_ptr + 9)
    bl02 = tl.load(base_ptr + 10)
    bl03 = tl.load(base_ptr + 11)
    bl10 = tl.load(base_ptr + 12)
    bl11 = tl.load(base_ptr + 13)
    bl12 = tl.load(base_ptr + 14)
    bl13 = tl.load(base_ptr + 15)
    bl20 = tl.load(base_ptr + 16)
    bl21 = tl.load(base_ptr + 17)
    bl22 = tl.load(base_ptr + 18)
    bl23 = tl.load(base_ptr + 19)
    bl30 = tl.load(base_ptr + 20)
    bl31 = tl.load(base_ptr + 21)
    bl32 = tl.load(base_ptr + 22)
    bl33 = tl.load(base_ptr + 23)
    # pre-clamp logits (needed for clamp-mask on bwd)
    l00 = ml00 * a2 + bl00
    l01 = ml01 * a2 + bl01
    l02 = ml02 * a2 + bl02
    l03 = ml03 * a2 + bl03
    l10 = ml10 * a2 + bl10
    l11 = ml11 * a2 + bl11
    l12 = ml12 * a2 + bl12
    l13 = ml13 * a2 + bl13
    l20 = ml20 * a2 + bl20
    l21 = ml21 * a2 + bl21
    l22 = ml22 * a2 + bl22
    l23 = ml23 * a2 + bl23
    l30 = ml30 * a2 + bl30
    l31 = ml31 * a2 + bl31
    l32 = ml32 * a2 + bl32
    l33 = ml33 * a2 + bl33
    # post-clamp
    if APPLY_CLAMP:
        c00 = tl.minimum(tl.maximum(l00, CLAMP_MIN), CLAMP_MAX)
        c01 = tl.minimum(tl.maximum(l01, CLAMP_MIN), CLAMP_MAX)
        c02 = tl.minimum(tl.maximum(l02, CLAMP_MIN), CLAMP_MAX)
        c03 = tl.minimum(tl.maximum(l03, CLAMP_MIN), CLAMP_MAX)
        c10 = tl.minimum(tl.maximum(l10, CLAMP_MIN), CLAMP_MAX)
        c11 = tl.minimum(tl.maximum(l11, CLAMP_MIN), CLAMP_MAX)
        c12 = tl.minimum(tl.maximum(l12, CLAMP_MIN), CLAMP_MAX)
        c13 = tl.minimum(tl.maximum(l13, CLAMP_MIN), CLAMP_MAX)
        c20 = tl.minimum(tl.maximum(l20, CLAMP_MIN), CLAMP_MAX)
        c21 = tl.minimum(tl.maximum(l21, CLAMP_MIN), CLAMP_MAX)
        c22 = tl.minimum(tl.maximum(l22, CLAMP_MIN), CLAMP_MAX)
        c23 = tl.minimum(tl.maximum(l23, CLAMP_MIN), CLAMP_MAX)
        c30 = tl.minimum(tl.maximum(l30, CLAMP_MIN), CLAMP_MAX)
        c31 = tl.minimum(tl.maximum(l31, CLAMP_MIN), CLAMP_MAX)
        c32 = tl.minimum(tl.maximum(l32, CLAMP_MIN), CLAMP_MAX)
        c33 = tl.minimum(tl.maximum(l33, CLAMP_MIN), CLAMP_MAX)
    else:
        c00 = l00
        c01 = l01
        c02 = l02
        c03 = l03
        c10 = l10
        c11 = l11
        c12 = l12
        c13 = l13
        c20 = l20
        c21 = l21
        c22 = l22
        c23 = l23
        c30 = l30
        c31 = l31
        c32 = l32
        c33 = l33

    # softmax rows
    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))
    e00 = tl.exp(c00 - m0)
    e01 = tl.exp(c01 - m0)
    e02 = tl.exp(c02 - m0)
    e03 = tl.exp(c03 - m0)
    e10 = tl.exp(c10 - m1)
    e11 = tl.exp(c11 - m1)
    e12 = tl.exp(c12 - m1)
    e13 = tl.exp(c13 - m1)
    e20 = tl.exp(c20 - m2)
    e21 = tl.exp(c21 - m2)
    e22 = tl.exp(c22 - m2)
    e23 = tl.exp(c23 - m2)
    e30 = tl.exp(c30 - m3)
    e31 = tl.exp(c31 - m3)
    e32 = tl.exp(c32 - m3)
    e33 = tl.exp(c33 - m3)
    inv_r0 = 1.0 / (e00 + e01 + e02 + e03)
    inv_r1 = 1.0 / (e10 + e11 + e12 + e13)
    inv_r2 = 1.0 / (e20 + e21 + e22 + e23)
    inv_r3 = 1.0 / (e30 + e31 + e32 + e33)
    v00 = e00 * inv_r0
    v01 = e01 * inv_r0
    v02 = e02 * inv_r0
    v03 = e03 * inv_r0
    v10 = e10 * inv_r1
    v11 = e11 * inv_r1
    v12 = e12 * inv_r1
    v13 = e13 * inv_r1
    v20 = e20 * inv_r2
    v21 = e21 * inv_r2
    v22 = e22 * inv_r2
    v23 = e23 * inv_r2
    v30 = e30 * inv_r3
    v31 = e31 * inv_r3
    v32 = e32 * inv_r3
    v33 = e33 * inv_r3
    # softmax output = S (row-sums-to-1). We treat S as the input to iteration 0.
    # Iteration 0: col-norm(S + eps), NO row-norm on iter 0 (matches fwd).
    #   pre_col: add eps to every element
    v00 = v00 + HC_EPS
    v01 = v01 + HC_EPS
    v02 = v02 + HC_EPS
    v03 = v03 + HC_EPS
    v10 = v10 + HC_EPS
    v11 = v11 + HC_EPS
    v12 = v12 + HC_EPS
    v13 = v13 + HC_EPS
    v20 = v20 + HC_EPS
    v21 = v21 + HC_EPS
    v22 = v22 + HC_EPS
    v23 = v23 + HC_EPS
    v30 = v30 + HC_EPS
    v31 = v31 + HC_EPS
    v32 = v32 + HC_EPS
    v33 = v33 + HC_EPS
    # col-norm
    inv_c0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
    inv_c1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
    inv_c2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
    inv_c3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
    v00 = v00 * inv_c0
    v01 = v01 * inv_c1
    v02 = v02 * inv_c2
    v03 = v03 * inv_c3
    v10 = v10 * inv_c0
    v11 = v11 * inv_c1
    v12 = v12 * inv_c2
    v13 = v13 * inv_c3
    v20 = v20 * inv_c0
    v21 = v21 * inv_c1
    v22 = v22 * inv_c2
    v23 = v23 * inv_c3
    v30 = v30 * inv_c0
    v31 = v31 * inv_c1
    v32 = v32 * inv_c2
    v33 = v33 * inv_c3
    # Save M0 to scratch (before iters 1..ITERS-1)
    tl.store(scratch_ptr + sb + 0 * 16 + 0, v00)
    tl.store(scratch_ptr + sb + 0 * 16 + 1, v01)
    tl.store(scratch_ptr + sb + 0 * 16 + 2, v02)
    tl.store(scratch_ptr + sb + 0 * 16 + 3, v03)
    tl.store(scratch_ptr + sb + 0 * 16 + 4, v10)
    tl.store(scratch_ptr + sb + 0 * 16 + 5, v11)
    tl.store(scratch_ptr + sb + 0 * 16 + 6, v12)
    tl.store(scratch_ptr + sb + 0 * 16 + 7, v13)
    tl.store(scratch_ptr + sb + 0 * 16 + 8, v20)
    tl.store(scratch_ptr + sb + 0 * 16 + 9, v21)
    tl.store(scratch_ptr + sb + 0 * 16 + 10, v22)
    tl.store(scratch_ptr + sb + 0 * 16 + 11, v23)
    tl.store(scratch_ptr + sb + 0 * 16 + 12, v30)
    tl.store(scratch_ptr + sb + 0 * 16 + 13, v31)
    tl.store(scratch_ptr + sb + 0 * 16 + 14, v32)
    tl.store(scratch_ptr + sb + 0 * 16 + 15, v33)

    # Iters 1 .. ITERS-1: row-norm then col-norm
    for k in tl.static_range(1, ITERS):
        ir0 = 1.0 / (v00 + v01 + v02 + v03 + HC_EPS)
        ir1 = 1.0 / (v10 + v11 + v12 + v13 + HC_EPS)
        ir2 = 1.0 / (v20 + v21 + v22 + v23 + HC_EPS)
        ir3 = 1.0 / (v30 + v31 + v32 + v33 + HC_EPS)
        v00 = v00 * ir0
        v01 = v01 * ir0
        v02 = v02 * ir0
        v03 = v03 * ir0
        v10 = v10 * ir1
        v11 = v11 * ir1
        v12 = v12 * ir1
        v13 = v13 * ir1
        v20 = v20 * ir2
        v21 = v21 * ir2
        v22 = v22 * ir2
        v23 = v23 * ir2
        v30 = v30 * ir3
        v31 = v31 * ir3
        v32 = v32 * ir3
        v33 = v33 * ir3
        ic0 = 1.0 / (v00 + v10 + v20 + v30 + HC_EPS)
        ic1 = 1.0 / (v01 + v11 + v21 + v31 + HC_EPS)
        ic2 = 1.0 / (v02 + v12 + v22 + v32 + HC_EPS)
        ic3 = 1.0 / (v03 + v13 + v23 + v33 + HC_EPS)
        v00 = v00 * ic0
        v01 = v01 * ic1
        v02 = v02 * ic2
        v03 = v03 * ic3
        v10 = v10 * ic0
        v11 = v11 * ic1
        v12 = v12 * ic2
        v13 = v13 * ic3
        v20 = v20 * ic0
        v21 = v21 * ic1
        v22 = v22 * ic2
        v23 = v23 * ic3
        v30 = v30 * ic0
        v31 = v31 * ic1
        v32 = v32 * ic2
        v33 = v33 * ic3
        # store
        so = sb + k * 16
        tl.store(scratch_ptr + so + 0, v00)
        tl.store(scratch_ptr + so + 1, v01)
        tl.store(scratch_ptr + so + 2, v02)
        tl.store(scratch_ptr + so + 3, v03)
        tl.store(scratch_ptr + so + 4, v10)
        tl.store(scratch_ptr + so + 5, v11)
        tl.store(scratch_ptr + so + 6, v12)
        tl.store(scratch_ptr + so + 7, v13)
        tl.store(scratch_ptr + so + 8, v20)
        tl.store(scratch_ptr + so + 9, v21)
        tl.store(scratch_ptr + so + 10, v22)
        tl.store(scratch_ptr + so + 11, v23)
        tl.store(scratch_ptr + so + 12, v30)
        tl.store(scratch_ptr + so + 13, v31)
        tl.store(scratch_ptr + so + 14, v32)
        tl.store(scratch_ptr + so + 15, v33)

    # ---- BACKWARD ----
    # dM = grad_comb_frag  (dL/dM_final)
    g00 = tl.load(grad_comb_ptr + cb + 0)
    g01 = tl.load(grad_comb_ptr + cb + 1)
    g02 = tl.load(grad_comb_ptr + cb + 2)
    g03 = tl.load(grad_comb_ptr + cb + 3)
    g10 = tl.load(grad_comb_ptr + cb + 4)
    g11 = tl.load(grad_comb_ptr + cb + 5)
    g12 = tl.load(grad_comb_ptr + cb + 6)
    g13 = tl.load(grad_comb_ptr + cb + 7)
    g20 = tl.load(grad_comb_ptr + cb + 8)
    g21 = tl.load(grad_comb_ptr + cb + 9)
    g22 = tl.load(grad_comb_ptr + cb + 10)
    g23 = tl.load(grad_comb_ptr + cb + 11)
    g30 = tl.load(grad_comb_ptr + cb + 12)
    g31 = tl.load(grad_comb_ptr + cb + 13)
    g32 = tl.load(grad_comb_ptr + cb + 14)
    g33 = tl.load(grad_comb_ptr + cb + 15)

    # Walk iters ITERS-1 .. 1 backward (row-norm then col-norm, undo col first, then row).
    for k_rev in tl.static_range(1, ITERS):
        k = ITERS - k_rev  # iters counted from top
        # Load M_after_iter_k (post-col-norm) from scratch -- this is what dM currently corresponds to
        so = sb + k * 16
        u00 = tl.load(scratch_ptr + so + 0)
        u01 = tl.load(scratch_ptr + so + 1)
        u02 = tl.load(scratch_ptr + so + 2)
        u03 = tl.load(scratch_ptr + so + 3)
        u10 = tl.load(scratch_ptr + so + 4)
        u11 = tl.load(scratch_ptr + so + 5)
        u12 = tl.load(scratch_ptr + so + 6)
        u13 = tl.load(scratch_ptr + so + 7)
        u20 = tl.load(scratch_ptr + so + 8)
        u21 = tl.load(scratch_ptr + so + 9)
        u22 = tl.load(scratch_ptr + so + 10)
        u23 = tl.load(scratch_ptr + so + 11)
        u30 = tl.load(scratch_ptr + so + 12)
        u31 = tl.load(scratch_ptr + so + 13)
        u32 = tl.load(scratch_ptr + so + 14)
        u33 = tl.load(scratch_ptr + so + 15)
        # Undo col-norm: M_after_col[i,j] = M_before_col[i,j] / (Sc[j] + eps),
        # where M_before_col = row-norm-output. But we don't have M_before_col in scratch;
        # we could recompute by dividing u by (row-norm reciprocal). Simpler: re-derive Sc[j]
        # from the invariant that after col-norm each column sums to Sc'[j]/(Sc[j]+eps)... too messy.
        # Actually easier: load M at iter k-1 (post col-norm of previous iter) as the input to
        # this iter's row-norm.
        so_prev = sb + (k - 1) * 16
        p00 = tl.load(scratch_ptr + so_prev + 0)
        p01 = tl.load(scratch_ptr + so_prev + 1)
        p02 = tl.load(scratch_ptr + so_prev + 2)
        p03 = tl.load(scratch_ptr + so_prev + 3)
        p10 = tl.load(scratch_ptr + so_prev + 4)
        p11 = tl.load(scratch_ptr + so_prev + 5)
        p12 = tl.load(scratch_ptr + so_prev + 6)
        p13 = tl.load(scratch_ptr + so_prev + 7)
        p20 = tl.load(scratch_ptr + so_prev + 8)
        p21 = tl.load(scratch_ptr + so_prev + 9)
        p22 = tl.load(scratch_ptr + so_prev + 10)
        p23 = tl.load(scratch_ptr + so_prev + 11)
        p30 = tl.load(scratch_ptr + so_prev + 12)
        p31 = tl.load(scratch_ptr + so_prev + 13)
        p32 = tl.load(scratch_ptr + so_prev + 14)
        p33 = tl.load(scratch_ptr + so_prev + 15)
        # Row-norm forward: Sr[i] = sum_j p[i,j], q[i,j] = p[i,j] / (Sr[i]+eps)
        Sr0 = p00 + p01 + p02 + p03
        ir0 = 1.0 / (Sr0 + HC_EPS)
        Sr1 = p10 + p11 + p12 + p13
        ir1 = 1.0 / (Sr1 + HC_EPS)
        Sr2 = p20 + p21 + p22 + p23
        ir2 = 1.0 / (Sr2 + HC_EPS)
        Sr3 = p30 + p31 + p32 + p33
        ir3 = 1.0 / (Sr3 + HC_EPS)
        q00 = p00 * ir0
        q01 = p01 * ir0
        q02 = p02 * ir0
        q03 = p03 * ir0
        q10 = p10 * ir1
        q11 = p11 * ir1
        q12 = p12 * ir1
        q13 = p13 * ir1
        q20 = p20 * ir2
        q21 = p21 * ir2
        q22 = p22 * ir2
        q23 = p23 * ir2
        q30 = p30 * ir3
        q31 = p31 * ir3
        q32 = p32 * ir3
        q33 = p33 * ir3
        # Col-norm forward: Sc[j] = sum_i q[i,j], u[i,j] = q[i,j] / (Sc[j]+eps)
        Sc0 = q00 + q10 + q20 + q30
        ic0 = 1.0 / (Sc0 + HC_EPS)
        Sc1 = q01 + q11 + q21 + q31
        ic1 = 1.0 / (Sc1 + HC_EPS)
        Sc2 = q02 + q12 + q22 + q32
        ic2 = 1.0 / (Sc2 + HC_EPS)
        Sc3 = q03 + q13 + q23 + q33
        ic3 = 1.0 / (Sc3 + HC_EPS)

        # dq from du(=g): dq[i,j] = ic[j] * (g[i,j] - u[i,j]*sum_i(g[i,j]))
        sc_dot0 = g00 * u00 + g10 * u10 + g20 * u20 + g30 * u30
        sc_dot1 = g01 * u01 + g11 * u11 + g21 * u21 + g31 * u31
        sc_dot2 = g02 * u02 + g12 * u12 + g22 * u22 + g32 * u32
        sc_dot3 = g03 * u03 + g13 * u13 + g23 * u23 + g33 * u33
        dq00 = ic0 * (g00 - sc_dot0)
        dq01 = ic1 * (g01 - sc_dot1)
        dq02 = ic2 * (g02 - sc_dot2)
        dq03 = ic3 * (g03 - sc_dot3)
        dq10 = ic0 * (g10 - sc_dot0)
        dq11 = ic1 * (g11 - sc_dot1)
        dq12 = ic2 * (g12 - sc_dot2)
        dq13 = ic3 * (g13 - sc_dot3)
        dq20 = ic0 * (g20 - sc_dot0)
        dq21 = ic1 * (g21 - sc_dot1)
        dq22 = ic2 * (g22 - sc_dot2)
        dq23 = ic3 * (g23 - sc_dot3)
        dq30 = ic0 * (g30 - sc_dot0)
        dq31 = ic1 * (g31 - sc_dot1)
        dq32 = ic2 * (g32 - sc_dot2)
        dq33 = ic3 * (g33 - sc_dot3)

        # dp from dq(=q-normalization by rows): dp[i,j] = ir[i] * (dq[i,j] - q[i,j]*sum_j(dq[i,j]))
        sr_dot0 = dq00 * q00 + dq01 * q01 + dq02 * q02 + dq03 * q03
        sr_dot1 = dq10 * q10 + dq11 * q11 + dq12 * q12 + dq13 * q13
        sr_dot2 = dq20 * q20 + dq21 * q21 + dq22 * q22 + dq23 * q23
        sr_dot3 = dq30 * q30 + dq31 * q31 + dq32 * q32 + dq33 * q33
        g00 = ir0 * (dq00 - sr_dot0)
        g01 = ir0 * (dq01 - sr_dot0)
        g02 = ir0 * (dq02 - sr_dot0)
        g03 = ir0 * (dq03 - sr_dot0)
        g10 = ir1 * (dq10 - sr_dot1)
        g11 = ir1 * (dq11 - sr_dot1)
        g12 = ir1 * (dq12 - sr_dot1)
        g13 = ir1 * (dq13 - sr_dot1)
        g20 = ir2 * (dq20 - sr_dot2)
        g21 = ir2 * (dq21 - sr_dot2)
        g22 = ir2 * (dq22 - sr_dot2)
        g23 = ir2 * (dq23 - sr_dot2)
        g30 = ir3 * (dq30 - sr_dot3)
        g31 = ir3 * (dq31 - sr_dot3)
        g32 = ir3 * (dq32 - sr_dot3)
        g33 = ir3 * (dq33 - sr_dot3)

    # Now g[i,j] = dL/dM at end of iter 0 (post-col-norm of first iter).
    # Iter 0 forward was: M0 = softmax(clogits).exp/rowsum , then +eps , then col-norm.
    # We need to invert: (1) col-norm, (2) +eps identity, (3) softmax rowwise.
    # Retrieve M0-input to col-norm: v_pre_col = softmax(logits) + HC_EPS (row-sums-to-1 + eps).
    Sr0 = e00 + e01 + e02 + e03
    sinv_r0 = 1.0 / Sr0
    Sr1 = e10 + e11 + e12 + e13
    sinv_r1 = 1.0 / Sr1
    Sr2 = e20 + e21 + e22 + e23
    sinv_r2 = 1.0 / Sr2
    Sr3 = e30 + e31 + e32 + e33
    sinv_r3 = 1.0 / Sr3
    sm00 = e00 * sinv_r0 + HC_EPS
    sm01 = e01 * sinv_r0 + HC_EPS
    sm02 = e02 * sinv_r0 + HC_EPS
    sm03 = e03 * sinv_r0 + HC_EPS
    sm10 = e10 * sinv_r1 + HC_EPS
    sm11 = e11 * sinv_r1 + HC_EPS
    sm12 = e12 * sinv_r1 + HC_EPS
    sm13 = e13 * sinv_r1 + HC_EPS
    sm20 = e20 * sinv_r2 + HC_EPS
    sm21 = e21 * sinv_r2 + HC_EPS
    sm22 = e22 * sinv_r2 + HC_EPS
    sm23 = e23 * sinv_r2 + HC_EPS
    sm30 = e30 * sinv_r3 + HC_EPS
    sm31 = e31 * sinv_r3 + HC_EPS
    sm32 = e32 * sinv_r3 + HC_EPS
    sm33 = e33 * sinv_r3 + HC_EPS
    # col sums of sm
    Sc0 = sm00 + sm10 + sm20 + sm30
    ic0 = 1.0 / (Sc0 + HC_EPS)
    Sc1 = sm01 + sm11 + sm21 + sm31
    ic1 = 1.0 / (Sc1 + HC_EPS)
    Sc2 = sm02 + sm12 + sm22 + sm32
    ic2 = 1.0 / (Sc2 + HC_EPS)
    Sc3 = sm03 + sm13 + sm23 + sm33
    ic3 = 1.0 / (Sc3 + HC_EPS)
    # u after iter-0 col-norm
    u00 = sm00 * ic0
    u01 = sm01 * ic1
    u02 = sm02 * ic2
    u03 = sm03 * ic3
    u10 = sm10 * ic0
    u11 = sm11 * ic1
    u12 = sm12 * ic2
    u13 = sm13 * ic3
    u20 = sm20 * ic0
    u21 = sm21 * ic1
    u22 = sm22 * ic2
    u23 = sm23 * ic3
    u30 = sm30 * ic0
    u31 = sm31 * ic1
    u32 = sm32 * ic2
    u33 = sm33 * ic3
    # dSm from du: dSm = ic[j] * (g - u * sum_i g_row_of_col)
    sc_dot0 = g00 * u00 + g10 * u10 + g20 * u20 + g30 * u30
    sc_dot1 = g01 * u01 + g11 * u11 + g21 * u21 + g31 * u31
    sc_dot2 = g02 * u02 + g12 * u12 + g22 * u22 + g32 * u32
    sc_dot3 = g03 * u03 + g13 * u13 + g23 * u23 + g33 * u33
    dsm00 = ic0 * (g00 - sc_dot0)
    dsm01 = ic1 * (g01 - sc_dot1)
    dsm02 = ic2 * (g02 - sc_dot2)
    dsm03 = ic3 * (g03 - sc_dot3)
    dsm10 = ic0 * (g10 - sc_dot0)
    dsm11 = ic1 * (g11 - sc_dot1)
    dsm12 = ic2 * (g12 - sc_dot2)
    dsm13 = ic3 * (g13 - sc_dot3)
    dsm20 = ic0 * (g20 - sc_dot0)
    dsm21 = ic1 * (g21 - sc_dot1)
    dsm22 = ic2 * (g22 - sc_dot2)
    dsm23 = ic3 * (g23 - sc_dot3)
    dsm30 = ic0 * (g30 - sc_dot0)
    dsm31 = ic1 * (g31 - sc_dot1)
    dsm32 = ic2 * (g32 - sc_dot2)
    dsm33 = ic3 * (g33 - sc_dot3)
    # sm = softmax_row(clogits) + eps  -> derivative wrt softmax output (row-normalized e):
    # d_softmax[i,j] = softmax[i,j] * (dsm[i,j] - sum_j dsm[i,j]*softmax[i,j])
    # softmax[i,j] here = e[i,j] * sinv_r[i]
    s00 = e00 * sinv_r0
    s01 = e01 * sinv_r0
    s02 = e02 * sinv_r0
    s03 = e03 * sinv_r0
    s10 = e10 * sinv_r1
    s11 = e11 * sinv_r1
    s12 = e12 * sinv_r1
    s13 = e13 * sinv_r1
    s20 = e20 * sinv_r2
    s21 = e21 * sinv_r2
    s22 = e22 * sinv_r2
    s23 = e23 * sinv_r2
    s30 = e30 * sinv_r3
    s31 = e31 * sinv_r3
    s32 = e32 * sinv_r3
    s33 = e33 * sinv_r3
    sr_dot0 = dsm00 * s00 + dsm01 * s01 + dsm02 * s02 + dsm03 * s03
    sr_dot1 = dsm10 * s10 + dsm11 * s11 + dsm12 * s12 + dsm13 * s13
    sr_dot2 = dsm20 * s20 + dsm21 * s21 + dsm22 * s22 + dsm23 * s23
    sr_dot3 = dsm30 * s30 + dsm31 * s31 + dsm32 * s32 + dsm33 * s33
    dc00 = s00 * (dsm00 - sr_dot0)
    dc01 = s01 * (dsm01 - sr_dot0)
    dc02 = s02 * (dsm02 - sr_dot0)
    dc03 = s03 * (dsm03 - sr_dot0)
    dc10 = s10 * (dsm10 - sr_dot1)
    dc11 = s11 * (dsm11 - sr_dot1)
    dc12 = s12 * (dsm12 - sr_dot1)
    dc13 = s13 * (dsm13 - sr_dot1)
    dc20 = s20 * (dsm20 - sr_dot2)
    dc21 = s21 * (dsm21 - sr_dot2)
    dc22 = s22 * (dsm22 - sr_dot2)
    dc23 = s23 * (dsm23 - sr_dot2)
    dc30 = s30 * (dsm30 - sr_dot3)
    dc31 = s31 * (dsm31 - sr_dot3)
    dc32 = s32 * (dsm32 - sr_dot3)
    dc33 = s33 * (dsm33 - sr_dot3)
    # Clamp bwd: pass-through mask (mask 1 if not clamped)
    if APPLY_CLAMP:
        dc00 = tl.where((l00 > CLAMP_MIN) & (l00 < CLAMP_MAX), dc00, 0.0)
        dc01 = tl.where((l01 > CLAMP_MIN) & (l01 < CLAMP_MAX), dc01, 0.0)
        dc02 = tl.where((l02 > CLAMP_MIN) & (l02 < CLAMP_MAX), dc02, 0.0)
        dc03 = tl.where((l03 > CLAMP_MIN) & (l03 < CLAMP_MAX), dc03, 0.0)
        dc10 = tl.where((l10 > CLAMP_MIN) & (l10 < CLAMP_MAX), dc10, 0.0)
        dc11 = tl.where((l11 > CLAMP_MIN) & (l11 < CLAMP_MAX), dc11, 0.0)
        dc12 = tl.where((l12 > CLAMP_MIN) & (l12 < CLAMP_MAX), dc12, 0.0)
        dc13 = tl.where((l13 > CLAMP_MIN) & (l13 < CLAMP_MAX), dc13, 0.0)
        dc20 = tl.where((l20 > CLAMP_MIN) & (l20 < CLAMP_MAX), dc20, 0.0)
        dc21 = tl.where((l21 > CLAMP_MIN) & (l21 < CLAMP_MAX), dc21, 0.0)
        dc22 = tl.where((l22 > CLAMP_MIN) & (l22 < CLAMP_MAX), dc22, 0.0)
        dc23 = tl.where((l23 > CLAMP_MIN) & (l23 < CLAMP_MAX), dc23, 0.0)
        dc30 = tl.where((l30 > CLAMP_MIN) & (l30 < CLAMP_MAX), dc30, 0.0)
        dc31 = tl.where((l31 > CLAMP_MIN) & (l31 < CLAMP_MAX), dc31, 0.0)
        dc32 = tl.where((l32 > CLAMP_MIN) & (l32 < CLAMP_MAX), dc32, 0.0)
        dc33 = tl.where((l33 > CLAMP_MIN) & (l33 < CLAMP_MAX), dc33, 0.0)
    # logits = mix*a2 + b   -> dmix_l = dc*a2 ; dbase_l = dc ; dalpha2_partial = sum(dc * mix)
    da2 = (
        dc00 * ml00
        + dc01 * ml01
        + dc02 * ml02
        + dc03 * ml03
        + dc10 * ml10
        + dc11 * ml11
        + dc12 * ml12
        + dc13 * ml13
        + dc20 * ml20
        + dc21 * ml21
        + dc22 * ml22
        + dc23 * ml23
        + dc30 * ml30
        + dc31 * ml31
        + dc32 * ml32
        + dc33 * ml33
    )

    # Pre-head bwd: pre_out = sigmoid(z_pre) + eps, grad w.r.t z_pre = grad_pre * s*(1-s)
    gpre0 = tl.load(grad_pre_ptr + pb + 0)
    gpre1 = tl.load(grad_pre_ptr + pb + 1)
    gpre2 = tl.load(grad_pre_ptr + pb + 2)
    gpre3 = tl.load(grad_pre_ptr + pb + 3)
    dz_pre0 = gpre0 * sp0 * (1.0 - sp0)
    dz_pre1 = gpre1 * sp1 * (1.0 - sp1)
    dz_pre2 = gpre2 * sp2 * (1.0 - sp2)
    dz_pre3 = gpre3 * sp3 * (1.0 - sp3)
    da0 = (
        dz_pre0 * mix_pre0
        + dz_pre1 * mix_pre1
        + dz_pre2 * mix_pre2
        + dz_pre3 * mix_pre3
    )

    # Post-head bwd: post_out = 2 * sigmoid(z_po), dz_po = grad_post * 2 * s*(1-s)
    gpo0 = tl.load(grad_post_ptr + pb + 0)
    gpo1 = tl.load(grad_post_ptr + pb + 1)
    gpo2 = tl.load(grad_post_ptr + pb + 2)
    gpo3 = tl.load(grad_post_ptr + pb + 3)
    dz_po0 = gpo0 * 2.0 * spo0 * (1.0 - spo0)
    dz_po1 = gpo1 * 2.0 * spo1 * (1.0 - spo1)
    dz_po2 = gpo2 * 2.0 * spo2 * (1.0 - spo2)
    dz_po3 = gpo3 * 2.0 * spo3 * (1.0 - spo3)
    da1 = dz_po0 * mix_po0 + dz_po1 * mix_po1 + dz_po2 * mix_po2 + dz_po3 * mix_po3

    # Store grad_mixes (T, 24)
    tl.store(grad_mixes_ptr + mb + 0, dz_pre0 * a0)
    tl.store(grad_mixes_ptr + mb + 1, dz_pre1 * a0)
    tl.store(grad_mixes_ptr + mb + 2, dz_pre2 * a0)
    tl.store(grad_mixes_ptr + mb + 3, dz_pre3 * a0)
    tl.store(grad_mixes_ptr + mb + 4, dz_po0 * a1)
    tl.store(grad_mixes_ptr + mb + 5, dz_po1 * a1)
    tl.store(grad_mixes_ptr + mb + 6, dz_po2 * a1)
    tl.store(grad_mixes_ptr + mb + 7, dz_po3 * a1)
    tl.store(grad_mixes_ptr + mb + 8, dc00 * a2)
    tl.store(grad_mixes_ptr + mb + 9, dc01 * a2)
    tl.store(grad_mixes_ptr + mb + 10, dc02 * a2)
    tl.store(grad_mixes_ptr + mb + 11, dc03 * a2)
    tl.store(grad_mixes_ptr + mb + 12, dc10 * a2)
    tl.store(grad_mixes_ptr + mb + 13, dc11 * a2)
    tl.store(grad_mixes_ptr + mb + 14, dc12 * a2)
    tl.store(grad_mixes_ptr + mb + 15, dc13 * a2)
    tl.store(grad_mixes_ptr + mb + 16, dc20 * a2)
    tl.store(grad_mixes_ptr + mb + 17, dc21 * a2)
    tl.store(grad_mixes_ptr + mb + 18, dc22 * a2)
    tl.store(grad_mixes_ptr + mb + 19, dc23 * a2)
    tl.store(grad_mixes_ptr + mb + 20, dc30 * a2)
    tl.store(grad_mixes_ptr + mb + 21, dc31 * a2)
    tl.store(grad_mixes_ptr + mb + 22, dc32 * a2)
    tl.store(grad_mixes_ptr + mb + 23, dc33 * a2)

    # grad_alpha partial (per-token)
    tl.store(grad_alpha_part_ptr + pid * 3 + 0, da0)
    tl.store(grad_alpha_part_ptr + pid * 3 + 1, da1)
    tl.store(grad_alpha_part_ptr + pid * 3 + 2, da2)

    # grad_base partial (per-token): dbase = dz_pre / dz_po / dc directly
    tl.store(grad_base_part_ptr + mb + 0, dz_pre0)
    tl.store(grad_base_part_ptr + mb + 1, dz_pre1)
    tl.store(grad_base_part_ptr + mb + 2, dz_pre2)
    tl.store(grad_base_part_ptr + mb + 3, dz_pre3)
    tl.store(grad_base_part_ptr + mb + 4, dz_po0)
    tl.store(grad_base_part_ptr + mb + 5, dz_po1)
    tl.store(grad_base_part_ptr + mb + 6, dz_po2)
    tl.store(grad_base_part_ptr + mb + 7, dz_po3)
    tl.store(grad_base_part_ptr + mb + 8, dc00)
    tl.store(grad_base_part_ptr + mb + 9, dc01)
    tl.store(grad_base_part_ptr + mb + 10, dc02)
    tl.store(grad_base_part_ptr + mb + 11, dc03)
    tl.store(grad_base_part_ptr + mb + 12, dc10)
    tl.store(grad_base_part_ptr + mb + 13, dc11)
    tl.store(grad_base_part_ptr + mb + 14, dc12)
    tl.store(grad_base_part_ptr + mb + 15, dc13)
    tl.store(grad_base_part_ptr + mb + 16, dc20)
    tl.store(grad_base_part_ptr + mb + 17, dc21)
    tl.store(grad_base_part_ptr + mb + 18, dc22)
    tl.store(grad_base_part_ptr + mb + 19, dc23)
    tl.store(grad_base_part_ptr + mb + 20, dc30)
    tl.store(grad_base_part_ptr + mb + 21, dc31)
    tl.store(grad_base_part_ptr + mb + 22, dc32)
    tl.store(grad_base_part_ptr + mb + 23, dc33)


# ---------------------------------------------------------------------------
# Heuristic-optimized vectorized small-stage backward kernel.
# Following baseline patterns: autotune BLOCK_T, explicit range() loops.
# Uses tle.load(is_async=True) for data prefetching when available.
# Processes BLOCK_T tokens per program for vectorized execution.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_T": 64}, num_warps=32, num_stages=2),
    ],
    key=["T", "ITERS"],
)
@triton.heuristics({"EVEN_T": lambda args: args["T"] % args["BLOCK_T"] == 0})
@triton.jit
def _small_stage_bwd_vec_kernel(
    mixes_ptr,  # (T, 24) fp32
    alpha_ptr,  # (3,)    fp32
    base_ptr,  # (24,)   fp32
    grad_pre_ptr,  # (T, 4)   fp32
    grad_post_ptr,  # (T, 4)   fp32
    grad_comb_ptr,  # (T, 16) fp32  (flattened from T,4,4)
    grad_mixes_ptr,  # (T, 24)  fp32     OUT
    grad_alpha_ptr,  # (3,)     fp32     OUT (atomic add)
    grad_base_ptr,  # (24,)    fp32     OUT (atomic add)
    scratch_ptr,  # (T, ITERS, 16) fp32
    T: tl.constexpr,
    HC_EPS: tl.constexpr,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    APPLY_CLAMP: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    EVEN_T: tl.constexpr,
):
    """Heuristic-optimized TLE vectorized small-stage backward. BLOCK_T tokens per program."""
    pid = tl.program_id(0)
    t_off = pid * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    offs_4 = tl.arange(0, 4)  # [4]

    # Load constants with TLE async (scalar loads don't need mask/other)
    a0 = tle.load(alpha_ptr + 0, is_async=True)
    a1 = tle.load(alpha_ptr + 1, is_async=True)
    a2 = tle.load(alpha_ptr + 2, is_async=True)
    base_pre = tle.load(base_ptr + offs_4, is_async=True)  # [4]
    base_post = tle.load(base_ptr + 4 + offs_4, is_async=True)  # [4]

    # TLE async prefetch mixes data (split mask vs no-mask path)
    if EVEN_T:
        mix_pre = tle.load(
            mixes_ptr + t_off[:, None] * 24 + offs_4[None, :], is_async=True
        )  # [BT,4]
        mix_post = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :], is_async=True
        )  # [BT,4]
        ml0 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :], is_async=True
        )
        ml1 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :], is_async=True
        )
        ml2 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :], is_async=True
        )
        ml3 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :], is_async=True
        )
    else:
        t_mask = t_off < T
        mix_pre = tle.load(
            mixes_ptr + t_off[:, None] * 24 + offs_4[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        mix_post = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        ml0 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        ml1 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        ml2 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        ml3 = tle.load(
            mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )

    base_l0 = tle.load(base_ptr + 8 + offs_4, is_async=True)  # [4]
    base_l1 = tle.load(base_ptr + 12 + offs_4, is_async=True)  # [4]
    base_l2 = tle.load(base_ptr + 16 + offs_4, is_async=True)  # [4]
    base_l3 = tle.load(base_ptr + 20 + offs_4, is_async=True)  # [4]

    # ======== FORWARD RECOMPUTE (vectorized) ========
    z_pre = mix_pre * a0 + base_pre[None, :]
    sp = tl.sigmoid(z_pre)  # [BT, 4]
    z_post = mix_post * a1 + base_post[None, :]
    spo = tl.sigmoid(z_post)  # [BT, 4]

    # Logits per row + clamp
    l0 = ml0 * a2 + base_l0[None, :]  # [BT, 4] row 0
    l1 = ml1 * a2 + base_l1[None, :]  # [BT, 4] row 1
    l2 = ml2 * a2 + base_l2[None, :]  # [BT, 4] row 2
    l3 = ml3 * a2 + base_l3[None, :]  # [BT, 4] row 3
    if APPLY_CLAMP:
        c0 = tl.minimum(tl.maximum(l0, CLAMP_MIN), CLAMP_MAX)
        c1 = tl.minimum(tl.maximum(l1, CLAMP_MIN), CLAMP_MAX)
        c2 = tl.minimum(tl.maximum(l2, CLAMP_MIN), CLAMP_MAX)
        c3 = tl.minimum(tl.maximum(l3, CLAMP_MIN), CLAMP_MAX)
    else:
        c0 = l0
        c1 = l1
        c2 = l2
        c3 = l3

    # Row-wise softmax
    mx0 = tl.max(c0, axis=1)[:, None]
    mx1 = tl.max(c1, axis=1)[:, None]
    mx2 = tl.max(c2, axis=1)[:, None]
    mx3 = tl.max(c3, axis=1)[:, None]
    e0 = tl.exp(c0 - mx0)
    e1 = tl.exp(c1 - mx1)
    e2 = tl.exp(c2 - mx2)
    e3 = tl.exp(c3 - mx3)
    rs0 = tl.sum(e0, axis=1)[:, None]
    rs1 = tl.sum(e1, axis=1)[:, None]
    rs2 = tl.sum(e2, axis=1)[:, None]
    rs3 = tl.sum(e3, axis=1)[:, None]
    v0 = e0 / rs0 + HC_EPS
    v1 = e1 / rs1 + HC_EPS
    v2 = e2 / rs2 + HC_EPS
    v3 = e3 / rs3 + HC_EPS

    # Iter 0: col-norm
    cs = v0 + v1 + v2 + v3 + HC_EPS
    v0 = v0 / cs
    v1 = v1 / cs
    v2 = v2 / cs
    v3 = v3 / cs

    # Store iter 0 to scratch (vectorized: 4 stores of [BT,4] vs 16 scalar stores)
    scr_stride: tl.constexpr = ITERS * 16
    scr_base = t_off * scr_stride  # [BT] base offset per token
    if EVEN_T:
        tl.store(scratch_ptr + scr_base[:, None] + offs_4[None, :], v0)
        tl.store(scratch_ptr + scr_base[:, None] + (4 + offs_4)[None, :], v1)
        tl.store(scratch_ptr + scr_base[:, None] + (8 + offs_4)[None, :], v2)
        tl.store(scratch_ptr + scr_base[:, None] + (12 + offs_4)[None, :], v3)
    else:
        tl.store(
            scratch_ptr + scr_base[:, None] + offs_4[None, :], v0, mask=t_mask[:, None]
        )
        tl.store(
            scratch_ptr + scr_base[:, None] + (4 + offs_4)[None, :],
            v1,
            mask=t_mask[:, None],
        )
        tl.store(
            scratch_ptr + scr_base[:, None] + (8 + offs_4)[None, :],
            v2,
            mask=t_mask[:, None],
        )
        tl.store(
            scratch_ptr + scr_base[:, None] + (12 + offs_4)[None, :],
            v3,
            mask=t_mask[:, None],
        )

    # Iters 1..ITERS-1 (explicit range() for compiler optimization)
    for k in range(1, ITERS):
        r0s = tl.sum(v0, axis=1)[:, None] + HC_EPS
        r1s = tl.sum(v1, axis=1)[:, None] + HC_EPS
        r2s = tl.sum(v2, axis=1)[:, None] + HC_EPS
        r3s = tl.sum(v3, axis=1)[:, None] + HC_EPS
        v0 = v0 / r0s
        v1 = v1 / r1s
        v2 = v2 / r2s
        v3 = v3 / r3s
        cs = v0 + v1 + v2 + v3 + HC_EPS
        v0 = v0 / cs
        v1 = v1 / cs
        v2 = v2 / cs
        v3 = v3 / cs
        k_off = k * 16
        if EVEN_T:
            tl.store(scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :], v0)
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :], v1
            )
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :], v2
            )
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :], v3
            )
        else:
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :],
                v0,
                mask=t_mask[:, None],
            )
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :],
                v1,
                mask=t_mask[:, None],
            )
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :],
                v2,
                mask=t_mask[:, None],
            )
            tl.store(
                scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :],
                v3,
                mask=t_mask[:, None],
            )

    # ======== BACKWARD ========
    if EVEN_T:
        g0 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + offs_4[None, :], is_async=True
        )
        g1 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + (4 + offs_4)[None, :], is_async=True
        )
        g2 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + (8 + offs_4)[None, :], is_async=True
        )
        g3 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + (12 + offs_4)[None, :], is_async=True
        )
    else:
        g0 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + offs_4[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        g1 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + (4 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        g2 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + (8 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
        g3 = tle.load(
            grad_comb_ptr + t_off[:, None] * 16 + (12 + offs_4)[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )

    # Walk iters backward: ITERS-1 .. 1 (explicit range())
    for k_rev in range(1, ITERS):
        k = ITERS - k_rev
        k_off = k * 16
        pk_off = (k - 1) * 16
        # Load u (iter k state) and p (iter k-1 state) from scratch with TLE async
        if EVEN_T:
            u0 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :],
                is_async=True,
            )
            u1 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :],
                is_async=True,
            )
            u2 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :],
                is_async=True,
            )
            u3 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :],
                is_async=True,
            )
            p0 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + offs_4)[None, :],
                is_async=True,
            )
            p1 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + 4 + offs_4)[None, :],
                is_async=True,
            )
            p2 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + 8 + offs_4)[None, :],
                is_async=True,
            )
            p3 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + 12 + offs_4)[None, :],
                is_async=True,
            )
        else:
            u0 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            u1 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + 4 + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            u2 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + 8 + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            u3 = tle.load(
                scratch_ptr + scr_base[:, None] + (k_off + 12 + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            p0 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            p1 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + 4 + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            p2 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + 8 + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )
            p3 = tle.load(
                scratch_ptr + scr_base[:, None] + (pk_off + 12 + offs_4)[None, :],
                mask=t_mask[:, None],
                other=0.0,
                is_async=True,
            )

        # Recompute row/col norms for backward
        Sr0 = tl.sum(p0, axis=1)[:, None]
        ir0 = 1.0 / (Sr0 + HC_EPS)
        Sr1 = tl.sum(p1, axis=1)[:, None]
        ir1 = 1.0 / (Sr1 + HC_EPS)
        Sr2 = tl.sum(p2, axis=1)[:, None]
        ir2 = 1.0 / (Sr2 + HC_EPS)
        Sr3 = tl.sum(p3, axis=1)[:, None]
        ir3 = 1.0 / (Sr3 + HC_EPS)
        q0 = p0 * ir0
        q1 = p1 * ir1
        q2 = p2 * ir2
        q3 = p3 * ir3
        Sc = q0 + q1 + q2 + q3
        ic = 1.0 / (Sc + HC_EPS)

        # Col-norm backward: dq = ic * (g - u * col_dot)
        sc_dot = g0 * u0 + g1 * u1 + g2 * u2 + g3 * u3
        dq0 = ic * (g0 - sc_dot)
        dq1 = ic * (g1 - sc_dot)
        dq2 = ic * (g2 - sc_dot)
        dq3 = ic * (g3 - sc_dot)

        # Row-norm backward: dp = ir * (dq - q * row_dot)
        sr_dot0 = tl.sum(dq0 * q0, axis=1)[:, None]
        sr_dot1 = tl.sum(dq1 * q1, axis=1)[:, None]
        sr_dot2 = tl.sum(dq2 * q2, axis=1)[:, None]
        sr_dot3 = tl.sum(dq3 * q3, axis=1)[:, None]
        g0 = ir0 * (dq0 - sr_dot0)
        g1 = ir1 * (dq1 - sr_dot1)
        g2 = ir2 * (dq2 - sr_dot2)
        g3 = ir3 * (dq3 - sr_dot3)

    # Iter 0 backward: col-norm → softmax
    sinv_r0 = 1.0 / rs0
    sinv_r1 = 1.0 / rs1
    sinv_r2 = 1.0 / rs2
    sinv_r3 = 1.0 / rs3
    sm0b = e0 * sinv_r0 + HC_EPS
    sm1b = e1 * sinv_r1 + HC_EPS
    sm2b = e2 * sinv_r2 + HC_EPS
    sm3b = e3 * sinv_r3 + HC_EPS
    Sc0 = sm0b + sm1b + sm2b + sm3b + HC_EPS
    ic0 = 1.0 / Sc0
    u0b = sm0b * ic0
    u1b = sm1b * ic0
    u2b = sm2b * ic0
    u3b = sm3b * ic0

    sc_dot_0 = g0 * u0b + g1 * u1b + g2 * u2b + g3 * u3b
    dsm0 = ic0 * (g0 - sc_dot_0)
    dsm1 = ic0 * (g1 - sc_dot_0)
    dsm2 = ic0 * (g2 - sc_dot_0)
    dsm3 = ic0 * (g3 - sc_dot_0)

    # Softmax backward
    s0v = e0 * sinv_r0
    s1v = e1 * sinv_r1
    s2v = e2 * sinv_r2
    s3v = e3 * sinv_r3
    sd0 = tl.sum(dsm0 * s0v, axis=1)[:, None]
    sd1 = tl.sum(dsm1 * s1v, axis=1)[:, None]
    sd2 = tl.sum(dsm2 * s2v, axis=1)[:, None]
    sd3 = tl.sum(dsm3 * s3v, axis=1)[:, None]
    dc0 = s0v * (dsm0 - sd0)
    dc1 = s1v * (dsm1 - sd1)
    dc2 = s2v * (dsm2 - sd2)
    dc3 = s3v * (dsm3 - sd3)

    # Clamp backward
    if APPLY_CLAMP:
        dc0 = tl.where((l0 > CLAMP_MIN) & (l0 < CLAMP_MAX), dc0, 0.0)
        dc1 = tl.where((l1 > CLAMP_MIN) & (l1 < CLAMP_MAX), dc1, 0.0)
        dc2 = tl.where((l2 > CLAMP_MIN) & (l2 < CLAMP_MAX), dc2, 0.0)
        dc3 = tl.where((l3 > CLAMP_MIN) & (l3 < CLAMP_MAX), dc3, 0.0)

    # grad_alpha partials
    da2 = (
        tl.sum(dc0 * ml0, axis=1)
        + tl.sum(dc1 * ml1, axis=1)
        + tl.sum(dc2 * ml2, axis=1)
        + tl.sum(dc3 * ml3, axis=1)
    )
    if EVEN_T:
        grad_pre_v = tle.load(
            grad_pre_ptr + t_off[:, None] * 4 + offs_4[None, :], is_async=True
        )
    else:
        grad_pre_v = tle.load(
            grad_pre_ptr + t_off[:, None] * 4 + offs_4[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
    dz_pre = grad_pre_v * sp * (1.0 - sp)
    da0 = tl.sum(dz_pre * mix_pre, axis=1)
    if EVEN_T:
        grad_post_v = tle.load(
            grad_post_ptr + t_off[:, None] * 4 + offs_4[None, :], is_async=True
        )
    else:
        grad_post_v = tle.load(
            grad_post_ptr + t_off[:, None] * 4 + offs_4[None, :],
            mask=t_mask[:, None],
            other=0.0,
            is_async=True,
        )
    dz_post = grad_post_v * 2.0 * spo * (1.0 - spo)
    da1 = tl.sum(dz_post * mix_post, axis=1)

    # Store grad_mixes [BT, 24]
    if EVEN_T:
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + offs_4[None, :], dz_pre * a0)
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :], dz_post * a1
        )
        tl.store(grad_mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :], dc0 * a2)
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :], dc1 * a2
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :], dc2 * a2
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :], dc3 * a2
        )
    else:
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + offs_4[None, :],
            dz_pre * a0,
            mask=t_mask[:, None],
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (4 + offs_4)[None, :],
            dz_post * a1,
            mask=t_mask[:, None],
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (8 + offs_4)[None, :],
            dc0 * a2,
            mask=t_mask[:, None],
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (12 + offs_4)[None, :],
            dc1 * a2,
            mask=t_mask[:, None],
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (16 + offs_4)[None, :],
            dc2 * a2,
            mask=t_mask[:, None],
        )
        tl.store(
            grad_mixes_ptr + t_off[:, None] * 24 + (20 + offs_4)[None, :],
            dc3 * a2,
            mask=t_mask[:, None],
        )

    # Atomic add to grad_alpha [3] and grad_base [24]
    # Sum across BLOCK_T tokens before atomic add to reduce contention
    da0_sum = tl.sum(da0)
    da1_sum = tl.sum(da1)
    da2_sum = tl.sum(da2)
    tl.atomic_add(grad_alpha_ptr + 0, da0_sum)
    tl.atomic_add(grad_alpha_ptr + 1, da1_sum)
    tl.atomic_add(grad_alpha_ptr + 2, da2_sum)

    # For grad_base, sum across BLOCK_T tokens for each of 24 elements
    dz_pre_sum = tl.sum(dz_pre, axis=0)  # (4,)
    dz_post_sum = tl.sum(dz_post, axis=0)  # (4,)
    dc0_sum = tl.sum(dc0, axis=0)  # (4,)
    dc1_sum = tl.sum(dc1, axis=0)  # (4,)
    dc2_sum = tl.sum(dc2, axis=0)  # (4,)
    dc3_sum = tl.sum(dc3, axis=0)  # (4,)

    tl.atomic_add(grad_base_ptr + offs_4, dz_pre_sum)
    tl.atomic_add(grad_base_ptr + (4 + offs_4), dz_post_sum)
    tl.atomic_add(grad_base_ptr + (8 + offs_4), dc0_sum)
    tl.atomic_add(grad_base_ptr + (12 + offs_4), dc1_sum)
    tl.atomic_add(grad_base_ptr + (16 + offs_4), dc2_sum)
    tl.atomic_add(grad_base_ptr + (20 + offs_4), dc3_sum)


def _small_stage_bwd(
    mixes: torch.Tensor,  # (T, 24) fp32
    alpha: torch.Tensor,  # (3,)    fp32
    base: torch.Tensor,  # (24,)   fp32
    h_res_logits: torch.Tensor,  # (T, 4, 4) fp32   pre-clamp logits (not strictly needed; recomputed)
    grad_pre: torch.Tensor,  # (T, 4)  fp32
    grad_post_out: torch.Tensor,  # (T, 4)  fp32
    grad_comb_frag: torch.Tensor,  # (T, 4, 4) fp32
    hc_eps: float,
    clamp_min: float,
    clamp_max: float,
    iter_times: int,
):
    """Fused Triton small-stage backward with atomic add for grad_alpha/grad_base."""
    T = mixes.shape[0]
    grad_mixes = torch.empty((T, 24), dtype=torch.float32, device=mixes.device)
    grad_alpha = torch.zeros((3,), dtype=torch.float32, device=mixes.device)
    grad_base = torch.zeros((24,), dtype=torch.float32, device=mixes.device)
    scratch = torch.empty((T, iter_times, 16), dtype=torch.float32, device=mixes.device)
    apply_clamp = 1 if (clamp_min != 0.0 or clamp_max != 0.0) else 0

    # Use heuristic-optimized vectorized kernel with autotune
    grid = lambda meta: ((T + meta["BLOCK_T"] - 1) // meta["BLOCK_T"],)

    _small_stage_bwd_vec_kernel[grid](
        mixes,
        alpha,
        base,
        grad_pre,
        grad_post_out,
        grad_comb_frag.reshape(T, 16),
        grad_mixes,
        grad_alpha,
        grad_base,
        scratch,
        T=T,
        HC_EPS=hc_eps,
        CLAMP_MIN=float(clamp_min),
        CLAMP_MAX=float(clamp_max),
        APPLY_CLAMP=apply_clamp,
        ITERS=int(iter_times),
    )
    return grad_mixes, grad_alpha, grad_base


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def mhc_pre_clamp_sinkhorn_backward(
    # forward saved
    x: torch.Tensor,  # (B,S,N,D) or (T,N,D) original x
    phi: torch.Tensor,  # (hcMix, hcMult*D) fp32
    alpha: torch.Tensor,  # (3,)
    base: torch.Tensor,  # (hcMix,)
    inv_rms: torch.Tensor,  # (T,) fp32
    x_scaled: torch.Tensor,  # (T, HC_D) fp32
    mixes: torch.Tensor,  # (T, hcMix) fp32
    h_res_logits: torch.Tensor,  # (T, hcMult, hcMult) fp32   (unused; autograd recomputes)
    pre: torch.Tensor,  # (T, hcMult) fp32
    # grad inputs
    grad_y: torch.Tensor,  # (B,S,D) or (T,D) same dtype as x
    grad_post_out: torch.Tensor,  # (T, hcMult) fp32
    grad_comb_frag: torch.Tensor,  # (T, hcMult, hcMult) fp32
    # hyperparams
    norm_eps: float = 1e-6,
    hc_eps: float = 1e-6,
    clamp_min: float = 0.0,
    clamp_max: float = 0.0,
    iter_times: int = 20,
):
    """Full backward.  Returns (grad_x, grad_phi, grad_alpha, grad_base).

    Layout-agnostic: accepts x as (B,S,N,D) or (T,N,D), grad_y as (B,S,D)
    or (T,D) without any reshape/contiguous copy.  Shape and strides are
    extracted directly from the original tensors and forwarded to kernels.
    """
    # ---- Extract shape & strides without reshape/flatten ----
    if x.dim() == 4:
        B, S, N, D = x.shape
        T = B * S
        shape = (B, S, N, D)
        # stride along the token (T) dimension: jump one S-row in the B*S view
        stride_x_t = x.stride(1)  # stride(S dim) = N*D when contiguous
        stride_x_n = x.stride(2)  # stride(N dim) = D   when contiguous
    elif x.dim() == 3:
        T, N, D = x.shape
        shape = (T, N, D)
        stride_x_t = x.stride(0)
        stride_x_n = x.stride(1)
    else:
        raise ValueError(f"unsupported x.dim()={x.dim()}")

    if grad_y.dim() == 3:
        _B, _S, _D = grad_y.shape
        assert _B * _S == T and _D == D
        stride_gy_t = grad_y.stride(1)  # stride(S dim) in the B*S view
    elif grad_y.dim() == 2:
        assert grad_y.shape == (T, D)
        stride_gy_t = grad_y.stride(0)
    else:
        raise ValueError(f"unsupported grad_y.dim()={grad_y.dim()}")

    assert N == 4
    hc_d = N * D
    hc_mix = N * (N + 2)  # noqa: F841

    # ---- Y-backward (only for grad_pre_from_y) ----
    grad_pre_from_y = torch.empty(T, N, dtype=torch.float32, device=x.device)
    grad_x_from_y_tmp = torch.empty(T, N, D, dtype=torch.float32, device=x.device)
    _y_bwd_kernel_hc4[(T,)](
        grad_y,
        x,
        pre,
        grad_x_from_y_tmp,
        grad_pre_from_y,
        stride_gy_t=stride_gy_t,
        stride_x_t=stride_x_t,
        stride_x_n=stride_x_n,
        D=D,
    )

    # ---- Small-stage bwd (fused Triton) ----
    grad_mixes, grad_alpha, grad_base = _small_stage_bwd(
        mixes,
        alpha.to(torch.float32),
        base.to(torch.float32),
        h_res_logits,
        grad_pre_from_y,
        grad_post_out,
        grad_comb_frag,
        hc_eps,
        clamp_min,
        clamp_max,
        iter_times,
    )

    # ---- GEMM bwd ----
    phi_f = phi.to(torch.float32)
    grad_x_scaled = torch.mm(grad_mixes, phi_f)  # (T, HC_D)
    grad_phi = torch.mm(grad_mixes.t(), x_scaled)  # (hcMix, HC_D)

    # ---- Fused RMSNorm bwd + Y-backward combine ----
    grad_x = torch.empty(T, hc_d, dtype=x.dtype, device=x.device)
    _fused_ybwd_rmsbwd_kernel[(T,)](
        grad_y,
        x,
        grad_x_scaled,
        inv_rms,
        pre,
        grad_x,
        grad_pre_from_y,
        stride_gy_t=stride_gy_t,
        stride_x_t=stride_x_t,
        stride_x_n=stride_x_n,
        D=D,
        HC_D=hc_d,
        D_INV=1.0 / hc_d,
    )

    return (
        grad_x.view(*shape),
        grad_phi.reshape(phi.shape),
        grad_alpha,
        grad_base,
    )


# ---------------------------------------------------------------------------
# Reference implementation (autograd through forward)
# ---------------------------------------------------------------------------
def mhc_pre_clamp_sinkhorn_backward_ref(
    x,
    phi,
    alpha,
    base,
    grad_y,
    grad_post_out,
    grad_comb_frag,
    norm_eps=1e-6,
    hc_eps=1e-6,
    clamp_min=0.0,
    clamp_max=0.0,
    iter_times=20,
):
    """Pure PyTorch reference (autograd through forward). Returns
    (grad_x, grad_phi, grad_alpha, grad_base).
    """
    orig_dtype = x.dtype
    x_ = x.detach().to(torch.float32).requires_grad_(True)
    phi_ = phi.detach().to(torch.float32).requires_grad_(True)
    alpha_ = alpha.detach().to(torch.float32).requires_grad_(True)
    base_ = base.detach().to(torch.float32).requires_grad_(True)

    if x_.dim() == 4:
        B, S, N, D = x_.shape
        xf = x_.reshape(B * S, N, D)
    else:
        xf = x_
        N = xf.shape[-2]
        D = xf.shape[-1]

    T = xf.shape[0]
    x_flat = xf.reshape(T, N * D)
    ms = x_flat.pow(2).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(ms + norm_eps)
    x_scaled = x_flat * inv
    mixes = x_scaled @ phi_.t()

    a = alpha_
    b = base_
    pre = torch.sigmoid(mixes[:, :N] * a[0] + b[:N]) + hc_eps
    # post_out = 2 * sigmoid(...) per aclnn spec.
    post_out = 2.0 * torch.sigmoid(mixes[:, N : 2 * N] * a[1] + b[N : 2 * N])
    logits = (mixes[:, 2 * N :] * a[2] + b[2 * N :]).reshape(T, N, N)
    if clamp_min != 0.0 or clamp_max != 0.0:
        logits_c = torch.clamp(logits, clamp_min, clamp_max)
    else:
        logits_c = logits
    row_max = logits_c.max(dim=-1, keepdim=True).values
    M = (logits_c - row_max).exp()
    M = M / M.sum(dim=-1, keepdim=True) + hc_eps
    M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iter_times - 1):
        M = M / (M.sum(dim=-1, keepdim=True) + hc_eps)
        M = M / (M.sum(dim=-2, keepdim=True) + hc_eps)
    # hin = sum_n(x * pre) -> (T, D). aclnn semantic.
    y = (xf * pre.unsqueeze(-1)).sum(dim=-2)  # (T, D)

    gy = grad_y.to(torch.float32).reshape(T, D)
    loss = (
        (y * gy).sum() + (post_out * grad_post_out).sum() + (M * grad_comb_frag).sum()
    )
    dx, dphi, da, db = torch.autograd.grad(loss, (x_, phi_, alpha_, base_))
    return dx.to(orig_dtype), dphi, da, db
