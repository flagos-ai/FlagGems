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

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8_fp4_mqa_logits"),
    key=["M", "N", "H", "D"],
)
@triton.jit
def _fp8_fp4_mqa_logits_kernel(
    Q_ptr,
    K_ptr,
    K_scale_ptr,
    W_ptr,
    O_ptr,
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_om,
    stride_on,
    stride_wm,
    stride_wh,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_BLOCK: tl.constexpr,
):
    """Fused FP8 MQA logits with head-batched tiled dot products.

    Computes logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n])
                                  * weights[m, h])

    K is loaded once per (BLOCK_M, BLOCK_N) tile and reused across HEAD_BLOCK
    heads to minimize global memory traffic.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Load K tile once — reused across all head batches
    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None] & (d_offs[None, :] < D),
        other=0.0,
    )

    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    # Accumulator for output: [BLOCK_M, BLOCK_N] in fp32
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for hb in range(0, H, HEAD_BLOCK):
        hb_offs = hb + tl.arange(0, HEAD_BLOCK)

        # Load Q for this head batch: [BLOCK_M, HEAD_BLOCK, D]
        q = tl.load(
            Q_ptr
            + m_offs[:, None, None] * stride_qm
            + hb_offs[None, :, None] * stride_qh
            + d_offs[None, None, :] * stride_qd,
            mask=m_mask[:, None, None]
            & (hb_offs[None, :, None] < H)
            & (d_offs[None, None, :] < D),
            other=0.0,
        )

        # Flatten to 2D for MMA: [BLOCK_M * HEAD_BLOCK, D]
        q_2d = tl.reshape(q, [BLOCK_M * HEAD_BLOCK, D])

        # Dot product: [BLOCK_M * HEAD_BLOCK, BLOCK_N]
        dot = tl.dot(q_2d, tl.trans(k))

        # Fused k_scale + ReLU activation
        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        # Load weights for this head batch: [BLOCK_M, HEAD_BLOCK]
        w = tl.load(
            W_ptr + m_offs[:, None] * stride_wm + hb_offs[None, :] * stride_wh,
            mask=m_mask[:, None] & (hb_offs[None, :] < H),
            other=0.0,
        )

        # Weight each head's contribution
        w_flat = tl.reshape(w, [BLOCK_M * HEAD_BLOCK])
        dot = dot * w_flat[:, None]

        # Reduce over head dimension: [BLOCK_M, HEAD_BLOCK, BLOCK_N] -> sum
        dot_3d = tl.reshape(dot, [BLOCK_M, HEAD_BLOCK, BLOCK_N])
        dot_reduced = tl.sum(dot_3d, axis=1)

        acc += dot_reduced

    # Store output tile
    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)


@libentry()
@triton.jit
def _clean_logits_kernel(
    O_ptr,
    KS_ptr,
    KE_ptr,
    M,
    N,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fill invalid positions with -inf based on per-row [ks, ke) ranges."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    ks = tl.load(KS_ptr + m_offs, mask=m_mask, other=0)
    ke = tl.load(KE_ptr + m_offs, mask=m_mask, other=0)

    invalid_mask = (n_offs[None, :] < ks[:, None]) | (n_offs[None, :] >= ke[:, None])
    write_mask = m_mask[:, None] & n_mask[None, :] & invalid_mask

    neg_inf = float("-inf")
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(
        out_ptrs,
        neg_inf + tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32),
        mask=write_mask,
    )


def fp8_fp4_mqa_logits(
    q: tuple,
    kv: tuple,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Triton implementation of fp8_fp4_mqa_logits.

    Computes weighted MQA logits with FP8 quantized Q and K tensors.
    Uses head-batched tiled dot products with K reuse for high throughput.

    Args:
        q: Tuple of (q_values [M, H, D] fp8, q_scale or None).
        kv: Tuple of (k_values [N, D] fp8, k_scales [N] fp32).
        weights: [M, H] fp32 per-head weights.
        cu_seqlen_ks: [M] int32 start indices for valid K range.
        cu_seqlen_ke: [M] int32 end indices for valid K range.
        clean_logits: Whether to fill invalid positions with -inf.

    Returns:
        logits: [M, N] fp32 output tensor.
    """
    logger.debug("GEMS FP8_FP4_MQA_LOGITS")

    q_values, _ = q
    k_values, k_scales = kv

    M, H, D = q_values.shape
    N = k_values.shape[0]

    logits = torch.empty((M, N), dtype=torch.float32, device=q_values.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _fp8_fp4_mqa_logits_kernel[grid](
        q_values,
        k_values,
        k_scales,
        weights,
        logits,
        M,
        N,
        H,
        D,
        q_values.stride(0),
        q_values.stride(1),
        q_values.stride(2),
        k_values.stride(0),
        k_values.stride(1),
        logits.stride(0),
        logits.stride(1),
        weights.stride(0),
        weights.stride(1),
    )

    if clean_logits:
        CLEAN_BLOCK_M = 8
        CLEAN_BLOCK_N = 128
        clean_grid = (
            triton.cdiv(M, CLEAN_BLOCK_M),
            triton.cdiv(N, CLEAN_BLOCK_N),
        )
        _clean_logits_kernel[clean_grid](
            logits,
            cu_seqlen_ks,
            cu_seqlen_ke,
            M,
            N,
            logits.stride(0),
            logits.stride(1),
            BLOCK_M=CLEAN_BLOCK_M,
            BLOCK_N=CLEAN_BLOCK_N,
        )

    return logits
