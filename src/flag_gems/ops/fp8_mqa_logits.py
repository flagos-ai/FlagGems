import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fp8_mqa_logits"),
    key=["M", "N", "D"],
)
@triton.jit
def _fp8_mqa_logits_kernel(
    Q,
    K,
    K_SCALES,
    WEIGHTS,
    CU_SEQLEN_KS,
    CU_SEQLEN_KE,
    LOGITS,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kd,
    M: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    N: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Optimized Triton kernel for FP8 MQA logits computation.

    Each program computes logits[m, n] = sum_h(ReLU(score[m, h, n]) * weights[m, h])
    where score[m, h, n] = sum_d(q[m, h, d] * k[n, d])

    Optimization: Each program handles a BLOCK_M x BLOCK_N tile.
    K is loaded once and reused across H dimension.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < M
    mask_n = offs_n < N

    ks_start = tl.load(CU_SEQLEN_KS + offs_m, mask=mask_m, other=0)
    ke_end = tl.load(CU_SEQLEN_KE + offs_m, mask=mask_m, other=N)

    k_scales = tl.load(K_SCALES + offs_n, mask=mask_n, other=1.0)
    k_scales = k_scales.to(tl.float32)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for h_idx in range(H):
        weight_ptrs = WEIGHTS + offs_m * H + h_idx
        weight_h = tl.load(weight_ptrs, mask=mask_m, other=0.0)
        weight_h = weight_h.to(tl.float32)

        score_h = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, D, BLOCK_D):
            d_mask = (d_start + offs_d) < D
            d_offs = d_start + offs_d

            q_ptrs = (
                Q
                + offs_m[:, None] * stride_qm
                + h_idx * stride_qh
                + d_offs[None, :] * stride_qd
            )
            q = tl.load(q_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0)
            q = q.to(tl.float32)

            k_ptrs = K + offs_n[:, None] * stride_kn + d_offs[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None] & d_mask[None, :], other=0.0)
            k = k.to(tl.float32) * k_scales[:, None]

            score_h += tl.dot(q, tl.trans(k))

        score_h = tl.maximum(score_h, 0.0)
        acc += score_h * weight_h[:, None]

    if CLEAN_LOGITS:
        n_valid = (offs_n[None, :] >= ks_start[:, None]) & (
            offs_n[None, :] < ke_end[:, None]
        )
        acc = tl.where(n_valid, acc, float("-inf"))

    out_ptrs = LOGITS + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    logger.debug("FP8 MQA LOGITS")

    k_fp8, k_scales = kv

    M, H, D = q.shape
    N = k_fp8.shape[0]

    logits = torch.zeros((M, N), dtype=torch.float32, device=q.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]),
        triton.cdiv(N, META["BLOCK_N"]),
    )

    _fp8_mqa_logits_kernel[grid](
        q,
        k_fp8,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        logits,
        q.stride(0),  # stride_qm
        q.stride(1),  # stride_qh
        q.stride(2),  # stride_qd
        k_fp8.stride(0),  # stride_kn
        k_fp8.stride(1),  # stride_kd
        M,
        H,
        D,
        N,
        clean_logits,
    )

    return logits
