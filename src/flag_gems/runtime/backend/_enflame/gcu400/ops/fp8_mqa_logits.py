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

logger = logging.getLogger(__name__)


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
    M,
    N,
    H: tl.constexpr,
    D: tl.constexpr,
    CLEAN_LOGITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    PAD_M: tl.constexpr,
):
    """GCU400 FP8 MQA logits.

    logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n]) * w[m,h]),
    Each K is loaded once per n and reused along H; when CLEAN_LOGITS is true, positions outside
    [cu_seqlen_ks, cu_seqlen_ke) are written as -inf in the same launch.

    Three GCU400-critical changes versus the original implementation:
      * M/N go from `tl.constexpr` back to runtime scalars; otherwise every new (M, N) triggers a
        recompile (the long-context decode at N=4096 has a very high first-call latency).
      * `tl.dot` needs a row dim >= 16 and a multiple of 16, so row indices are padded via PAD_M
        and the real row count is controlled with a BLOCK_M mask; the original silently computed
        wrong results for M<16.
      * HEAD_GROUP stacks neighbouring heads onto the dot row axis (row index = head*PAD_M + row)
        and reshapes back to [HEAD_GROUP, PAD_M, BLOCK_N] to reduce along head, filling the MMA
        row dim for small decode M (M=1, N=4096 measured 0.19ms -> 0.080ms); prefill uses
        HEAD_GROUP=1 and falls back to a per-head loop; logits are no longer pre-zeroed with `torch.zeros`.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stacked = tl.arange(0, HEAD_GROUP * PAD_M)
    head_off = stacked // PAD_M
    row_off = stacked % PAD_M
    offs_m = pid_m * BLOCK_M + row_off
    stack_mask = (row_off < BLOCK_M) & (offs_m < M)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N
    offs_d = tl.arange(0, D)

    k_scales = tl.load(K_SCALES + offs_n, mask=mask_n, other=1.0).to(tl.float32)
    k = tl.load(
        K + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
        mask=mask_n[:, None],
        other=0.0,
    ).to(tl.float32)

    acc = tl.zeros([PAD_M, BLOCK_N], dtype=tl.float32)
    for h_start in range(0, H, HEAD_GROUP):
        q = tl.load(
            Q
            + offs_m[:, None] * stride_qm
            + (h_start + head_off)[:, None] * stride_qh
            + offs_d[None, :] * stride_qd,
            mask=stack_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        score = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        score = tl.maximum(score * k_scales[None, :], 0.0)

        weight = tl.load(
            WEIGHTS + offs_m * H + (h_start + head_off), mask=stack_mask, other=0.0
        ).to(tl.float32)
        if HEAD_GROUP == 1:
            acc += score * weight[:, None]
        else:
            acc += tl.sum(
                tl.reshape(score * weight[:, None], [HEAD_GROUP, PAD_M, BLOCK_N]),
                axis=0,
            )

    out_rows = pid_m * BLOCK_M + tl.arange(0, PAD_M)
    row_ok = out_rows < M
    if CLEAN_LOGITS:
        ks_start = tl.load(CU_SEQLEN_KS + out_rows, mask=row_ok, other=0)
        ke_end = tl.load(CU_SEQLEN_KE + out_rows, mask=row_ok, other=N)
        n_valid = (offs_n[None, :] >= ks_start[:, None]) & (
            offs_n[None, :] < ke_end[:, None]
        )
        acc = tl.where(n_valid, acc, float("-inf"))

    out_ptrs = LOGITS + out_rows[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=row_ok[:, None] & mask_n[None, :])


def _select_tile(m: int) -> tuple:
    """Picks the tile by row count (measured optimum on GCU400, see the kernel docstring).

    Returns (BLOCK_M, BLOCK_N, HEAD_GROUP, PAD_M).
    """
    if m <= 16:
        return 16, 128, 8, 16
    if m <= 64:
        return 16, 256, 4, 16
    return 64, 256, 1, 64


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME FP8_MQA_LOGITS")

    k_fp8, k_scales = kv

    M, H, D = q.shape
    N = k_fp8.shape[0]

    # Both clean_logits branches write the whole tensor (invalid positions set to -inf), so no torch.zeros pre-clear is
    # needed
    logits = torch.empty((M, N), dtype=torch.float32, device=q.device)

    block_m, block_n, head_group, pad_m = _select_tile(M)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

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
        N,
        H,
        D,
        clean_logits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_GROUP=head_group,
        PAD_M=pad_m,
        num_warps=4,
    )

    return logits
