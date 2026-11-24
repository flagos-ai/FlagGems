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
def _fp8_fp4_mqa_logits_kernel(
    Q_ptr,
    K_ptr,
    K_scale_ptr,
    W_ptr,
    KS_ptr,
    KE_ptr,
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
    CLEAN_LOGITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_GROUP: tl.constexpr,
    PAD_M: tl.constexpr,
):
    """GCU400 fused FP8 MQA logits。

    logits[m, n] = sum_h( ReLU(dot(q[m,h,:], k[n,:]) * k_scale[n]) * w[m,h] )，
    CLEAN_LOGITS 为真时在同一 kernel 内把 [cu_seqlen_ks, cu_seqlen_ke) 之外的
    位置写成 -inf，省掉一次独立 launch。

    与通用实现的差异（均为 GCU400 实测结论）：
      * `tl.dot` 的 M 维必须是 16 的倍数，否则静默算错；所以行索引全部走
        PAD_M（>=16）补齐，真实行数由 BLOCK_M 掩码控制，M=1 的 decode 也正确。
      * 行数少（decode，M<=16）时只能切出 cdiv(N, BLOCK_N) 个 program，MMA
        行维严重空转。HEAD_GROUP 把相邻 HEAD_GROUP 个 head 叠到 dot 的行轴上
        （行号 = head*PAD_M + row），算完 reshape 回 [HEAD_GROUP, PAD_M, BLOCK_N]
        沿 head 归约；M=1、N=4096 实测 0.19ms -> 0.080ms。
      * HEAD_GROUP=1 时跳过 reshape，等价逐 head 循环（prefill 大 M 更优）。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    stacked = tl.arange(0, HEAD_GROUP * PAD_M)
    head_off = stacked // PAD_M
    row_off = stacked % PAD_M
    m_offs = pid_m * BLOCK_M + row_off
    stack_mask = (row_off < BLOCK_M) & (m_offs < M)

    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    d_offs = tl.arange(0, D)

    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    acc = tl.zeros([PAD_M, BLOCK_N], dtype=tl.float32)
    for h0 in range(0, H, HEAD_GROUP):
        q = tl.load(
            Q_ptr
            + m_offs[:, None] * stride_qm
            + (h0 + head_off)[:, None] * stride_qh
            + d_offs[None, :] * stride_qd,
            mask=stack_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        dot = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        w = tl.load(
            W_ptr + m_offs * stride_wm + (h0 + head_off) * stride_wh,
            mask=stack_mask,
            other=0.0,
        )
        if HEAD_GROUP == 1:
            acc += dot * w[:, None]
        else:
            acc += tl.sum(
                tl.reshape(dot * w[:, None], [HEAD_GROUP, PAD_M, BLOCK_N]), axis=0
            )

    out_rows = pid_m * BLOCK_M + tl.arange(0, PAD_M)
    row_ok = out_rows < M
    if CLEAN_LOGITS:
        ks = tl.load(KS_ptr + out_rows, mask=row_ok, other=0)
        ke = tl.load(KE_ptr + out_rows, mask=row_ok, other=N)
        n_valid = (n_offs[None, :] >= ks[:, None]) & (n_offs[None, :] < ke[:, None])
        acc = tl.where(n_valid, acc, float("-inf"))

    write_mask = row_ok[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + out_rows[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=write_mask)


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
    On GCU400 the clean_logits pass is fused into the main kernel so a
    single call launches exactly one kernel.

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
    logger.debug("GEMS_ENFLAME FP8_FP4_MQA_LOGITS")

    q_values, _ = q
    k_values, k_scales = kv

    M, H, D = q_values.shape
    N = k_values.shape[0]

    logits = torch.empty((M, N), dtype=torch.float32, device=q_values.device)

    block_m, block_n, head_group, pad_m = _select_tile(M)
    grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))

    _fp8_fp4_mqa_logits_kernel[grid](
        q_values,
        k_values,
        k_scales,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
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
        clean_logits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        HEAD_GROUP=head_group,
        PAD_M=pad_m,
        num_warps=4,
    )

    return logits


def _select_tile(m: int) -> tuple:
    """按行数选 tile（GCU400 实测最优，见 kernel docstring）。

    decode（行少）用 8 个 head 叠行 + 128 列 tile，让 MMA 行维不空转；
    prefill（行多）退回逐 head 循环，用 64x256 大 tile 提高复用。
    返回 (BLOCK_M, BLOCK_N, HEAD_GROUP, PAD_M)。
    """
    if m <= 16:
        return 16, 128, 8, 16
    if m <= 64:
        return 16, 256, 4, 16
    return 64, 256, 1, 64
