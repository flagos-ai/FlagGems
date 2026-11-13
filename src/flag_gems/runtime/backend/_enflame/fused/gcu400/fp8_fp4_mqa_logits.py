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
    HEAD_BLOCK: tl.constexpr,
):
    """GCU400 adaptation of the fused FP8 MQA logits kernel.

    Computes logits[m, n] = sum_h(ReLU(sum_d(q[m,h,d]*k[n,d]) * k_scale[n])
                                  * weights[m, h])
    and, when CLEAN_LOGITS is set, fills positions outside the per-row
    [cu_seqlen_ks, cu_seqlen_ke) K range with -inf in the same launch.

    GCU-specific deviations from the generic version (validated op-by-op):
      * Per-head loop with 2D [rows, D] dot operands converted to fp32
        before `tl.dot`. The generic head-batched 3D-load -> reshape ->
        dot -> reshape -> sum(axis=1) chain silently miscompiles for
        BLOCK_M=1 tiles (all values wrong), and bare fp8 `tl.dot` is not
        reliable on this backend.
      * The dot row count is padded to at least 16 via PAD_M: `tl.dot`
        with an M-dim of 1 produces silently wrong results on GCU (both
        fp8 and fp32 operands), so BLOCK_M in {1, 2, 4, 8} must not map
        1:1 onto the dot M-dim.
      * HEAD_BLOCK is kept (unused) so the shared generic tune-config
        space and cached tuning records stay launch-compatible.
    """
    PAD_M: tl.constexpr = 16 if BLOCK_M < 16 else BLOCK_M
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row = tl.arange(0, PAD_M)
    m_offs = pid_m * BLOCK_M + row
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, D)

    row_active = row < BLOCK_M
    m_mask = row_active & (m_offs < M)
    n_mask = n_offs < N

    k = tl.load(
        K_ptr + n_offs[:, None] * stride_kn + d_offs[None, :] * stride_kd,
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_scale = tl.load(K_scale_ptr + n_offs, mask=n_mask, other=0.0)

    acc = tl.zeros([PAD_M, BLOCK_N], dtype=tl.float32)
    for h in range(0, H):
        q = tl.load(
            Q_ptr
            + m_offs[:, None] * stride_qm
            + h * stride_qh
            + d_offs[None, :] * stride_qd,
            mask=m_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        dot = tl.dot(q, tl.trans(k), out_dtype=tl.float32)
        dot = tl.maximum(dot * k_scale[None, :], 0.0)

        w = tl.load(W_ptr + m_offs * stride_wm + h * stride_wh, mask=m_mask, other=0.0)
        acc += dot * w[:, None]

    if CLEAN_LOGITS:
        ks = tl.load(KS_ptr + m_offs, mask=m_mask, other=0)
        ke = tl.load(KE_ptr + m_offs, mask=m_mask, other=0)
        n_valid = (n_offs[None, :] >= ks[:, None]) & (n_offs[None, :] < ke[:, None])
        acc = tl.where(n_valid, acc, float("-inf"))

    write_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = O_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
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
    )

    return logits
