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

import torch
import triton
import triton.language as tl


@triton.jit
def topk_gating_softmax_kernel(
    input_ptr,
    finished_ptr,  # interface reserved, not yet used
    output_ptr,
    indices_ptr,
    source_rows_ptr,
    num_rows,
    k,
    num_experts,
    start_expert,
    end_expert,
    renormalize: tl.constexpr,
    INDEX_TY: tl.constexpr,
    BLOCK_SIZE_ROWS: tl.constexpr,
    BLOCK_SIZE_EXPERTS: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = tl.arange(0, BLOCK_SIZE_ROWS) + pid * BLOCK_SIZE_ROWS
    valid_rows = rows < num_rows

    cols = start_expert + tl.arange(0, BLOCK_SIZE_EXPERTS)
    valid_cols = cols < end_expert

    logits = tl.load(
        input_ptr + rows[:, None] * num_experts + cols[None, :],
        mask=valid_rows[:, None] & valid_cols[None, :],
        other=-float("inf"),
    ).to(tl.float32)

    row_max = tl.max(logits, axis=1)[:, None]
    exp_vals = tl.exp(logits - row_max)
    probs = exp_vals / (tl.sum(exp_vals, axis=1)[:, None] + 1e-8)

    selected_sum = tl.zeros([BLOCK_SIZE_ROWS], dtype=tl.float32)
    for ki in range(k):
        curr_max = tl.max(probs, axis=1)
        curr_arg = tl.argmax(probs, axis=1) + start_expert

        tl.store(output_ptr + rows * k + ki, curr_max, mask=valid_rows)
        tl.store(indices_ptr + rows * k + ki, curr_arg.to(INDEX_TY), mask=valid_rows)
        tl.store(
            source_rows_ptr + rows * k + ki,
            (ki * num_rows + rows).to(tl.int32),
            mask=valid_rows,
        )
        if renormalize:
            selected_sum += curr_max

        probs = tl.where(
            cols[None, :] == (curr_arg[:, None] - start_expert), -float("inf"), probs
        )

    if renormalize:
        norm = selected_sum + 1e-8
        for ki in range(k):
            idx = rows * k + ki
            val = tl.load(output_ptr + idx, mask=valid_rows)
            tl.store(output_ptr + idx, val / norm, mask=valid_rows)


def _pick_block_size(num_tokens, num_experts, topk):
    """BLOCK_SIZE_EXPERTS / BLOCK_SIZE_ROWS / num_warps 启发式分派。

    列宽(experts 数)对齐到 >=128 的 2 幂: GCU 实测表明窄 tile(如 64 列)的
    axis=1 归约布局明显退化, pad 到 128+ 可获 ~4x 提升; 行数按
    'topk 越大 tile 越小' 与 grid 并行度折衷选取。gcu400 验证结论:
    topk<=8 走 4096 元素大 tile, topk>8 且 E==128 走 512 小 tile,
    其余 1024。默认 num_warps=8; decode 小 batch (T<=16) 降为
    rows=4/warps=4 (~1.6x, 16 行 tile 仅 1 行有效时空转严重)。

    用纯 Python 位运算实现 (避免 torch/triton API 热路径开销,
    topk_softmax 在 decode 场景每 token 调起)。
    """
    # round-up to power-of-2, 再 round 到 32 倍数 (2 幂>=32 时等价其本身)
    p2 = 1 << (num_experts - 1).bit_length()
    experts_block = (p2 + 31) // 32 * 32
    if experts_block < 128:
        experts_block = 128
    elif experts_block > 1024:
        experts_block = 1024
    if topk <= 8:
        target_tile = 4096
    elif experts_block == 128:
        target_tile = 512
    else:
        target_tile = 1024
    rows_block = target_tile // experts_block
    if rows_block > 16:
        rows_block = 16
    if experts_block < 1024:
        if rows_block < 4:
            rows_block = 4
    elif rows_block < 2:
        rows_block = 2
    num_warps = 8
    if num_tokens <= 16:
        # decode 小 batch: 单/少量 token 的行, 收敛 tile 减少空转
        if rows_block > 4:
            rows_block = 4
        num_warps = 4
    elif num_tokens >= 4096 and rows_block == 16 and experts_block <= 512:
        # 大批量 + 宽 tile (如 8192x256x8): rows=32/warps=16 更优 (~6%)
        rows_block = 32
        num_warps = 16
    elif num_tokens >= 32:
        # 小 batch 保 grid 并行度 (>=32 programs)
        g = num_tokens // 32
        if rows_block > g:
            rows_block = g
    if rows_block < 1:
        rows_block = 1
    return experts_block, rows_block, num_warps


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_indices: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    num_tokens = gating_output.shape[0]
    num_experts = gating_output.shape[1]
    topk = topk_weights.shape[1]
    assert topk <= 32
    if topk_indices.dtype == torch.int32:
        index_ty = tl.int32
    # elif topk_indices.dtype == torch.uint32:
    #     index_ty = tl.uint32
    elif topk_indices.dtype == torch.int64:
        index_ty = tl.int64
    else:
        raise TypeError("topk_indices must be int32/int64/uint32")

    experts_block, rows_block, num_warps = _pick_block_size(
        num_tokens, num_experts, topk
    )

    grid = (-(-num_tokens // rows_block),)

    # 位置参数 launch: 减少 kwargs 绑定的 host 开销 (decode 热路径)
    topk_gating_softmax_kernel[grid](
        gating_output,
        None,
        topk_weights,
        topk_indices,
        token_expert_indices,
        num_tokens,
        topk,
        num_experts,
        0,  # start_expert
        num_experts,  # end_expert
        renormalize,
        index_ty,
        rows_block,
        experts_block,
        num_warps=num_warps,
    )
