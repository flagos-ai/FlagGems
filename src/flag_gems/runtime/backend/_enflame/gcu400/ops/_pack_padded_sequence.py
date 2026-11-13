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

"""GCU400 专属 ``_pack_padded_sequence``：降低主机元数据与 H2D 开销。

打包行按时间步主序排列，``row -> (t, b)`` 的映射由 lengths 派生。原始实现
在主机侧生成两张等长索引表并各做一次 H2D（同步拷贝），小 shape 的固定开销
占主导。本实现提供三条可切换路径：二维 grid 双表（原始）、一维 grid 合并
双表（一次 H2D）、以及只传 ``batch_sizes`` 前缀和的小表并在 kernel 内还原
行索引。小 shape 走合并/小表路径，大 shape 保留向量化主机构造。
"""

import logging

import torch
import triton
import triton.language as tl

import flag_gems


logger = logging.getLogger(__name__)

# 按 lengths 内容缓存只读的累计行数表；表内容完全由 CPU lengths 决定，
# 不含任何输入张量数据，因此地址复用与原地修改都不会影响正确性。
_TABLE_CACHE = {}
_TABLE_CACHE_CAP = 64


def _get_cumsum_table(lengths_cpu, cumsum, device):
    """命中缓存时省掉一次同步 H2D，未命中时才构造并上传小表。"""
    cache_key = (str(device), tuple(lengths_cpu.tolist()))
    cached_table = _TABLE_CACHE.get(cache_key)
    if cached_table is None:
        cached_table = cumsum.to(device)
        if len(_TABLE_CACHE) >= _TABLE_CACHE_CAP:
            _TABLE_CACHE.clear()
        _TABLE_CACHE[cache_key] = cached_table
    return cached_table


# --- TUNING PARAMS START ---
# 编译期 / 启动期配置，由优化迭代按轮次调整，不改变算子语义。
META_MODE = 2
GRID_MODE = 1
BLOCK_FEAT_CAP = 1024
NUM_WARPS = 1
NUM_STAGES = 1
FAST_HOST_META = 0
FAST_META_LIMIT = 2048
TABLE_CAP = 64
NON_BLOCKING = 0
# --- TUNING PARAMS END ---


@triton.jit
def _pack_copy_kernel(
    input_ptr,
    data_ptr,
    t_ptr,
    b_ptr,
    stride_t,
    stride_b,
    feat_size,
    BLOCK_FEAT: tl.constexpr,
):
    """二维 grid 原始实现：一行一个 program，从两张索引表读取 (t, b)。"""
    row = tl.program_id(0)
    feat_block = tl.program_id(1)

    t = tl.load(t_ptr + row)
    b = tl.load(b_ptr + row)

    offsets = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = offsets < feat_size

    src = t * stride_t + b * stride_b + offsets
    dst = row * feat_size + offsets

    vals = tl.load(input_ptr + src, mask=feat_mask, other=0.0)
    tl.store(data_ptr + dst, vals, mask=feat_mask)


@triton.jit
def _pack_copy_kernel_1d(
    input_ptr,
    data_ptr,
    meta_ptr,
    stride_t,
    stride_b,
    feat_size,
    NUM_FEAT_BLOCKS: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
):
    """一维 grid：合并后的 [total_len, 2] 索引表一次 H2D，pid 线性展开 (row, feat_block)。"""
    pid = tl.program_id(0)
    row = pid // NUM_FEAT_BLOCKS
    feat_block = pid % NUM_FEAT_BLOCKS

    t = tl.load(meta_ptr + row * 2)
    b = tl.load(meta_ptr + row * 2 + 1)

    offsets = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = offsets < feat_size
    src = t * stride_t + b * stride_b + offsets
    dst = row * feat_size + offsets

    vals = tl.load(input_ptr + src, mask=feat_mask, other=0.0)
    tl.store(data_ptr + dst, vals, mask=feat_mask)


@triton.jit
def _pack_copy_kernel_table(
    input_ptr,
    data_ptr,
    cumsum_ptr,
    max_len,
    stride_t,
    stride_b,
    feat_size,
    NUM_FEAT_BLOCKS: tl.constexpr,
    BLOCK_FEAT: tl.constexpr,
    LOG_STEPS: tl.constexpr,
):
    """一维 grid：只传各时间步累计行数，kernel 内由 row 反解 (t, b)。

    ``cumsum[t]`` 是前 t+1 个时间步的打包行总数；满足 ``row < cumsum[t]``
    的最小 t 即时间步，行内序号为 ``row - cumsum[t-1]``。表长只有 max_len，
    显著小于 total_len，小 shape 可省去大索引表的 H2D 同步。
    """
    pid = tl.program_id(0)
    row = pid // NUM_FEAT_BLOCKS
    feat_block = pid % NUM_FEAT_BLOCKS

    # cumsum 严格递增，二分反解时间步比 BLOCK_STEPS 元素向量归约更省指令。
    low_step = 0
    high_step = max_len
    for _ in tl.static_range(LOG_STEPS):
        mid_step = (low_step + high_step) // 2
        end_at_mid = tl.load(
            cumsum_ptr + mid_step, mask=mid_step < max_len, other=2147483647
        )
        active_at_mid = row >= end_at_mid
        low_step = tl.where(active_at_mid, mid_step + 1, low_step)
        high_step = tl.where(active_at_mid, high_step, mid_step)
    t = low_step
    prev_end = tl.load(cumsum_ptr + t - 1, mask=t > 0, other=0)
    b = row - prev_end

    offsets = feat_block * BLOCK_FEAT + tl.arange(0, BLOCK_FEAT)
    feat_mask = offsets < feat_size
    src = t * stride_t + b * stride_b + offsets
    dst = row * feat_size + offsets

    vals = tl.load(input_ptr + src, mask=feat_mask, other=0.0)
    tl.store(data_ptr + dst, vals, mask=feat_mask)


def _pack_padded_sequence(input, lengths, batch_first):
    logger.debug("GEMS _PACK_PADDED_SEQUENCE")

    if batch_first:
        batch_size = input.shape[0]
        stride_b, stride_t = input.stride(0), input.stride(1)
    else:
        batch_size = input.shape[1]
        stride_t, stride_b = input.stride(0), input.stride(1)

    feat_shape = input.shape[2:]
    feat_size = 1
    for size in feat_shape:
        feat_size *= size

    # aten 保证 lengths 是 CPU int64 且降序；元数据计算全部留在主机侧，
    # 避免为每个打包行额外发起设备侧查询。
    lengths_cpu = lengths.to(torch.int64).cpu()
    if batch_size == 0:
        data = torch.empty(0, *feat_shape, dtype=input.dtype, device=input.device)
        return data, torch.empty(0, dtype=torch.int64)

    max_len = int(lengths_cpu[0].item())
    total_len = int(lengths_cpu.sum().item())

    steps = torch.arange(max_len, dtype=torch.int64).unsqueeze(1)
    active = lengths_cpu.unsqueeze(0) > steps
    batch_sizes = active.sum(dim=1)
    cumsum = flag_gems.cumsum(batch_sizes, dim=0)

    data = torch.empty(total_len, *feat_shape, dtype=input.dtype, device=input.device)

    if total_len > 0 and feat_size > 0:
        block_feat = min(triton.next_power_of_2(feat_size), BLOCK_FEAT_CAP)
        num_feat_blocks = triton.cdiv(feat_size, block_feat)
        meta_mode = META_MODE
        # 小表路径需要 kernel 内对 max_len 做一次归约，过大的 max_len 不划算。
        if meta_mode == 2 and max_len > TABLE_CAP:
            meta_mode = 1

        if meta_mode == 0:
            # 原始路径：两张独立索引表 + 二维 grid。
            step_ids = torch.arange(max_len, dtype=torch.int64)
            time_index = torch.repeat_interleave(step_ids, batch_sizes)
            run_starts = torch.repeat_interleave(cumsum - batch_sizes, batch_sizes)
            batch_index = torch.arange(total_len, dtype=torch.int64) - run_starts
            row2t = time_index.to(input.device)
            row2b = batch_index.to(input.device)
            grid = (total_len, num_feat_blocks)
            _pack_copy_kernel[grid](
                input,
                data,
                row2t,
                row2b,
                stride_t,
                stride_b,
                feat_size,
                BLOCK_FEAT=block_feat,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
        elif meta_mode == 1:
            # 合并索引表：一次 H2D 完成 (t, b) 两列。
            if FAST_HOST_META and total_len <= FAST_META_LIMIT:
                pairs = []
                for time_step, count in enumerate(batch_sizes.tolist()):
                    pairs.extend(
                        (time_step, batch_index) for batch_index in range(int(count))
                    )
                meta_cpu = torch.tensor(pairs, dtype=torch.int64).reshape(-1, 2)
            else:
                step_ids = torch.arange(max_len, dtype=torch.int64)
                time_index = torch.repeat_interleave(step_ids, batch_sizes)
                run_starts = torch.repeat_interleave(cumsum - batch_sizes, batch_sizes)
                batch_index = torch.arange(total_len, dtype=torch.int64) - run_starts
                meta_cpu = flag_gems.stack((time_index, batch_index), dim=1).contiguous()
            meta = meta_cpu.to(input.device, non_blocking=bool(NON_BLOCKING))
            if GRID_MODE:
                grid = (total_len * num_feat_blocks,)
                _pack_copy_kernel_1d[grid](
                    input,
                    data,
                    meta,
                    stride_t,
                    stride_b,
                    feat_size,
                    num_feat_blocks,
                    block_feat,
                    num_warps=NUM_WARPS,
                    num_stages=NUM_STAGES,
                )
            else:
                # 合并表同样可以按二维 grid 由两个独立视图读取。
                row2t = meta[:, 0]
                row2b = meta[:, 1]
                grid = (total_len, num_feat_blocks)
                _pack_copy_kernel[grid](
                    input,
                    data,
                    row2t,
                    row2b,
                    stride_t,
                    stride_b,
                    feat_size,
                    BLOCK_FEAT=block_feat,
                    num_warps=NUM_WARPS,
                    num_stages=NUM_STAGES,
                )
        else:
            # 小表路径：只传累计行数，kernel 内反解时间步与批内序号。
            table = _get_cumsum_table(lengths_cpu, cumsum, input.device)
            grid = (total_len * num_feat_blocks,)
            _pack_copy_kernel_table[grid](
                input,
                data,
                table,
                max_len,
                stride_t,
                stride_b,
                feat_size,
                num_feat_blocks,
                block_feat,
                max(1, (max_len + 1).bit_length()),
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )

    return data, batch_sizes
