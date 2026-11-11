# Copyright 2026, The FlagOS Contributors.
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
"""GCU400 专属 ``gru_data``（packed sequence GRU 前向）。

公共实现的核心递推（input GEMM / step / persistent / gemv kernel）已是 Triton
实现且按 shape 有策略分支，未做改动；优化集中在 packed 数据的前后处理：
unpack / pack 两个 kernel 的 grid 为 ``num_steps × batch``，每个 CTA 沿特征维按
``BLOCK_F=256`` 循环搬运。对较大的 input_size / hidden_size，固定 256 会引入多次
循环迭代；这里让 ``BLOCK_F`` 自适应（或按扫描结果固定）以一次覆盖整段特征维，
并把 offsets / bs32 两块小缓冲合并为一次分配。

``BASELINE_MODE=1`` 直接转发到公共实现，用于采集等价基线。
"""

import importlib
import logging

import torch
import triton

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
_public = importlib.import_module("flag_gems.ops.gru")

# --- TUNING PARAMS START ---
# 编译期配置，由优化迭代按轮次调整，不改变算子语义。
BASELINE_MODE = 0
PACK_BLOCK_MODE = 2
PACK_BLOCK_FIXED = 64
OFFSETS_BLOCK_MODE = 0
OFFSETS_BLOCK_FIXED = 1024
FUSE_OFFSETS = 1
HOST_OFFSETS = 1
UNPACK_NUM_WARPS = 4
PACK_NUM_WARPS = 4
# --- TUNING PARAMS END ---


def _pack_block_size(input_size, hidden_total):
    """选择 unpack / pack kernel 沿特征维的 tile，必须是 2 的幂。"""
    if PACK_BLOCK_MODE == 0:
        return 256
    if PACK_BLOCK_MODE == 1:
        return max(256, triton.next_power_of_2(max(input_size, hidden_total)))
    return max(32, triton.next_power_of_2(PACK_BLOCK_FIXED))


def _offsets_block_size(num_steps):
    if OFFSETS_BLOCK_MODE == 0:
        return _public._ceil_power_of_2(num_steps)
    return max(32, triton.next_power_of_2(OFFSETS_BLOCK_FIXED))


def gru_data(
    data,
    batch_sizes,
    hx,
    params,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
):
    """packed GRU 前向，签名与公共实现一致。"""
    logger.debug("GEMS_ENFLAME_GCU400 GRU_DATA")
    if BASELINE_MODE == 1:
        return _public.gru_data(
            data,
            batch_sizes,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
        )

    if data.dim() != 2:
        raise RuntimeError("gru.data: packed data must have 2 dimensions")
    if batch_sizes.dim() != 1:
        raise RuntimeError("gru.data: batch_sizes must be 1-dimensional")
    if num_layers <= 0:
        raise RuntimeError("gru.data: num_layers must be greater than zero")
    if not 0.0 <= dropout <= 1.0:
        raise RuntimeError("gru.data: dropout probability must be between 0 and 1")
    if data.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems gru.data supports float16, bfloat16, float32, and float64"
        )
    num_directions = 2 if bidirectional else 1
    num_states = num_layers * num_directions
    if hx.dim() != 3:
        raise RuntimeError("gru.data: hidden state must have 3 dimensions")
    if hx.shape[0] != num_states:
        raise RuntimeError("gru.data: hidden state rows mismatch")
    expected_params = num_states * (4 if has_biases else 2)
    if len(params) != expected_params:
        raise RuntimeError("gru.data: parameter count mismatch")

    num_steps = batch_sizes.numel()
    input_size = data.shape[1]
    batch = hx.shape[1]
    hidden_size = hx.shape[2]
    hidden_total = hidden_size * num_directions
    device = data.device
    # pack_padded_sequence 生成的 batch_sizes 常驻 CPU；保留它以便 host 侧算前缀和。
    host_batch_sizes = batch_sizes if batch_sizes.device.type == "cpu" else None
    if not (HOST_OFFSETS and host_batch_sizes is not None):
        batch_sizes = batch_sizes.to(device)

    if HOST_OFFSETS and host_batch_sizes is not None:
        # 在 host 侧算 exclusive prefix sum，一次构造 (2, num_steps) int32 后单次 H2D，
        # 省掉 _batch_offsets_kernel 的 launch 与 int64→int32 转换。
        lengths = [int(value) for value in host_batch_sizes.tolist()]
        exclusive_offsets = []
        running_sum = 0
        for value in lengths:
            exclusive_offsets.append(running_sum)
            running_sum += value
        offsets_pair = torch.tensor(
            [exclusive_offsets, lengths], dtype=torch.int32, device=device
        )
        offsets, bs32 = offsets_pair[0], offsets_pair[1]
    else:
        # offsets / bs32 合并为一次分配，两个行视图仍保持连续。
        if FUSE_OFFSETS:
            offsets_pair = _public._empty((2, num_steps), torch.int32, device)
            offsets = offsets_pair[0]
            bs32 = offsets_pair[1]
        else:
            offsets = _public._empty((num_steps,), torch.int32, device)
            bs32 = _public._empty((num_steps,), torch.int32, device)
        with torch_device_fn.device(device):
            _public._batch_offsets_kernel[(1,)](
                batch_sizes,
                offsets,
                bs32,
                num_steps,
                BLOCK=_offsets_block_size(num_steps),
            )

    x_padded = _public._empty((num_steps, batch, input_size), data.dtype, device)
    block_f = _pack_block_size(input_size, hidden_total)
    with torch_device_fn.device(device):
        _public._unpack_padded_kernel[(num_steps * batch,)](
            data,
            x_padded,
            offsets,
            bs32,
            input_size,
            batch,
            data.stride(0),
            x_padded.stride(0),
            x_padded.stride(1),
            x_padded.stride(2),
            BLOCK_F=block_f,
            num_warps=UNPACK_NUM_WARPS,
        )

    final_h = _public._empty((num_states, batch, hidden_size), data.dtype, device)
    out_padded = _public._empty((num_steps, batch, hidden_total), data.dtype, device)
    _public._gru_forward_impl(
        x_padded,
        hx,
        params,
        out_padded,
        final_h,
        num_layers,
        num_directions,
        hidden_size,
        input_size,
        batch,
        num_steps,
        has_biases,
        train,
        dropout,
        batch_sizes=bs32,
    )

    out_packed = _public._empty((data.shape[0], hidden_total), data.dtype, device)
    with torch_device_fn.device(device):
        _public._pack_output_kernel[(num_steps * batch,)](
            out_padded,
            out_packed,
            offsets,
            bs32,
            hidden_total,
            batch,
            out_padded.stride(0),
            out_padded.stride(1),
            out_padded.stride(2),
            out_packed.stride(0),
            BLOCK_F=block_f,
            num_warps=PACK_NUM_WARPS,
        )
    return out_packed, final_h
