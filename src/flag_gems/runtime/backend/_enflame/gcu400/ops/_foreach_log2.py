# Copyright 2026, The FlagOS Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``_foreach_log2``（out-of-place）。

公共 ``foreach_unary`` 执行器用二维 grid 承载 chunk table，在 GCU400 上会撞
grid.y≤255 限制且 metadata 依赖 64 位偏移。这里提供两条 Triton 路径：

* ``FUSED_LAUNCH=1``：把 tensor 列表的 (输入指针, 输出指针, 元素数) 打包成
  int64 metadata，一维 grid 一次 launch 处理全部 tensor，消除逐 tensor 的
  launch 固定开销；这是 GCU400 上该算子的主路径。
* ``FUSED_LAUNCH=0``：逐 tensor launch 的一维固定 grid 跨步 kernel，作为
  回退与性能对照。

非浮点、复数或混合 dtype 的输入回退公共 ``apply_unary``，以维持 ATen 的
dtype 提升策略（整型/布尔提升为 float32）。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.ops._foreach_unary import apply_unary
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# 编译期配置，由优化迭代按轮次调整，不改变算子语义。
FUSED_LAUNCH = 1
BLOCK_SIZE = 16384
FIXED_GRID = 48
NUM_WARPS = 4
MAX_CHUNKS_CAP = 0
USE_DEVICE_CONTEXT = 0
USE_MULTIPLE_OF = 1
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---

_FLOAT_DTYPES = (torch.float16, torch.float32, torch.bfloat16)
_TL_DTYPES = {
    torch.float16: tl.float16,
    torch.float32: tl.float32,
    torch.bfloat16: tl.bfloat16,
}


@libentry()
@triton.jit(do_not_specialize=["N"])
def _foreach_log2_out_kernel(
    src_ptr,
    dst_ptr,
    N,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """逐 tensor launch：一维固定 grid 跨步读取输入并写出 log2。"""
    arange = tl.arange(0, BLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for block_id in tl.range(pid, tl.cdiv(N, BLOCK), num_programs):
        offs = block_id * BLOCK + arange
        mask = offs < N
        value = tl.load(src_ptr + offs, mask=mask).to(tl.float32)
        tl.store(dst_ptr + offs, tl.log2(value), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["NT", "CHUNKS"])
def _foreach_log2_fused_kernel(
    meta_ptr,
    NT,
    CHUNKS,
    IN_DT: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_MULTIPLE_OF: tl.constexpr,
):
    """单次 launch：metadata 布局为 [输入指针×NT, 输出指针×NT, 元素数×NT]。

    一维 grid 把 ``(tensor_id, chunk_id)`` 扁平化，避免 GCU400 的 grid.y
    上限；超出各 tensor 自身元素数的 program 由 mask 全屏蔽。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for task_id in tl.range(pid, NT * CHUNKS, num_programs):
        tensor_id = task_id // CHUNKS
        chunk_id = task_id % CHUNKS
        in_v = tl.load(meta_ptr + tensor_id)
        out_v = tl.load(meta_ptr + NT + tensor_id)
        n_elements = tl.load(meta_ptr + 2 * NT + tensor_id)
        idx = chunk_id * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n_elements
        if USE_MULTIPLE_OF and (in_v & 15) == 0 and (out_v & 15) == 0:
            in_ptr = tl.multiple_of(in_v.to(tl.pointer_type(IN_DT)), 16)
            out_ptr = tl.multiple_of(out_v.to(tl.pointer_type(IN_DT)), 16)
        else:
            in_ptr = in_v.to(tl.pointer_type(IN_DT))
            out_ptr = out_v.to(tl.pointer_type(IN_DT))
        value = tl.load(in_ptr + idx, mask=mask, other=0).to(tl.float32)
        tl.store(out_ptr + idx, tl.log2(value), mask=mask)


def _foreach_log2(self):
    logger.debug("GEMS_GCU400 FOREACH_LOG2")
    all_float = all(
        tensor.dtype in _FLOAT_DTYPES and not tensor.dtype.is_complex for tensor in self
    )
    homogeneous = len({tensor.dtype for tensor in self}) == 1
    if FALLBACK_TO_PUBLIC or not (all_float and homogeneous):
        return apply_unary("log2", self, inplace=False)

    # out-of-place 需要为每条输入 tensor 单独分配同 dtype/同形状的输出；
    # 输出列表顺序必须与 self 一致。
    outputs = [torch.empty_like(tensor) for tensor in self]
    device = self[0].device
    device_context = (
        torch_device_fn.device(device) if USE_DEVICE_CONTEXT else nullcontext()
    )

    if FUSED_LAUNCH:
        tensor_count = len(self)
        # metadata 用 int64 保存设备指针，必须开启 GCU 64 位支持，否则
        # torch_gcu 会把 int64 降级成 int32 导致指针被截断。
        metadata = torch.empty(3 * tensor_count, dtype=torch.int64)
        for index, (src_tensor, dst_tensor) in enumerate(zip(self, outputs)):
            metadata[index] = src_tensor.data_ptr()
            metadata[tensor_count + index] = dst_tensor.data_ptr()
            metadata[2 * tensor_count + index] = src_tensor.numel()
        metadata = metadata.to(device)
        max_numel = max(tensor.numel() for tensor in self)
        chunks = triton.cdiv(max_numel, BLOCK_SIZE)
        if MAX_CHUNKS_CAP:
            chunks = min(chunks, MAX_CHUNKS_CAP)
        grid_size = min(tensor_count * chunks, FIXED_GRID * tensor_count)
        with device_context:
            _foreach_log2_fused_kernel[(grid_size,)](
                metadata,
                tensor_count,
                chunks,
                _TL_DTYPES[self[0].dtype],
                BLOCK_SIZE,
                USE_MULTIPLE_OF,
                num_warps=NUM_WARPS,
            )
        return outputs

    launch_pairs = list(zip(self, outputs))
    with device_context:
        for src_tensor, dst_tensor in launch_pairs:
            num_elems = src_tensor.numel()
            grid = (min(triton.cdiv(num_elems, BLOCK_SIZE), FIXED_GRID),)
            _foreach_log2_out_kernel[grid](
                src_tensor,
                dst_tensor,
                num_elems,
                BLOCK_SIZE,
                FIXED_GRID,
                num_warps=NUM_WARPS,
            )
    return outputs
