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

"""Triton implementation of ``aten::_indices_copy`` for sparse COO tensors.

GCU400 的 sparse COO 构造当前回退 SparseCPU，``self._indices()`` 位于 CPU，
Triton kernel 只能访问 GCU 指针。这里先把索引缓冲搬上设备、由 Triton kernel
完成核心拷贝，再按输出张量的设备写回；设备搬运属于不可避免的前后处理。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)

# 优化循环按轮次替换的可调常量；保持单份 kernel 定义，避免重复特化带来的冷编译。
FLAT_BLOCK_SIZE = 2048
FLAT_NUM_WARPS = 4
FLAT_FIXED_GRID = 0
TWO_D_BLOCK_SIZE = 4096
TWO_D_NUM_WARPS = 4
REBIND_OUT_TO_SOURCE_DEVICE = True


@triton.jit
def _indices_copy_flat(
    src_ptr,
    dst_ptr,
    numel,
    BLOCK: tl.constexpr,
    FIXED_GRID: tl.constexpr,
):
    """连续索引缓冲的一维拷贝；FIXED_GRID>0 时用固定 grid 加跨步循环。"""
    lane_offsets = tl.arange(0, BLOCK)
    num_tiles = tl.cdiv(numel, BLOCK)
    if FIXED_GRID > 0:
        for tile_id in tl.range(tl.program_id(0), num_tiles, FIXED_GRID):
            offsets = tile_id * BLOCK + lane_offsets
            mask = offsets < numel
            values = tl.load(src_ptr + offsets, mask=mask)
            tl.store(dst_ptr + offsets, values, mask=mask)
    else:
        offsets = tl.program_id(0) * BLOCK + lane_offsets
        mask = offsets < numel
        values = tl.load(src_ptr + offsets, mask=mask)
        tl.store(dst_ptr + offsets, values, mask=mask)


@triton.jit
def _indices_copy_2d(src_ptr, dst_ptr, nnz, s0, s1, BLOCK: tl.constexpr):
    """非连续索引视图的二维 strided gather，按行写回紧凑输出。"""
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < nnz
    values = tl.load(src_ptr + row * s0 + cols * s1, mask=mask)
    tl.store(dst_ptr + row * nnz + cols, values, mask=mask)


def _current_gcu_device():
    """返回当前 GCU 设备；缺少 current_device 时退回默认设备号 0。"""
    device_index = (
        torch.gcu.current_device() if hasattr(torch.gcu, "current_device") else 0
    )
    return torch.device("gcu", device_index)


def _indices_copy_impl(self, out=None, *, caller_out=False):
    """把 sparse COO 的索引缓冲拷贝到新 buffer 或调用方给定的 ``out``。

    源索引在 CPU 时先做一次 H2D，Triton kernel 在设备上完成拷贝；若调用方
    ``out`` 与源索引设备不一致，则在写回后把同一对象重绑到源设备 storage，
    保证 ``out`` 变体返回对象与比较口径一致。
    """
    plain_indices = self._indices()
    sparse_dim, nnz = plain_indices.shape
    if out is None:
        out = torch.empty(
            (sparse_dim, nnz), dtype=torch.int64, device=plain_indices.device
        )
    if nnz == 0:
        return out

    gcu_device = _current_gcu_device()
    source = (
        plain_indices
        if plain_indices.device.type == "gcu"
        else plain_indices.to(gcu_device)
    )
    if out.device.type == "gcu":
        destination = out
    else:
        destination = torch.empty(
            (sparse_dim, nnz), dtype=torch.int64, device=gcu_device
        )

    with torch_device_fn.device(gcu_device):
        if source.is_contiguous():
            numel = sparse_dim * nnz
            block_size = FLAT_BLOCK_SIZE
            if FLAT_FIXED_GRID > 0:
                grid = (min(FLAT_FIXED_GRID, triton.cdiv(numel, block_size)),)
            else:
                grid = (triton.cdiv(numel, block_size),)
            _indices_copy_flat[grid](
                source,
                destination,
                numel,
                BLOCK=block_size,
                FIXED_GRID=FLAT_FIXED_GRID,
                num_warps=FLAT_NUM_WARPS,
            )
        else:
            stride_row, stride_col = source.stride()
            block_size = TWO_D_BLOCK_SIZE
            grid = (sparse_dim, triton.cdiv(nnz, block_size))
            _indices_copy_2d[grid](
                source,
                destination,
                nnz,
                stride_row,
                stride_col,
                BLOCK=block_size,
                num_warps=TWO_D_NUM_WARPS,
            )

    if destination is not out:
        out.copy_(destination.to(out.device))
    elif caller_out and out.device != plain_indices.device:
        # CPU sparse 索引 + GCU out 的环境差异：保留同一 out 对象，重绑其 storage。
        # set_ 禁止跨设备（实测 RuntimeError），data 赋值可原地切换设备且保持对象同一性。
        if REBIND_OUT_TO_SOURCE_DEVICE:
            out.data = destination.to(plain_indices.device)
    return out


def _indices_copy(self: torch.Tensor) -> torch.Tensor:
    """``aten::_indices_copy``：返回索引缓冲的独立拷贝。"""
    logger.debug("GEMS _INDICES_COPY")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_indices_copy only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    return _indices_copy_impl(self)


def _indices_copy_out(self: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    """``aten::_indices_copy.out``：把索引缓冲写入调用方提供的 ``out``。"""
    logger.debug("GEMS _INDICES_COPY_OUT")
    if not isinstance(self, torch.Tensor) or self.layout != torch.sparse_coo:
        raise RuntimeError(
            "_indices_copy only supports sparse coordinate (coo) tensors, "
            f"but got layout {getattr(self, 'layout', None)}"
        )
    if out.dtype != torch.int64:
        raise RuntimeError(
            "Expected out tensor to have dtype torch.int64, but got "
            f"{out.dtype} instead"
        )
    sparse_dim, nnz = self._indices().shape
    if tuple(out.shape) != (sparse_dim, nnz):
        out.resize_((sparse_dim, nnz))
    return _indices_copy_impl(self, out=out, caller_out=True)
