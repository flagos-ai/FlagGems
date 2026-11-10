# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GCU400 专属 ``fractional_max_pool3d_backward``。

算子语义：把 ``grad_output`` 按 ``indices`` scatter 回输入位置，同一输入位置
可能被多个输出命中，因此参考实现是累加。本文件提供三条同语义的执行路径：

* 2D tile + 标量原子累加（原始实现形态，作为安全 fallback）；
* 1D flat + 标量原子累加（把每 CTA 负责的元素数放大，压调度开销）；
* 零原子路径：先用探测 kernel 建 inverse map 判定「每个目标地址是否只有一个
  写者」，无重复时用普通 ``tl.store`` 直接写；探测到重复或越界则回退到原子
  路径，保证任意合法 indices 都与参考实现一致。

grad_input 的累加 dtype、tile 形状、launch 参数均由下方 TUNING PARAMS 驱动，
便于按 shape / dtype 分档，同时避免 autotune 在 GCU 上的多组冷编译。
公共实现文件未修改。
"""

import logging
from contextlib import nullcontext

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# KERNEL_MODE: 0=tile/atomic, 1=tile/atomic, 2=flat/atomic, 3=flat/atomic,
#              4=flat/store, 5=tile/store, 7=flat-fixed-grid/store
KERNEL_MODE = 7
USE_FP32_ACC = 0
BLOCK_H = 8
BLOCK_W = 16
FLAT_BLOCK = 1024
NUM_WARPS = 4
NUM_STAGES = 1
FIXED_GRID = 48
SMALL_OUT_ELEMENTS = 262144
CONTIGUOUS_INPUTS = 1
USE_DEVICE_CONTEXT = 0
STORE_MIN_OUT_ELEMENTS = 65536
CHECK_BLOCK = 2048
CHECK_GRID_SIZE = 24
USE_SHAPE_TIER = 1
TIER_THRESHOLD = 65536
TIER_SMALL_MODE = 1
# --- TUNING PARAMS END ---

STORE_MODES = (4, 5, 7)


@libentry()
@triton.jit
def _fmp3d_bwd_tile_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_spatial,
    out_d,
    out_h,
    out_w,
    out_spatial,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    IS_FP32_ACC: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    """按 (nc 平面, d*h块*w块) 二维 grid 把梯度 scatter 回输入位置。"""
    pid_plane = tl.program_id(0)
    pid_dhw = tl.program_id(1)

    num_h_blocks = tl.cdiv(out_h, BLOCK_H)
    num_w_blocks = tl.cdiv(out_w, BLOCK_W)
    num_hw_blocks = num_h_blocks * num_w_blocks
    d_idx = pid_dhw // num_hw_blocks
    hw_remainder = pid_dhw % num_hw_blocks
    h_block_idx = hw_remainder // num_w_blocks
    w_block_idx = hw_remainder % num_w_blocks

    h_out_offsets = h_block_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    w_out_offsets = w_block_idx * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (h_out_offsets < out_h)[:, None] & (w_out_offsets < out_w)[None, :]
    out_offsets = (
        pid_plane * out_spatial
        + d_idx * out_h * out_w
        + h_out_offsets[:, None] * out_w
        + w_out_offsets[None, :]
    )

    grad_vals = tl.load(grad_output_ptr + out_offsets, mask=out_mask, other=0.0)
    idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)
    valid_mask = out_mask & (idx_vals >= 0) & (idx_vals < in_spatial)
    target_offsets = pid_plane * in_spatial + idx_vals

    if IS_FP32_ACC:
        acc = grad_vals.to(tl.float32)
    else:
        acc = grad_vals
    acc = acc.to(grad_input_ptr.dtype.element_ty)
    if USE_ATOMIC:
        tl.atomic_add(grad_input_ptr + target_offsets, acc, mask=valid_mask)
    else:
        tl.store(grad_input_ptr + target_offsets, acc, mask=valid_mask)


@libentry()
@triton.jit
def _fmp3d_bwd_flat_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_spatial,
    out_spatial,
    BLOCK: tl.constexpr,
    IS_FP32_ACC: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    """按 (nc 平面, 输出空间块) 二维 grid 展平处理，避免 2D tile 的整数分解。"""
    pid_plane = tl.program_id(0)
    pid_block = tl.program_id(1)

    spatial_offsets = pid_block * BLOCK + tl.arange(0, BLOCK)
    out_mask = spatial_offsets < out_spatial
    out_offsets = pid_plane * out_spatial + spatial_offsets

    grad_vals = tl.load(grad_output_ptr + out_offsets, mask=out_mask, other=0.0)
    idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)
    valid_mask = out_mask & (idx_vals >= 0) & (idx_vals < in_spatial)
    target_offsets = pid_plane * in_spatial + idx_vals

    if IS_FP32_ACC:
        acc = grad_vals.to(tl.float32)
    else:
        acc = grad_vals
    acc = acc.to(grad_input_ptr.dtype.element_ty)
    if USE_ATOMIC:
        tl.atomic_add(grad_input_ptr + target_offsets, acc, mask=valid_mask)
    else:
        tl.store(grad_input_ptr + target_offsets, acc, mask=valid_mask)


@libentry()
@triton.jit
def _fmp3d_bwd_flat_fixed_kernel(
    grad_output_ptr,
    indices_ptr,
    grad_input_ptr,
    in_spatial,
    out_spatial,
    total_tiles,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    IS_FP32_ACC: tl.constexpr,
    USE_ATOMIC: tl.constexpr,
):
    """固定 grid + 核内跨步：program 数与问题规模解耦，匹配 SIP 拓扑。"""
    pid = tl.program_id(0)
    blocks_per_plane = tl.cdiv(out_spatial, BLOCK)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        plane = tile_id // blocks_per_plane
        block_in_plane = tile_id % blocks_per_plane
        spatial_offsets = block_in_plane * BLOCK + tl.arange(0, BLOCK)
        out_mask = spatial_offsets < out_spatial
        out_offsets = plane * out_spatial + spatial_offsets

        grad_vals = tl.load(grad_output_ptr + out_offsets, mask=out_mask, other=0.0)
        idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)
        valid_mask = out_mask & (idx_vals >= 0) & (idx_vals < in_spatial)
        target_offsets = plane * in_spatial + idx_vals

        if IS_FP32_ACC:
            acc = grad_vals.to(tl.float32)
        else:
            acc = grad_vals
        acc = acc.to(grad_input_ptr.dtype.element_ty)
        if USE_ATOMIC:
            tl.atomic_add(grad_input_ptr + target_offsets, acc, mask=valid_mask)
        else:
            tl.store(grad_input_ptr + target_offsets, acc, mask=valid_mask)


@libentry()
@triton.jit
def _fmp3d_bwd_probe_kernel(
    indices_ptr,
    inv_ptr,
    flags_ptr,
    in_spatial,
    out_spatial,
    total_out,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """建 inverse map：inv[目标地址] = 最后写入该地址的输出序号。

    同时用 flags[0] 记录越界索引；越界时统一回退到原子路径，由参考语义处理。
    """
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(total_out, BLOCK)
    for block_id in tl.range(pid, num_blocks, GRID_SIZE):
        out_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        out_mask = out_offsets < total_out
        idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)
        plane = out_offsets // out_spatial
        target_offsets = plane * in_spatial + idx_vals
        valid_mask = out_mask & (idx_vals >= 0) & (idx_vals < in_spatial)
        tl.store(inv_ptr + target_offsets, out_offsets.to(tl.int32), mask=valid_mask)
        has_invalid = tl.sum((out_mask & ~valid_mask).to(tl.int32), axis=0) > 0
        tl.store(flags_ptr, 1, mask=has_invalid)


@libentry()
@triton.jit
def _fmp3d_bwd_dup_kernel(
    indices_ptr,
    inv_ptr,
    flags_ptr,
    in_spatial,
    out_spatial,
    total_out,
    BLOCK: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """检查重复：若某输出的 owner 不是自己，说明同一目标地址有多个写者。"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(total_out, BLOCK)
    for block_id in tl.range(pid, num_blocks, GRID_SIZE):
        out_offsets = block_id * BLOCK + tl.arange(0, BLOCK)
        out_mask = out_offsets < total_out
        idx_vals = tl.load(indices_ptr + out_offsets, mask=out_mask, other=-1)
        plane = out_offsets // out_spatial
        target_offsets = plane * in_spatial + idx_vals
        valid_mask = out_mask & (idx_vals >= 0) & (idx_vals < in_spatial)
        owner = tl.load(
            inv_ptr + target_offsets,
            mask=valid_mask,
            other=out_offsets.to(tl.int32),
        )
        has_dup = (
            tl.sum(
                (valid_mask & (owner != out_offsets.to(tl.int32))).to(tl.int32),
                axis=0,
            )
            > 0
        )
        tl.store(flags_ptr + 1, 1, mask=has_dup)


def _can_use_store_path(
    indices, in_spatial, out_spatial, total_out, device, kernel_mode
):
    """判定是否可以走零原子 store；重复/越界时返回 False 由原子路径兜底。"""
    if kernel_mode not in STORE_MODES:
        return False
    if total_out < STORE_MIN_OUT_ELEMENTS:
        return False

    flags = torch.zeros(2, dtype=torch.int32, device=device)
    inv_map = torch.empty(
        (in_spatial * (total_out // out_spatial),), dtype=torch.int32, device=device
    )
    num_blocks = triton.cdiv(total_out, CHECK_BLOCK)
    grid = (min(num_blocks, CHECK_GRID_SIZE),)
    _fmp3d_bwd_probe_kernel[grid](
        indices,
        inv_map,
        flags,
        in_spatial,
        out_spatial,
        total_out,
        BLOCK=CHECK_BLOCK,
        GRID_SIZE=grid[0],
        num_warps=NUM_WARPS,
    )
    _fmp3d_bwd_dup_kernel[grid](
        indices,
        inv_map,
        flags,
        in_spatial,
        out_spatial,
        total_out,
        BLOCK=CHECK_BLOCK,
        GRID_SIZE=grid[0],
        num_warps=NUM_WARPS,
    )
    invalid_index, duplicate_index = flags.tolist()
    return invalid_index == 0 and duplicate_index == 0


def fractional_max_pool3d_backward(
    grad_output: torch.Tensor,
    input: torch.Tensor,
    kernel_size,
    output_size,
    indices: torch.Tensor,
):
    logger.debug("GEMS_GCU400 FRACTIONAL_MAX_POOL3D_BACKWARD")
    if isinstance(output_size, int):
        out_d = out_h = out_w = output_size
    else:
        out_d, out_h, out_w = output_size

    in_n, in_c, in_d, in_h, in_w = input.shape
    in_spatial = in_d * in_h * in_w
    out_spatial = out_d * out_h * out_w
    num_planes = in_n * in_c
    total_out = num_planes * out_spatial

    if CONTIGUOUS_INPUTS:
        grad_output = grad_output.contiguous()
        indices = indices.contiguous()

    # 按输出规模分档：大 shape 用零原子 store 路径，小 shape 用大 tile 原子路径，
    # 避免探测同步与固定 grid 在小 shape 上摊薄不掉固定开销。
    effective_mode = KERNEL_MODE
    if USE_SHAPE_TIER and total_out < TIER_THRESHOLD:
        effective_mode = TIER_SMALL_MODE

    use_store_path = _can_use_store_path(
        indices,
        in_spatial,
        out_spatial,
        total_out,
        grad_output.device,
        effective_mode,
    )
    use_fp32_acc = USE_FP32_ACC == 1 and not use_store_path
    acc_dtype = torch.float32 if use_fp32_acc else grad_output.dtype
    grad_input = torch.zeros_like(input, dtype=acc_dtype)
    if grad_input.numel() == 0:
        return grad_input.to(grad_output.dtype)

    # 零原子路径下每个目标地址只有一个写者；否则沿用标量原子累加语义
    use_atomic = not use_store_path
    is_fp32_acc = acc_dtype == torch.float32

    block_h, block_w = BLOCK_H, BLOCK_W

    device_context = (
        torch_device_fn.device(grad_output.device)
        if USE_DEVICE_CONTEXT
        else nullcontext()
    )
    with device_context:
        if effective_mode in (0, 1, 5):
            grid = (
                num_planes,
                out_d * triton.cdiv(out_h, block_h) * triton.cdiv(out_w, block_w),
            )
            _fmp3d_bwd_tile_kernel[grid](
                grad_output,
                indices,
                grad_input,
                in_spatial,
                out_d,
                out_h,
                out_w,
                out_spatial,
                block_h,
                block_w,
                is_fp32_acc,
                use_atomic,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
        elif effective_mode in (2, 3, 4, 6):
            grid = (num_planes, triton.cdiv(out_spatial, FLAT_BLOCK))
            _fmp3d_bwd_flat_kernel[grid](
                grad_output,
                indices,
                grad_input,
                in_spatial,
                out_spatial,
                FLAT_BLOCK,
                is_fp32_acc,
                use_atomic,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )
        else:
            blocks_per_plane = triton.cdiv(out_spatial, FLAT_BLOCK)
            total_tiles = num_planes * blocks_per_plane
            grid = (min(total_tiles, FIXED_GRID),)
            _fmp3d_bwd_flat_fixed_kernel[grid](
                grad_output,
                indices,
                grad_input,
                in_spatial,
                out_spatial,
                total_tiles,
                FLAT_BLOCK,
                grid[0],
                is_fp32_acc,
                use_atomic,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )

    if acc_dtype != grad_output.dtype:
        return grad_input.to(grad_output.dtype)
    return grad_input
