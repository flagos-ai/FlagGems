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
import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.config_utils import MAX_GRID_DIM

logger = logging.getLogger(__name__)

MAX_BLOCK_SIZE = 32768
_FORWARD_WARPS = 4  # 前向 launch 的 warp 数，按 shape 扫描后固定

# ---- Row-batch (2D [ROWS_PER_CTA, BLOCK] tile) budget --------------------
# 移植自本 backend 已验证的 layernorm 族行批 kernel（layernorm.py /
# fused/gcu400/skip_layernorm.py，compiler_issues/
# vllm_fused_q_kv_rmsnorm_rowbatch_bcast_miscompile.md 的 225 config 验证集）：
# - tile 元素预算：fp32 活集（x tile + 平方积 + [1,BLOCK] 权重）保持在每个
#   program 的 DSM 预算内；
# - R >= 32 且 32768 元素 tile 是 triton_gcu 静默错值区；tile <= 16384 且
#   R <= 32 已跨 225 (shape, dtype, R, num_warps) 组合验证正确。
# 扩展区（16384 < tile <= dtype 预算）：按编译产物 gcu.dsm_memory_size 实测
# 标定——fp16 R=16/B=4096（tile 65536）dsm=282KB、fp32 R=8/B=4096（tile
# 32768）dsm=299KB，均 < 448KB 双 program 线保 48-way；fp32 tile 65536 实测
# dsm=561KB 超线锁 24-way 反慢。扩展区 R 一律封 16（远离 R>=32 错值触发轴）。
# 大 tile 档实测收益（(4096,4096) do_bench median）：fp16 R=4 68.6us ->
# R=8 49.5us -> R=16 40.9us（1.64TB/s）；fp32 R=4 88.3 -> R=8 71.6us。
_ROW_BATCH_MAX_BLOCK = 16384
_ROW_TILE_ELEM_MAX = 16384
_ROW_TILE_ELEM_MAX_F16 = 65536
_ROW_TILE_ELEM_MAX_F32 = 32768
_ROWS_PER_CTA_CAP = 32
_ROWS_PER_CTA_CAP_EXT = 16  # 扩展区（tile > 16384）的 R 上限
_ROWS_PER_CTA_MIN = 4
_TARGET_CTAS = 48
# 行批 tile 低于该 lane 数时每 program 固定成本占比反超并行度收益。
_ROW_TILE_MIN_LANES = 4096
# 行数铺满设备（R=4 时 >= 48 tile）才走行批；更小输入保留逐行 legacy kernel。
_ROW_BATCH_MIN_ROWS = _TARGET_CTAS * _ROWS_PER_CTA_MIN
# 1-D 尾臂编译进 kernel 时必须 num_warps=1（nw>=2 使尾臂劣化整个 2-D 热路径
# 1.3-1.9x，2026-11-08 vllm skip_layer_norm 实测）；尾臂整体编译掉
# （HAS_TAIL=False）且无 [R] 副输出 store 时，[1,BLOCK] 2-D 形状权重 +
# 软流水循环 + num_warps=2 在 fused_add_rms_norm 行批 kernel 已验证正确。
# 带 [R] 副输出 store（SAVE_INV_RMS）时保守锁 nw=1（layernorm.py 同款）。
_TAIL_NUM_WARPS = 1
_ROWS_NUM_WARPS = 2


def _row_tile_elem_max(dtype):
    if dtype == torch.float16 or dtype == torch.bfloat16:
        return _ROW_TILE_ELEM_MAX_F16
    if dtype == torch.float32:
        return _ROW_TILE_ELEM_MAX_F32
    return _ROW_TILE_ELEM_MAX


def _rows_per_cta(m, n, dtype):
    """为 [R, BLOCK] 2-D tile launch 选行批因子 R。

    行宽已占满 tile 预算或输入太小铺不满设备时返回 1（legacy 逐行 kernel）；
    否则取保持约 _TARGET_CTAS 个 tile 在飞的最大 2 的幂，受按 dtype 标定的
    DSM tile 预算与已验证安全的 R 上限约束（tile <= 16384 时 R <= 32 的
    225-config 验证区；扩展区 R <= 16），并给 tile 一个最小 lane 下限以摊薄
    每 program 固定成本。
    """
    block = triton.next_power_of_2(n)
    if block > _ROW_BATCH_MAX_BLOCK:
        return 1
    tile_max = _row_tile_elem_max(dtype)
    tile_cap = tile_max // max(block, 1)
    if tile_cap < _ROWS_PER_CTA_MIN or m < _ROW_BATCH_MIN_ROWS:
        return 1
    rows = m // _TARGET_CTAS
    rows = 1 << max(0, rows.bit_length() - 1) if rows >= 1 else 1
    rows = max(rows, _ROW_TILE_MIN_LANES // max(block, 1))
    rows = max(rows, _ROWS_PER_CTA_MIN)
    rows = max(1, min(_ROWS_PER_CTA_CAP, tile_cap, rows))
    if rows * block > _ROW_TILE_ELEM_MAX:
        # 扩展区：大 tile 档把 R 压回 16 以内（R>=32 + >=32768 tile 是
        # 已知静默错值触发轴）。
        rows = min(rows, _ROWS_PER_CTA_CAP_EXT)
    return rows


@triton.jit
def _rms_norm_rows(
    Y,
    INV_RMS,
    X,
    W,
    pid,
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,
    x_stride_c: tl.constexpr,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """一个 program 处理 ROWS 行：单块路径下 w 只加载一次供所有行复用。

    每行仍用标量 tl.sum(axis=0) 归约（与 1D kernel 完全等价）。本 legacy
    形态保留给小 M / 宽 N / 行批关闭的档位；大 M 档位走 rms_norm_rows_kernel
    的行批 2-D tile（其 axis=1 归约须配 [1,BLOCK] 2-D 形状权重 + nw 纪律，
    早期无免疫配方的 1-D w 广播 + nw>=2 组合才是错值触发窗）。
    """
    cols = tl.arange(0, BLOCK_SIZE)
    cmask = cols < N
    if SINGLE_BLOCK:
        w_shared = tl.load(W + cols, mask=cmask, other=0.0)
    else:
        w_shared = tl.full((BLOCK_SIZE,), 0.0, tl.float32)

    for r in range(ROWS):
        row = pid * ROWS + r
        if row < M:
            Y_cur = row * y_stride_r + Y
            X_cur = row * x_stride_r + X
            if SINGLE_BLOCK:
                x = tl.load(X_cur + cols * x_stride_c, mask=cmask, other=0.0).to(
                    tl.float32
                )
                var = tl.sum(x * x, axis=0) / N
                rrms = tl.math.rsqrt(var + eps)
                y = (x * rrms).to(Y_cur.dtype.element_ty) * w_shared
                tl.store(Y_cur + cols * y_stride_c, y, mask=cmask)
                if SAVE_INV_RMS:
                    tl.store(INV_RMS + row, rrms)
            else:
                # 行宽超过单 tile 上限：分块累加方差，再分块写结果
                _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
                for off in range(0, N, BLOCK_SIZE):
                    c = off + tl.arange(0, BLOCK_SIZE)
                    m = c < N
                    t = tl.load(X_cur + c * x_stride_c, mask=m, other=0.0).to(
                        tl.float32
                    )
                    _var += t * t
                var = tl.sum(_var, axis=0) / N
                rrms = tl.math.rsqrt(var + eps)
                for off in range(0, N, BLOCK_SIZE):
                    c = off + tl.arange(0, BLOCK_SIZE)
                    m = c < N
                    t = tl.load(X_cur + c * x_stride_c, mask=m, other=0.0).to(
                        tl.float32
                    )
                    w = tl.load(W + c, mask=m, other=0.0)
                    y = (t * rrms).to(Y_cur.dtype.element_ty) * w
                    tl.store(Y_cur + c * y_stride_c, y, mask=m)
                if SAVE_INV_RMS:
                    tl.store(INV_RMS + row, rrms)


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_kernel(
    Y,
    INV_RMS,
    X,
    W,
    y_stride_r: tl.constexpr,
    y_stride_c: tl.constexpr,
    x_stride_r: tl.constexpr,
    x_stride_c: tl.constexpr,
    M,
    N,
    eps,
    total_programs,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    for pid in range(tl.program_id(0), total_programs, tl.num_programs(0)):
        _rms_norm_rows(
            Y,
            INV_RMS,
            X,
            W,
            pid,
            y_stride_r,
            y_stride_c,
            x_stride_r,
            x_stride_c,
            M,
            N,
            eps,
            BLOCK_SIZE,
            ROWS,
            SINGLE_BLOCK,
            SAVE_INV_RMS,
        )


@triton.jit
def _rms_norm_row(
    Y,
    INV_RMS,
    X,
    W,
    row,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    EXACT: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """1-D 单行体（尾块回退，任意 BLOCK 宽度下均正确）。"""
    Y += row * N
    X += row * N
    cols = tl.arange(0, BLOCK_SIZE)
    if EXACT:
        x = tl.load(X + cols).to(tl.float32)
        w = tl.load(W + cols)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + row, rrms)
    else:
        mask = cols < N
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0)
        var = tl.sum(x * x, axis=0) / N
        rrms = tl.math.rsqrt(var + eps)
        y = (x * rrms).to(Y.dtype.element_ty) * w
        tl.store(Y + cols, y, mask=mask)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + row, rrms)


@triton.jit
def _rms_norm_rows_tile(
    Y,
    INV_RMS,
    X,
    w,
    row0,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,
    SAVE_INV_RMS: tl.constexpr,
):
    """整 [ROWS, BLOCK_SIZE] tile 体；w 为 [1, BLOCK_SIZE] 2-D 形状。"""
    rows = row0 + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = X + rows[:, None] * N + cols[None, :]
    y_ptrs = Y + rows[:, None] * N + cols[None, :]
    if EXACT:
        x = tl.load(x_ptrs).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rrms = tl.math.rsqrt(var + eps)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + rows, rrms)
        y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w
        tl.store(y_ptrs, y)
    else:
        col_mask = cols < N
        x = tl.load(x_ptrs, mask=col_mask[None, :], other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / N
        rrms = tl.math.rsqrt(var + eps)
        if SAVE_INV_RMS:
            tl.store(INV_RMS + rows, rrms)
        y = (x * rrms[:, None]).to(Y.dtype.element_ty) * w
        tl.store(y_ptrs, y, mask=col_mask[None, :])


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_rows_kernel(
    Y,
    INV_RMS,
    X,
    W,
    M,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    EXACT: tl.constexpr,  # N == BLOCK_SIZE: 无列掩码
    HAS_TAIL: tl.constexpr,  # M % ROWS != 0: 1-D 尾臂编译进 kernel
    SAVE_INV_RMS: tl.constexpr,
):
    """行批 kernel：每 program 经 [ROWS, BLOCK_SIZE] 2-D tile 归一化多行，
    摊薄 one-row-per-program 形态在大 M 下主导的每 program 固定成本
    （load 派发 + 归约树 + 权重装载）。grid <= 48 的 persistent tile 循环
    消除波间 CTA retire/重发射间隙，且 launch grid 保持 1-D，不会触达
    GCU 65535 grid.x 上限。残行尾块走 1-D 行体（2-D masked 尾块变体是
    本 backend 的静默错值区）。权重按 [1, BLOCK_SIZE] 2-D 形状在循环外
    装载一次：1-D 向量广播操作数在 BLOCK 512/1024 + num_warps>=2 是已
    验证错值窗，2-D 形状形态在 fused_add_rms_norm 行批 kernel 全触发窗
    验证正确。
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(M, ROWS)
    cols = tl.arange(0, BLOCK_SIZE)
    # 权重是循环不变量：每 program 装载一次，[1, BLOCK_SIZE] 2-D 形状。
    if EXACT:
        w = tl.load(W + cols[None, :])
    else:
        col_mask0 = cols < N
        w = tl.load(W + cols[None, :], mask=col_mask0[None, :], other=0.0)
    if HAS_TAIL:
        # 1-D 尾臂编译进 kernel：必须 num_warps=1 launch。
        for tile in range(pid, num_tiles, num_programs):
            row0 = tile * ROWS
            if row0 + ROWS <= M:
                _rms_norm_rows_tile(
                    Y,
                    INV_RMS,
                    X,
                    w,
                    row0,
                    N,
                    eps,
                    BLOCK_SIZE,
                    ROWS,
                    EXACT,
                    SAVE_INV_RMS,
                )
            else:
                # 尾块（残行）：逐行 1-D 形态处理。
                for i in tl.static_range(ROWS):
                    row = row0 + i
                    if row < M:
                        _rms_norm_row(
                            Y,
                            INV_RMS,
                            X,
                            W,
                            row,
                            N,
                            eps,
                            BLOCK_SIZE,
                            EXACT,
                            SAVE_INV_RMS,
                        )
    else:
        # host 保证 M % ROWS == 0：每个 tile 全满、循环体无分支纯 2-D；
        # 软流水该循环。
        for tile in tl.range(pid, num_tiles, num_programs, num_stages=2):
            row0 = tile * ROWS
            _rms_norm_rows_tile(
                Y,
                INV_RMS,
                X,
                w,
                row0,
                N,
                eps,
                BLOCK_SIZE,
                ROWS,
                EXACT,
                SAVE_INV_RMS,
            )


@libentry()
@triton.jit(do_not_specialize=["eps"])
def rms_norm_grad_dx_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DX,  # pointer to the output
    W,  # pointer to the weights
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    for i in range(BLOCK_SIZE_M):
        pid = tl.program_id(0) * BLOCK_SIZE_M + i
        DX_cur = pid * dx_stride_r + DX
        X_cur = pid * x_stride_r + X
        DY_cur = pid * x_stride_r + DY
        INV_RMS_cur = pid + INV_RMS

        inv_rms = tl.load(INV_RMS_cur).to(tl.float32)

        _row_sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            dy_w = dy * w
            normalized_buf = x * inv_rms
            _row_sum += normalized_buf * dy_w
        row_sum_stats = tl.sum(_row_sum, axis=0)

        for off in range(0, N, BLOCK_SIZE):
            cols = off + tl.arange(0, BLOCK_SIZE)
            mask = cols < N
            x = tl.load(X_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            dy = tl.load(DY_cur + cols * x_stride_c, mask, other=0.0).to(tl.float32)
            w = tl.load(W + cols, mask=mask, other=0.0)
            dy_w = dy * w
            normalized_buf = x * inv_rms
            norm_val = normalized_buf / N
            dx = (dy_w - norm_val * row_sum_stats) * inv_rms
            tl.store(DX_cur + cols * dx_stride_c, dx, mask=mask)


@libentry()
@triton.jit
def rms_norm_grad_dw_kernel(
    X,  # pointer to the input
    DY,
    INV_RMS,  # pointer to inverse rms
    DW,  # pointer to the output
    dx_stride_r,
    dx_stride_c,
    x_stride_r,  # how much to increase the pointer when moving by 1 row
    x_stride_c,  # how much to increase the pointer when moving by 1 col
    M,  # number of rows in X
    N,  # number of columns in X
    ROW_BLOCK_SIZE: tl.constexpr,
    COL_BLOCK_SIZE: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    row_start = row_pid * ROW_BLOCK_SIZE
    col_start = col_pid * COL_BLOCK_SIZE

    offset = row_start * x_stride_r + col_start * x_stride_c
    X += offset
    DY += offset
    INV_RMS += row_start

    rows = tl.arange(0, ROW_BLOCK_SIZE)
    cols = tl.arange(0, COL_BLOCK_SIZE)

    row_mask = (row_start + rows) < M
    col_mask = (col_start + cols) < N

    x = tl.load(
        X + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    inv_rms = tl.load(INV_RMS + rows, row_mask, other=0.0).to(tl.float32)
    dy = tl.load(
        DY + rows[:, None] * x_stride_r + cols[None, :] * x_stride_c,
        row_mask[:, None] & col_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    d_weight = x * dy * inv_rms[:, None]
    partial_dweight_sum = tl.sum(d_weight, axis=0)

    tl.store(
        DW + row_pid * N + col_start + cols,
        partial_dweight_sum,
        mask=col_mask,
    )


class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, normalized_shape, weight, eps=1e-5):
        logger.debug("GEMS_ENFLAME RMSNORM_FORWARD")
        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        single_block = triton.next_power_of_2(N) <= MAX_BLOCK_SIZE
        # 只有需要反向时才保存 inv_rms；benchmark / 推理路径不分配也不写
        save_inv_rms = x.requires_grad or weight.requires_grad
        x = x.contiguous()
        weight = weight.contiguous()
        y = torch.empty_like(x)
        inv_rms = (
            torch.empty((M,), device=x.device, dtype=torch.float32)
            if save_inv_rms
            else x
        )

        with torch_device_fn.device(x.device):
            # 行批 2-D tile 快路径：摊薄逐行 kernel 的每行固定成本。
            rows_per_cta = _rows_per_cta(M, N, x.dtype)
            if rows_per_cta > 1:
                num_tiles = triton.cdiv(M, rows_per_cta)
                has_tail = (M % rows_per_cta) != 0
                grid_m = min(num_tiles, _TARGET_CTAS)
                if save_inv_rms:
                    # 带 [R] 副输出 store 时保守锁 nw=1 且尾臂按运行时
                    # 分支保留（layernorm.py 同款已验证形态）。
                    kernel_has_tail = True
                    num_warps = _TAIL_NUM_WARPS
                else:
                    kernel_has_tail = has_tail
                    # 尾臂编译进 kernel 时必须 nw=1；纯 2-D 无尾臂可用
                    # nw=2 + 软流水（fused_add_rms_norm 验证形态）。
                    num_warps = _TAIL_NUM_WARPS if has_tail else _ROWS_NUM_WARPS
                rms_norm_rows_kernel[grid_m,](
                    y,
                    inv_rms,
                    x,
                    weight,
                    M,
                    N,
                    eps,
                    BLOCK_SIZE=BLOCK_SIZE,
                    ROWS=rows_per_cta,
                    EXACT=(N == BLOCK_SIZE),
                    HAS_TAIL=kernel_has_tail,
                    SAVE_INV_RMS=save_inv_rms,
                    num_warps=num_warps,
                )
            else:
                # 每个 program 处理多行：w 只加载一次、program 数按 M 减少
                if single_block:
                    if M >= 512:
                        rows = 8
                    elif M >= 128:
                        rows = 4
                    else:
                        rows = 2
                else:
                    rows = 1
                total_programs = triton.cdiv(M, rows)
                grid_m = min(total_programs, MAX_GRID_DIM)

                rms_norm_kernel[grid_m,](
                    y,
                    inv_rms,
                    x,
                    weight,
                    N,
                    1,
                    N,
                    1,
                    M,
                    N,
                    eps,
                    total_programs,
                    BLOCK_SIZE,
                    rows,
                    single_block,
                    save_inv_rms,
                    num_warps=_FORWARD_WARPS,
                )

        ctx.save_for_backward(x, inv_rms, weight)
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, dy):
        logger.debug("GEMS_ENFLAME RMSNORM_BACKWARD")
        x, inv_rms, weight = ctx.saved_tensors
        normalized_shape = ctx.normalized_shape
        eps = ctx.eps

        dim = x.ndim - len(normalized_shape)
        M = math.prod(x.shape[:dim])
        N = math.prod(normalized_shape)

        BLOCK_SIZE = min(triton.next_power_of_2(N), MAX_BLOCK_SIZE)
        x = x.contiguous()
        weight = weight.contiguous()
        dx = torch.empty_like(x)

        grid_m = M
        BLOCK_SIZE_M = 1
        if M > 65535:
            grid_m = MAX_GRID_DIM
            BLOCK_SIZE_M = (M + MAX_GRID_DIM - 1) // MAX_GRID_DIM

        with torch_device_fn.device(x.device):
            rms_norm_grad_dx_kernel[grid_m,](
                x, dy, inv_rms, dx, weight, N, 1, N, 1, N, eps, BLOCK_SIZE, BLOCK_SIZE_M
            )

        ROW_BLOCK_SIZE = 16
        COL_BLOCK_SIZE = 256
        row_block_num = triton.cdiv(M, ROW_BLOCK_SIZE)
        col_block_num = triton.cdiv(N, COL_BLOCK_SIZE)

        partial_buffer = torch.empty(
            (row_block_num, N), dtype=torch.float32, device=x.device
        )

        with torch_device_fn.device(x.device):
            rms_norm_grad_dw_kernel[row_block_num, col_block_num](
                x,
                dy,
                inv_rms,
                partial_buffer,
                N,
                1,
                N,
                1,
                M,
                N,
                ROW_BLOCK_SIZE,
                COL_BLOCK_SIZE,
            )
            # TODO(haizhu.shao TBD) after support dim=0 in triton_gcu backend,
            # we can directly use flag_gems.sum(partial_buffer, dim=0, dtype=x.dtype)
            dw = torch.sum(partial_buffer, dim=0).to(x.dtype).reshape(-1)

        return dx, None, dw, None


def rms_norm(x, normalized_shape, weight, eps=1e-5):
    return RmsNorm.apply(x, normalized_shape, weight, eps)
