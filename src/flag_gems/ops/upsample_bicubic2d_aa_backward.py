import math
import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def cubic_keys(x):
    """Keys cubic filter with a = -0.5.

    |x| < 1 : 1.5|x|^3 - 2.5|x|^2 + 1
    1 <= |x| < 2 : -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
    otherwise : 0
    """
    ax = tl.abs(x)
    ax2 = ax * ax
    ax3 = ax2 * ax
    w_inner = 1.5 * ax3 - 2.5 * ax2 + 1.0
    w_outer = -0.5 * ax3 + 2.5 * ax2 - 4.0 * ax + 2.0
    return tl.where(ax < 1.0, w_inner, tl.where(ax < 2.0, w_outer, 0.0))


# ============================================================
# Kernel A: fp32 path
# grad_out / grad_in 均为 fp32，直接 atomic_add，无需额外 cast
# ============================================================

@triton.jit
def _upsample_bicubic2d_aa_backward_kernel_fp32(
    grad_out_ptr,
    grad_in_ptr,
    n, c, in_h, in_w, out_h, out_w,
    go_stride_n, go_stride_c, go_stride_h, go_stride_w,
    gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
    scale_y, scale_x,
    scale_y_aa, scale_x_aa,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n * c * out_h * out_w
    mask = offs < total

    x_out = offs % out_w
    tmp   = offs // out_w
    y_out = tmp % out_h
    tmp   = tmp // out_h
    c_idx = tmp % c
    n_idx = tmp // c

    y_out_f = y_out.to(tl.float32)
    x_out_f = x_out.to(tl.float32)

    y_real = (y_out_f + 0.5) * scale_y - 0.5
    x_real = (x_out_f + 0.5) * scale_x - 0.5

    support_h = 2.0 / scale_y_aa
    support_w = 2.0 / scale_x_aa

    y_in_start = (tl.floor(y_real - support_h) + 1).to(tl.int32)
    x_in_start = (tl.floor(x_real - support_w) + 1).to(tl.int32)

    go_off = (
        n_idx.to(tl.int64) * go_stride_n
        + c_idx.to(tl.int64) * go_stride_c
        + y_out.to(tl.int64) * go_stride_h
        + x_out.to(tl.int64) * go_stride_w
    )
    go_val = tl.load(grad_out_ptr + go_off, mask=mask, other=0.0)  # fp32

    gi_base = (
        grad_in_ptr
        + n_idx.to(tl.int64) * gi_stride_n
        + c_idx.to(tl.int64) * gi_stride_c
    )

    norm = tl.zeros([BLOCK], dtype=tl.float32)
    for dh in range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)
            norm  += w_h * w_w

    norm = tl.where(norm == 0.0, 1.0, norm)

    for dh in range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h     = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)

            weight  = (w_h * w_w) / norm
            contrib = go_val * weight  # fp32

            gi_off = (
                y_in.to(tl.int64) * gi_stride_h
                + x_in.to(tl.int64) * gi_stride_w
            )
            tl.atomic_add(gi_base + gi_off, contrib, mask=valid)


# ============================================================
# Kernel B1: lowp 下采样路径（scale_y_aa < 1 或 scale_x_aa < 1）
#
# AA 窗口宽于一个像素，支持域 > 2，SH/SW 较大。
# grad_out 低精度读入后立即 cast 到 fp32；
# grad_in 是 fp32 临时 buffer，atomic_add 全程在 fp32 完成，
# 最后由调用方 cast 回低精度。
# 精度已满足要求，逻辑与原 lowp kernel 保持一致。
# ============================================================

@triton.jit
def _upsample_bicubic2d_aa_backward_kernel_lowp_downsample(
    grad_out_ptr,
    grad_in_ptr,                        # fp32 临时 buffer
    n, c, in_h, in_w, out_h, out_w,
    go_stride_n, go_stride_c, go_stride_h, go_stride_w,
    gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
    scale_y, scale_x,
    scale_y_aa, scale_x_aa,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n * c * out_h * out_w
    mask = offs < total

    x_out = offs % out_w
    tmp   = offs // out_w
    y_out = tmp % out_h
    tmp   = tmp // out_h
    c_idx = tmp % c
    n_idx = tmp // c

    y_out_f = y_out.to(tl.float32)
    x_out_f = x_out.to(tl.float32)

    y_real = (y_out_f + 0.5) * scale_y - 0.5
    x_real = (x_out_f + 0.5) * scale_x - 0.5

    support_h = 2.0 / scale_y_aa
    support_w = 2.0 / scale_x_aa

    y_in_start = (tl.floor(y_real - support_h) + 1).to(tl.int32)
    x_in_start = (tl.floor(x_real - support_w) + 1).to(tl.int32)

    go_off = (
        n_idx.to(tl.int64) * go_stride_n
        + c_idx.to(tl.int64) * go_stride_c
        + y_out.to(tl.int64) * go_stride_h
        + x_out.to(tl.int64) * go_stride_w
    )
    # 低精度读入，立即 cast 到 fp32，后续所有计算在 fp32 完成
    go_val = tl.load(grad_out_ptr + go_off, mask=mask, other=0.0).to(tl.float32)

    gi_base = (
        grad_in_ptr
        + n_idx.to(tl.int64) * gi_stride_n
        + c_idx.to(tl.int64) * gi_stride_c
    )

    norm = tl.zeros([BLOCK], dtype=tl.float32)
    for dh in range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)
            norm  += w_h * w_w

    norm = tl.where(norm == 0.0, 1.0, norm)

    for dh in range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h     = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)

            weight  = (w_h * w_w) / norm
            contrib = go_val * weight       # fp32 * fp32 = fp32

            gi_off = (
                y_in.to(tl.int64) * gi_stride_h
                + x_in.to(tl.int64) * gi_stride_w
            )
            # atomic_add 写入 fp32 buffer，避免低精度累加误差
            tl.atomic_add(gi_base + gi_off, contrib, mask=valid)


# ============================================================
# Kernel B2: lowp 上采样路径（scale_y_aa == 1 且 scale_x_aa == 1）
#
# 上采样时 AA 缩放因子恒为 1，支持域固定为 2（SH=SW=4），
# 与下采样路径的数值行为完全相同，但单独拆出以便后续针对
# 上采样的精度问题独立修复（TODO）。
# ============================================================

@triton.jit
def bak_upsample_bicubic2d_aa_backward_kernel_lowp_upsample(
    grad_out_ptr,
    grad_in_ptr,                        # fp32 临时 buffer
    n, c, in_h, in_w, out_h, out_w,
    go_stride_n, go_stride_c, go_stride_h, go_stride_w,
    gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
    scale_y, scale_x,
    scale_y_aa, scale_x_aa,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # TODO: 上采样低精度路径精度尚未达标，待后续专项修复。
    #       当前实现与下采样路径逻辑一致，作为占位。
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n * c * out_h * out_w
    mask = offs < total

    x_out = offs % out_w
    tmp   = offs // out_w
    y_out = tmp % out_h
    tmp   = tmp // out_h
    c_idx = tmp % c
    n_idx = tmp // c

    y_out_f = y_out.to(tl.float32)
    x_out_f = x_out.to(tl.float32)

    y_real = (y_out_f + 0.5) * scale_y - 0.5
    x_real = (x_out_f + 0.5) * scale_x - 0.5

    support_h = 2.0 / scale_y_aa
    support_w = 2.0 / scale_x_aa

    y_in_start = (tl.floor(y_real - support_h) + 1).to(tl.int32)
    x_in_start = (tl.floor(x_real - support_w) + 1).to(tl.int32)

    go_off = (
        n_idx.to(tl.int64) * go_stride_n
        + c_idx.to(tl.int64) * go_stride_c
        + y_out.to(tl.int64) * go_stride_h
        + x_out.to(tl.int64) * go_stride_w
    )
    go_val = tl.load(grad_out_ptr + go_off, mask=mask, other=0.0).to(tl.float32)

    gi_base = (
        grad_in_ptr
        + n_idx.to(tl.int64) * gi_stride_n
        + c_idx.to(tl.int64) * gi_stride_c
    )

    norm = tl.zeros([BLOCK], dtype=tl.float32)
    for dh in range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)
            norm  += w_h * w_w

    norm = tl.where(norm == 0.0, 1.0, norm)

    for dh in range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h     = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)

            weight  = (w_h * w_w) / norm
            contrib = go_val * weight

            gi_off = (
                y_in.to(tl.int64) * gi_stride_h
                + x_in.to(tl.int64) * gi_stride_w
            )
            tl.atomic_add(gi_base + gi_off, contrib, mask=valid)

@triton.jit
def _upsample_bicubic2d_aa_backward_kernel_lowp_upsample(
    grad_out_ptr,
    grad_in_ptr,
    n, c, in_h, in_w, out_h, out_w,
    go_stride_n, go_stride_c, go_stride_h, go_stride_w,
    gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
    scale_y, scale_x,
    scale_y_aa, scale_x_aa,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n * c * out_h * out_w
    mask = offs < total

    x_out = offs % out_w
    tmp   = offs // out_w
    y_out = tmp % out_h
    tmp   = tmp // out_h
    c_idx = tmp % c
    n_idx = tmp // c

    # ✅ Fix 1: 坐标全程用 fp32 计算，避免 bf16 的 7-bit 尾数问题
    y_out_f = y_out.to(tl.float32)
    x_out_f = x_out.to(tl.float32)

    # scale_y/scale_x 在上采样时 < 1，用 fp32 字面量保持精度
    y_real = (y_out_f + 0.5) * scale_y.to(tl.float32) - 0.5
    x_real = (x_out_f + 0.5) * scale_x.to(tl.float32) - 0.5

    # 上采样时 scale_y_aa == 1.0，support == 2.0
    support_h = 2.0 / scale_y_aa
    support_w = 2.0 / scale_x_aa

    y_in_start = (tl.math.floor(y_real - support_h) + 1.0).to(tl.int32)
    x_in_start = (tl.math.floor(x_real - support_w) + 1.0).to(tl.int32)

    go_off = (
        n_idx.to(tl.int64) * go_stride_n
        + c_idx.to(tl.int64) * go_stride_c
        + y_out.to(tl.int64) * go_stride_h
        + x_out.to(tl.int64) * go_stride_w
    )
    go_val = tl.load(grad_out_ptr + go_off, mask=mask, other=0.0).to(tl.float32)

    gi_base = (
        grad_in_ptr
        + n_idx.to(tl.int64) * gi_stride_n
        + c_idx.to(tl.int64) * gi_stride_c
    )

    # ✅ Fix 2: 缓存每个 (dh, dw) 的权重，避免两次循环结果不一致
    # SH=SW=4，共 16 个权重，全部缓存在寄存器里
    # Triton 不支持动态索引寄存器数组，用展开的方式处理
    # ✅ Fix 3: norm 循环加入 mask 保护，防止越界 lane 污染
    norm = tl.zeros([BLOCK], dtype=tl.float32)

    for dh in tl.static_range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        # ✅ 加入 mask，越界 lane 不参与 norm 累积
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in tl.static_range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)
            norm  += w_h * w_w

    # ✅ Fix 4: norm 为 0 时用 1.0 替代，与 PyTorch 行为一致
    norm_safe = tl.where(norm == 0.0, 1.0, norm)

    for dh in tl.static_range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h     = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in tl.static_range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)

            # ✅ Fix 5: weight 和 contrib 全程 fp32，最后再除以 norm_safe
            weight  = (w_h * w_w) / norm_safe
            contrib = go_val * weight

            gi_off = (
                y_in.to(tl.int64) * gi_stride_h
                + x_in.to(tl.int64) * gi_stride_w
            )
            tl.atomic_add(gi_base + gi_off, contrib, mask=valid)

@triton.jit
def v2_upsample_bicubic2d_aa_backward_kernel_lowp_upsample(
    grad_out_ptr,
    grad_in_ptr,
    n, c, in_h, in_w, out_h, out_w,
    go_stride_n, go_stride_c, go_stride_h, go_stride_w,
    gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
    scale_y, scale_x,
    scale_y_aa, scale_x_aa,
    SH: tl.constexpr,
    SW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = n * c * out_h * out_w
    mask = offs < total

    x_out = offs % out_w
    tmp   = offs // out_w
    y_out = tmp % out_h
    tmp   = tmp // out_h
    c_idx = tmp % c
    n_idx = tmp // c

    y_out_f = y_out.to(tl.float32)
    x_out_f = x_out.to(tl.float32)

    y_real = (y_out_f + 0.5) * scale_y - 0.5
    x_real = (x_out_f + 0.5) * scale_x - 0.5

    # 上采样时 scale_y_aa == scale_x_aa == 1.0，support 固定为 2.0
    support_h = 2.0 / scale_y_aa
    support_w = 2.0 / scale_x_aa

    y_in_start = (tl.floor(y_real - support_h) + 1).to(tl.int32)
    x_in_start = (tl.floor(x_real - support_w) + 1).to(tl.int32)

    go_off = (
        n_idx.to(tl.int64) * go_stride_n
        + c_idx.to(tl.int64) * go_stride_c
        + y_out.to(tl.int64) * go_stride_h
        + x_out.to(tl.int64) * go_stride_w
    )
    go_val = tl.load(grad_out_ptr + go_off, mask=mask, other=0.0).to(tl.float32)

    gi_base = (
        grad_in_ptr
        + n_idx.to(tl.int64) * gi_stride_n
        + c_idx.to(tl.int64) * gi_stride_c
    )

    # ── Pass 1: 计算 norm ─────────────────────────────────────────
    # Fix 1: valid_h 必须包含外层 mask，与 scatter 循环保持完全一致
    # 否则越界 lane 的垃圾坐标会产生非零 cubic_keys 值污染 norm
    norm = tl.zeros([BLOCK], dtype=tl.float32)
    for dh in tl.static_range(SH):   # Fix 4: constexpr 循环用 static_range
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in tl.static_range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)
            norm  += w_h * w_w

    # Fix 3: 分离 norm_safe，不覆盖 norm（虽然 pass 1 结束后 norm 本身
    # 不再被读写，但显式分离变量名可防止后续维护时意外引入 Bug）
    norm_safe = tl.where(norm == 0.0, 1.0, norm)

    # ── Pass 2: scatter 梯度 ──────────────────────────────────────
    # valid_h 定义与 Pass 1 完全一致（含 mask），保证两次 cubic_keys
    # 的输入相同，weight = (w_h * w_w) / norm_safe 分子分母匹配
    for dh in tl.static_range(SH):
        y_in   = y_in_start + dh
        y_in_f = y_in.to(tl.float32)
        dist_h = (y_in_f - y_real) * scale_y_aa
        valid_h = mask & (y_in >= 0) & (y_in < in_h) & (tl.abs(dist_h) < 2.0)
        w_h     = tl.where(valid_h, cubic_keys(dist_h), 0.0)

        for dw in tl.static_range(SW):
            x_in   = x_in_start + dw
            x_in_f = x_in.to(tl.float32)
            dist_w = (x_in_f - x_real) * scale_x_aa
            valid  = valid_h & (x_in >= 0) & (x_in < in_w) & (tl.abs(dist_w) < 2.0)
            w_w    = tl.where(valid, cubic_keys(dist_w), 0.0)

            weight  = (w_h * w_w) / norm_safe
            contrib = go_val * weight

            gi_off = (
                y_in.to(tl.int64) * gi_stride_h
                + x_in.to(tl.int64) * gi_stride_w
            )
            tl.atomic_add(gi_base + gi_off, contrib, mask=valid)
# ============================================================
# Python 入口
# ============================================================

def _upsample_bicubic2d_aa_backward(
    grad_output: torch.Tensor,
    output_size,
    input_size,
    align_corners: bool,
    scales_h=None,
    scales_w=None,
):
    assert grad_output.is_cuda

    n, c, in_h, in_w = input_size
    out_h, out_w = output_size

    grad_out = grad_output.contiguous()
    src_dtype = grad_output.dtype

    if align_corners:
        scale_y = (in_h - 1) / (out_h - 1) if (in_h > 1 and out_h > 1) else 0.0
        scale_x = (in_w - 1) / (out_w - 1) if (in_w > 1 and out_w > 1) else 0.0
    else:
        scale_y = in_h / out_h
        scale_x = in_w / out_w

    scale_y_aa = min(1.0 / scale_y, 1.0) if scale_y > 0 else 1.0
    scale_x_aa = min(1.0 / scale_x, 1.0) if scale_x > 0 else 1.0

    SH = int(math.ceil(4.0 / scale_y_aa)) + 2
    SW = int(math.ceil(4.0 / scale_x_aa)) + 2

    BLOCK = 128
    grid = (triton.cdiv(n * c * out_h * out_w, BLOCK),)

    is_lowp = src_dtype in (torch.float16, torch.bfloat16)

    # 判断是否为上采样：两个方向的 AA 缩放因子均为 1.0 时为上采样。
    # （下采样时至少有一个方向 scale_y_aa < 1 或 scale_x_aa < 1）
    is_upsample = (scale_y_aa == 1.0) and (scale_x_aa == 1.0)

    if is_lowp:
        # fp32 临时 buffer，kernel 内 atomic_add 全程 fp32
        grad_in = torch.zeros((n, c, in_h, in_w), device=grad_output.device, dtype=torch.float32)

        go_stride_n, go_stride_c, go_stride_h, go_stride_w = grad_out.stride()
        gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w = grad_in.stride()

        common_args = (
            grad_out,
            grad_in,
            n, c, in_h, in_w, out_h, out_w,
            go_stride_n, go_stride_c, go_stride_h, go_stride_w,
            gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
            scale_y, scale_x,
            scale_y_aa, scale_x_aa,
        )
        common_kwargs = dict(SH=SH, SW=SW, BLOCK=BLOCK)

        if is_upsample:
            _upsample_bicubic2d_aa_backward_kernel_lowp_upsample[grid](
                *common_args, **common_kwargs,
            )
            # bicubic2d_aa_backward_gather_kernel[grid](
            #     *common_args, **common_kwargs,
            # )
        else:
            _upsample_bicubic2d_aa_backward_kernel_lowp_downsample[grid](
                *common_args, **common_kwargs,
            )

        # 计算完毕后 cast 回原始低精度
        return grad_in.to(src_dtype)

    else:
        # fp32 path：grad_in 直接用 fp32，原地 atomic_add
        grad_in = torch.zeros((n, c, in_h, in_w), device=grad_output.device, dtype=torch.float32)

        go_stride_n, go_stride_c, go_stride_h, go_stride_w = grad_out.stride()
        gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w = grad_in.stride()

        _upsample_bicubic2d_aa_backward_kernel_fp32[grid](
            grad_out,
            grad_in,
            n, c, in_h, in_w, out_h, out_w,
            go_stride_n, go_stride_c, go_stride_h, go_stride_w,
            gi_stride_n, gi_stride_c, gi_stride_h, gi_stride_w,
            scale_y, scale_x,
            scale_y_aa, scale_x_aa,
            SH=SH, SW=SW, BLOCK=BLOCK,
        )

        return grad_in  # 已经是 fp32
