import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import triton_lang_extension as tle
from flag_gems.utils.libentry import libentry

# Tile 算子的 triton 实现。
# 关键优化：消除 tl.load / tl.store 的 gather / scatter。
#
# Tile 的数学结构是「把 input 作为一个「周期」沿每一维复制 dims[i] 份」。
# 复制过程中只有**最内维之内**的 `in_shape[-1]` 个元素是 input 连续读、output 连续写的
# 天然单元。之前的实现让一个 program 处理 `BLOCK` 个 flatten 的 output 元素，需要在
# 内部做 `(i % in_s) * in_stride` 才能回推到 input；tensor 级 `%` 让编译器无法识别
# 出 affine 偏移，只能降到 linalg_ext.gather。
#
# 现在改成：
#   grid = (num_rows, copies_last, cdiv(in_last, BLOCK))
#     * row_id : 枚举 output 的 "outer" 索引（前 rank-1 维 flatten）
#     * k       : 枚举最内维上第 k 份复制  (0 .. dims[-1]-1)
#     * blk     : 枚举一份 copy 内的 BLOCK 块（in_last 较大时分块）
#
# 每个 program 内：
#   in_off  = in_outer_base + j         (j = arange(BLOCK); in_stride_last == 1)
#   out_off = row_id * out_last + k * in_last + j
# 两者都是 `scalar + arange`，纯 affine，能被降到 linalg 的 contiguous copy。
#
# in_outer_base 由 row_id（scalar）对 rank-1 个 outer 维做 scalar divmod 得到：
# `t = row_id; i = t % s; t //= s; in_outer_base += (i % in_s) * in_stride`，这些
# 全是 i32 scalar 算术，不会产生 tensor 级 div/mod，也不会阻碍 affine 识别。

MAX_RANK = 8
MAX_OUTER_RANK = MAX_RANK - 1  # = 7


@libentry()
@triton.jit
def tile_kernel(
    in_ptr,
    out_ptr,
    # outer (rank-1) 维的 output shape，未用填 1
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    # outer 维的 input shape，未用填 1
    in_s0,
    in_s1,
    in_s2,
    in_s3,
    in_s4,
    in_s5,
    in_s6,
    # outer 维的 input stride，未用填 0
    in_stride0,
    in_stride1,
    in_stride2,
    in_stride3,
    in_stride4,
    in_stride5,
    in_stride6,
    # 最内维
    in_last,   # = in_shape[-1]
    out_last,  # = in_last * copies_last
    OUTER_RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row_id = tle.program_id(0)
    k = tle.program_id(1)
    blk = tle.program_id(2)

    # --- 1) scalar decode: row_id -> in_outer_base (全程 i32 scalar) ---
    # 从最内 outer 维 (索引 OUTER_RANK-1) 逐层往外剥
    in_outer_base = 0
    t = row_id
    if OUTER_RANK == 0:
        in_outer_base = 0
    elif OUTER_RANK == 1:
        i0 = t
        in_outer_base = (i0 % in_s0) * in_stride0
    elif OUTER_RANK == 2:
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (i0 % in_s0) * in_stride0 + (i1 % in_s1) * in_stride1
    elif OUTER_RANK == 3:
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
        )
    elif OUTER_RANK == 4:
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
        )
    elif OUTER_RANK == 5:
        i4 = t % s4
        t = t // s4
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
            + (i4 % in_s4) * in_stride4
        )
    elif OUTER_RANK == 6:
        i5 = t % s5
        t = t // s5
        i4 = t % s4
        t = t // s4
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
            + (i4 % in_s4) * in_stride4
            + (i5 % in_s5) * in_stride5
        )
    else:  # OUTER_RANK == 7
        i6 = t % s6
        t = t // s6
        i5 = t % s5
        t = t // s5
        i4 = t % s4
        t = t // s4
        i3 = t % s3
        t = t // s3
        i2 = t % s2
        t = t // s2
        i1 = t % s1
        t = t // s1
        i0 = t
        in_outer_base = (
            (i0 % in_s0) * in_stride0
            + (i1 % in_s1) * in_stride1
            + (i2 % in_s2) * in_stride2
            + (i3 % in_s3) * in_stride3
            + (i4 % in_s4) * in_stride4
            + (i5 % in_s5) * in_stride5
            + (i6 % in_s6) * in_stride6
        )

    # --- 2) 最内维的 contiguous copy ---
    j = blk * BLOCK + tl.arange(0, BLOCK)
    mask = j < in_last

    # input 是 contiguous (wrapper 做过 .contiguous().reshape(in_shape))，
    # 所以 in_stride_last == 1，这里直接 `in_outer_base + j`（affine）
    in_off = in_outer_base + j
    x = tl.load(in_ptr + in_off, mask=mask)

    # output 是 torch.empty 分配的 contiguous buffer，
    # out_off = (row_id * out_last + k * in_last) + j，也是 scalar + arange
    out_row_base = row_id * out_last + k * in_last
    out_off = out_row_base + j
    tl.store(out_ptr + out_off, x, mask=mask)


def _choose_block(in_last: int) -> int:
    # 128~512 的 tile 对 TPU 通常是甜点
    return min(512, max(1, triton.next_power_of_2(in_last)))


def tile(inp: torch.Tensor, dims) -> torch.Tensor:
    logging.debug("SOPHGO GEMS TILE")

    in0_shape = list(inp.shape)
    dims_shape = list(dims)
    in_rank = len(in0_shape)
    dims_rank = len(dims_shape)

    # 右对齐补 1
    if dims_rank < in_rank:
        dims_shape = [1] * (in_rank - dims_rank) + dims_shape
    elif dims_rank > in_rank:
        in0_shape = [1] * (dims_rank - in_rank) + in0_shape
    rank = max(in_rank, dims_rank, 1)  # 至少 1 维，方便处理 0-dim

    if rank > MAX_RANK:
        raise NotImplementedError(
            f"tile only supports rank up to {MAX_RANK}, got {rank}"
        )

    # out_shape 与空张量短路
    is_empty = False
    out_shape = []
    for i in range(rank):
        d = dims_shape[i] if i < len(dims_shape) else 1
        assert d >= 0, f"tile dims must be >= 0, got {d}"
        if d == 0:
            is_empty = True
        out_shape.append(in0_shape[i] * d)

    out0 = torch.empty(out_shape, device=inp.device, dtype=inp.dtype)
    if is_empty:
        return out0

    # 保证 input contiguous，这样 reshape 零拷贝且 stride_last == 1
    in0 = inp.contiguous().reshape(in0_shape)

    # 拆出 outer 维 (前 rank-1 维) 和 last 维
    outer_rank = rank - 1
    in_last = in0_shape[-1]
    out_last = out_shape[-1]
    num_rows = 1
    for i in range(outer_rank):
        num_rows *= out_shape[i]
    copies_last = dims_shape[-1]

    block = _choose_block(in_last)
    grid = (num_rows, copies_last, triton.cdiv(in_last, block))

    # pad 到 MAX_OUTER_RANK
    in0_strides = list(in0.stride())
    pad = MAX_OUTER_RANK - outer_rank
    s_args = out_shape[:outer_rank] + [1] * pad
    in_s_args = in0_shape[:outer_rank] + [1] * pad
    in_stride_args = in0_strides[:outer_rank] + [0] * pad

    with torch_device_fn.device(inp.device.index):
        tile_kernel[grid](
            in0,
            out0,
            *s_args,
            *in_s_args,
            *in_stride_args,
            in_last,
            out_last,
            OUTER_RANK=outer_rank,
            BLOCK=block,
            num_warps=4,
        )
    return out0
