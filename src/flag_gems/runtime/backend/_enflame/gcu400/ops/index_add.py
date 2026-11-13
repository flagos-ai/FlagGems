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
#
# index_add / index_add_ 的 GCU400 特化实现（零原子版本）。
#
# 原始实现用标量原子累加把每个 src 元素累加到 out[index[j]]，而 GCU400 标量
# 原子吞吐只有约 0.39 G elem/s（随机普通写 21.5 G elem/s，相差 55 倍），实测
# core benchmark 仅 0.03~0.07x。本实现彻底移除数据路径上的原子写：
#
# 1. 无重复情形（benchmark 的 randperm 全部命中）走 1D/2D scatter：重复检测
#    在设备上完成（inv[d]=j 的 inverse map + 比较 j 是否等于 inv[d]），确认
#    无重复后每个输出地址只有一个写者，用 load + add + store 替代原子累加
#    （条目 30 ① 地址独占写，零额外流量、零原子）。
# 2. 有重复且规模较小走 index_add_reduce_small_kernel：输出驱动的 one-hot
#    加权 reduce_sum，每个输出地址由唯一 program 写，天然支持任意重复模式。
# 3. 通用重复情形走 index_add_reduce_kernel：由 inv[d] 指定的 owner program
#    用 tl.sum 在 j 维做 reduce_sum 规约再单次写入，全程零原子。
# 4. 检测全程不用 torch.sort（实测设备侧 sort + 同步会把 2048 元素 case 拖到
#    2.6 秒），只读两个 int32 标量做一次 host 同步，同时完成边界校验；
#    J <= 4096 时单 program 检测并自己清零 flags，省一次 zeros launch。
# 5. 静态 1D / 2D kernel（无运行期 codegen），(outer, j, i) 展平到单维 grid，
#    grid.x 超 65535 时由 kernel 内 grid-stride 循环兜底。
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

_INT32_OFFSET_LIMIT = 1 << 31

_BLOCK_J = 2048  # 1D kernel 每 program 处理的 index 元素数
_BLOCK_J_2D = 4  # 2D kernel 在 index 维上的 tile
_BLOCK_I_2D = 256  # 2D kernel 在 inner 维上的 tile

# GCU400 硬件限制：grid.x <= 65535
_MAX_GRID_X = 65535
_MAX_BLOCK_J = 16384
_MAX_2D_TILE_ELEMENTS = 4096  # 2D tile 的 DSM 预算上限

_BLOCK_PROBE = 2048  # 大 J 探测 kernel 的 j tile
_SINGLE_CHECK_MAX_J = 4096  # 单 program 能覆盖的最大 index 长度
_MAX_REDUCE_TILE = 32768  # reduce_small 的 (BLOCK_D * BLOCK_J) 元素上限
# 有重复 index 且规模不大时，输出驱动 reduce_sum 比 owner 扫描更快
_REDUCE_BUDGET = 1 << 25
_BLOCK_P_REDUCE = 256  # owner reduce_sum 扫描 j 的 tile
_BLOCK_I_REDUCE = 32  # owner reduce_sum 的 inner 维 tile
_NUM_WARPS = 4  # 统一 launch 参数，便于按 shape 扫描最优 warp 数


def _next_pow2(value):
    power = 1
    while power < value:
        power <<= 1
    return power


@libentry()
@triton.jit
def index_add_check_kernel(
    index_ptr,
    inv_ptr,
    flags_ptr,
    J,
    D,
    BLOCK_J: tl.constexpr,
):
    """单 program 覆盖全部 index：建 inverse map、查边界、查重复。

    debug_barrier 保证同 program 内 inv 的 store 对后续 load 可见（GCU400 上
    该语义在 sort 算子里已验证成立）。flags 由 host 侧 zeros 初始化，这里只写 1。
    """
    rj = tl.arange(0, BLOCK_J)
    mask = rj < J
    # 单 program 独占 flags，可直接清零，省掉一次 torch.zeros launch
    tl.store(flags_ptr, 0)
    tl.store(flags_ptr + 1, 0)
    tl.debug_barrier()
    d = tl.load(index_ptr + rj, mask=mask, other=-1)
    valid = mask & (d >= 0) & (d < D)
    tl.store(inv_ptr + d, rj.to(tl.int32), mask=valid)
    bad = tl.sum((mask & ~valid).to(tl.int32), axis=0) > 0
    tl.store(flags_ptr, 1, mask=bad)
    tl.debug_barrier()
    owner = tl.load(inv_ptr + d, mask=valid, other=rj.to(tl.int32))
    dup = tl.sum((valid & (owner != rj.to(tl.int32))).to(tl.int32), axis=0) > 0
    tl.store(flags_ptr + 1, 1, mask=dup)


@libentry()
@triton.jit
def index_add_probe_kernel(
    index_ptr,
    inv_ptr,
    flags_ptr,
    J,
    D,
    total_tiles,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """大 J 的 probe：inv[index[j]] = j，并检查边界。"""
    pid = tl.program_id(0)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        rj = tile_id * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = rj < J
        d = tl.load(index_ptr + rj, mask=mask, other=0)
        valid = mask & (d >= 0) & (d < D)
        tl.store(inv_ptr + d, rj.to(tl.int32), mask=valid)
        bad = tl.sum((mask & ~valid).to(tl.int32), axis=0) > 0
        tl.store(flags_ptr, 1, mask=bad)


@libentry()
@triton.jit
def index_add_dup_kernel(
    index_ptr,
    inv_ptr,
    flags_ptr,
    J,
    D,
    total_tiles,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """大 J 的重复检测：多个 j 指向同一 d 时 inv[d] 只能保留其中一个。"""
    pid = tl.program_id(0)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        rj = tile_id * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = rj < J
        d = tl.load(index_ptr + rj, mask=mask, other=0)
        safe = mask & (d >= 0) & (d < D)
        owner = tl.load(inv_ptr + d, mask=safe, other=rj.to(tl.int32))
        mismatch = safe & (owner != rj.to(tl.int32))
        dup = tl.sum(mismatch.to(tl.int32), axis=0) > 0
        tl.store(flags_ptr + 1, 1, mask=dup)


def _inspect_index(index, D, flags, device):
    """判定重复模式并校验边界，返回 (need_reduce, inv)。

    小 J 用一个 program 覆盖全部 index（1 次 launch）；大 J 用 probe + dup
    两个 kernel。flags[0]=越界、flags[1]=重复，一次 tolist() 读两个标量，
    是整条路径里唯一的 host 同步。
    """
    J = index.numel()
    inv = torch.empty((D,), dtype=torch.int32, device=device)
    if J <= _SINGLE_CHECK_MAX_J:
        index_add_check_kernel[(1,)](
            index,
            inv,
            flags,
            J,
            D,
            BLOCK_J=_next_pow2(J),
            num_warps=_NUM_WARPS,
        )
    else:
        block_j = min(_BLOCK_PROBE, _next_pow2(J))
        total = triton.cdiv(J, block_j)
        grid = min(total, _MAX_GRID_X)
        index_add_probe_kernel[(grid,)](
            index,
            inv,
            flags,
            J,
            D,
            total,
            BLOCK_J=block_j,
            GRID_SIZE=grid,
            num_warps=_NUM_WARPS,
        )
        index_add_dup_kernel[(grid,)](
            index,
            inv,
            flags,
            J,
            D,
            total,
            BLOCK_J=block_j,
            GRID_SIZE=grid,
            num_warps=_NUM_WARPS,
        )
    err, dup = flags.tolist()
    assert err == 0, "0 <= index < self.size(dim)"
    return dup > 0, inv


def _tile_layout(J, inner, outer):
    """按 grid.x 上限选 tile，返回 (block_j, block_i, total_tiles, grid_size)。"""
    if inner == 1:
        block_i = 1
        block_j = min(_BLOCK_J, _next_pow2(J))
        cap_j = _MAX_BLOCK_J
    else:
        block_i = min(_BLOCK_I_2D, _next_pow2(inner))
        block_j = _BLOCK_J_2D
        cap_j = max(1, _MAX_2D_TILE_ELEMENTS // block_i)

    blocks_i = triton.cdiv(inner, block_i)
    max_blocks_j = max(1, _MAX_GRID_X // (outer * blocks_i))
    needed_j = _next_pow2(triton.cdiv(J, max_blocks_j))
    block_j = min(cap_j, max(block_j, needed_j))

    total_tiles = outer * triton.cdiv(J, block_j) * blocks_i
    return block_j, block_i, total_tiles, min(total_tiles, _MAX_GRID_X)


@libentry()
@triton.jit
def index_add_reduce_small_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    J,
    D,
    alpha,
    src_so,
    out_oo,
    total_tiles,
    BLOCK_D: tl.constexpr,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """小 shape 的输出驱动 reduce_sum：每个输出地址由唯一 program 写。

    match[d, j] = (index[j] == d)，对 j 维用 tl.sum 加权归约，天然支持
    任意重复模式；边界合法性已由 _inspect_index 校验。
    """
    pid = tl.program_id(0)
    blocks_d = tl.cdiv(D, BLOCK_D)
    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // blocks_d
        db = tile_id % blocks_d

        rd = db * BLOCK_D + tl.arange(0, BLOCK_D)
        rj = tl.arange(0, BLOCK_J)
        jmask = rj < J
        dj = tl.load(index_ptr + rj, mask=jmask, other=-1)

        s = tl.load(src_ptr + o * src_so + rj, mask=jmask, other=0).to(tl.float32)
        match = (dj[None, :] == rd[:, None]) & jmask[None, :]
        acc = tl.sum(tl.where(match, s[None, :], 0.0), axis=1) * alpha

        out_off = o * out_oo + rd
        dmask = rd < D
        cur = tl.load(out_ptr + out_off, mask=dmask, other=0).to(tl.float32)
        tl.store(
            out_ptr + out_off,
            (cur + acc).to(out_ptr.dtype.element_ty),
            mask=dmask,
        )


@libentry()
@triton.jit
def index_add_1d_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    J,
    alpha,
    src_so,
    out_oo,
    total_tiles,
    USE_INT64: tl.constexpr,
    BLOCK_J: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """dim 为最后一维（inner == 1）的快速路径。

    index 无重复时每个 out[o, index[j]] 只有一个写者，用 load-add-store
    完全替代标量原子累加；fp32 累加后再 cast，避免 fp16 二次舍入。
    """
    pid = tl.program_id(0)
    blocks_j = tl.cdiv(J, BLOCK_J)

    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // blocks_j
        jb = tile_id % blocks_j

        rj = jb * BLOCK_J + tl.arange(0, BLOCK_J)
        mask = rj < J

        src_off = o * src_so + rj
        idx = tl.load(index_ptr + rj, mask=mask, other=0)
        val = tl.load(src_ptr + src_off, mask=mask, other=0).to(tl.float32)

        if USE_INT64:
            out_off = o.to(tl.int64) * out_oo + idx.to(tl.int64)
        else:
            out_off = o * out_oo + idx

        cur = tl.load(out_ptr + out_off, mask=mask, other=0).to(tl.float32)
        tl.store(
            out_ptr + out_off,
            (cur + val * alpha).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def index_add_kernel(
    index_ptr,
    src_ptr,
    out_ptr,
    J,
    inner,
    alpha,
    src_so,
    src_sj,
    out_oo,
    out_od,
    total_tiles,
    USE_INT64: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_I: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """通用路径：(outer, j, i) 展平到单维 grid，地址独占写。"""
    pid = tl.program_id(0)
    blocks_i = tl.cdiv(inner, BLOCK_I)
    blocks_j = tl.cdiv(J, BLOCK_J)
    per_outer = blocks_j * blocks_i

    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // per_outer
        rem = tile_id % per_outer
        jb = rem // blocks_i
        ib = rem % blocks_i

        rj = jb * BLOCK_J + tl.arange(0, BLOCK_J)
        ri = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        mask = (rj < J)[:, None] & (ri < inner)[None, :]

        src_off = o * src_so + rj[:, None] * src_sj + ri[None, :]
        idx = tl.load(index_ptr + rj, mask=rj < J, other=0)

        val = tl.load(src_ptr + src_off, mask=mask, other=0).to(tl.float32)

        if USE_INT64:
            base = o.to(tl.int64) * out_oo + idx.to(tl.int64) * out_od
        else:
            base = o * out_oo + idx * out_od
        out_off = base[:, None] + ri[None, :]

        cur = tl.load(out_ptr + out_off, mask=mask, other=0).to(tl.float32)
        tl.store(
            out_ptr + out_off,
            (cur + val * alpha).to(out_ptr.dtype.element_ty),
            mask=mask,
        )


@libentry()
@triton.jit
def index_add_reduce_kernel(
    index_ptr,
    inv_ptr,
    src_ptr,
    out_ptr,
    J,
    inner,
    alpha,
    src_so,
    src_sj,
    out_oo,
    out_od,
    total_tiles,
    USE_INT64: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_I: tl.constexpr,
    GRID_SIZE: tl.constexpr,
):
    """有重复 index 时的 reduce_sum 规约路径（全程零原子）。

    每个 program 对应一个 (o, j, i_tile)；只有 j == inv[index[j]] 的 owner
    用 tl.sum 在 j 维规约该目标的所有贡献并单次写入，保证写者唯一。
    """
    pid = tl.program_id(0)
    blocks_i = tl.cdiv(inner, BLOCK_I)
    per_outer = J * blocks_i

    for tile_id in tl.range(pid, total_tiles, GRID_SIZE):
        o = tile_id // per_outer
        rem = tile_id % per_outer
        j = rem // blocks_i
        ib = rem % blocks_i

        d = tl.load(index_ptr + j)
        owner = tl.load(inv_ptr + d)
        if j == owner:
            ri = ib * BLOCK_I + tl.arange(0, BLOCK_I)
            acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
            for q0 in tl.range(0, J, BLOCK_P):
                rq = q0 + tl.arange(0, BLOCK_P)
                rd = tl.load(index_ptr + rq, mask=rq < J, other=-1)
                hit = (rd == d) & (rq < J)
                src_off = o * src_so + rq[:, None] * src_sj + ri[None, :]
                vals = tl.load(
                    src_ptr + src_off,
                    mask=hit[:, None] & (ri[None, :] < inner),
                    other=0,
                ).to(tl.float32)
                acc += tl.sum(vals, axis=0)

            if USE_INT64:
                base = o.to(tl.int64) * out_oo + d.to(tl.int64) * out_od
            else:
                base = o * out_oo + d * out_od
            out_off = base + ri
            mask_i = ri < inner
            cur = tl.load(out_ptr + out_off, mask=mask_i, other=0).to(tl.float32)
            tl.store(
                out_ptr + out_off,
                (cur + acc * alpha).to(out_ptr.dtype.element_ty),
                mask=mask_i,
            )


def _check_args(inp, dim, index, src):
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    assert (
        inp.ndim == src.ndim
    ), "Self and source should have the same number of dimensions"
    assert index.numel() == src.size(dim), (
        "The dimth dimension of source must have the same size as "
        "the length of index"
    )
    for d in range(inp.ndim):
        if d != dim % inp.ndim:
            assert inp.size(d) == src.size(
                d
            ), "src.size(d) == self.size(d) for all dimensions d != dim"


def _index_add_impl(out, index, src, dim, alpha):
    """把 index_add 的加和写进 out（out 已是 inp 的副本或 in-place 目标）。"""
    dim = dim % out.ndim
    shape = out.shape
    D = shape[dim]
    outer = 1
    for size in shape[:dim]:
        outer *= size
    inner = 1
    for size in shape[dim + 1 :]:
        inner *= size
    J = index.numel()
    if J == 0 or inner == 0 or outer == 0:
        return

    # (outer, D, inner) 三段展平要求行主序连续；非连续时做一次拷贝，
    # in-place 场景结束后再拷回（保证对调用方可见的仍是同一个张量）
    out_work = out if out.is_contiguous() else out.contiguous()
    src_work = src if src.is_contiguous() else src.contiguous()

    idx = index if index.dtype == torch.int32 else index.to(torch.int32)

    use_int64 = out_work.numel() >= _INT32_OFFSET_LIMIT
    src_so = J * inner
    src_sj = inner
    out_oo = D * inner
    out_od = inner

    # 单 program 检测可自己清零 flags（省一次 zeros launch）；大 J 的多
    # program 检测必须先把 flags 初始化为 0
    if J <= _SINGLE_CHECK_MAX_J:
        flags = torch.empty((2,), dtype=torch.int32, device=out.device)
    else:
        flags = torch.zeros((2,), dtype=torch.int32, device=out.device)

    with torch_device_fn.device(out.device):
        need_reduce, inv = _inspect_index(idx, D, flags, out.device)
        if need_reduce and inner == 1 and outer * D * J <= _REDUCE_BUDGET:
            # 有重复且规模小：输出驱动 reduce_sum，每个 d 由唯一 program 写
            block_j = _next_pow2(J)
            block_d = min(_next_pow2(D), max(1, _MAX_REDUCE_TILE // block_j))
            total = outer * triton.cdiv(D, block_d)
            grid = min(total, _MAX_GRID_X)
            index_add_reduce_small_kernel[(grid,)](
                idx,
                src_work,
                out_work,
                J,
                D,
                alpha,
                src_so,
                out_oo,
                total,
                BLOCK_D=block_d,
                BLOCK_J=block_j,
                GRID_SIZE=grid,
                num_warps=_NUM_WARPS,
            )
        elif need_reduce:
            # 通用重复路径：owner program 用 tl.sum 规约后独占写
            block_i = min(_BLOCK_I_REDUCE, _next_pow2(inner))
            total = outer * J * triton.cdiv(inner, block_i)
            grid = min(total, _MAX_GRID_X)
            index_add_reduce_kernel[(grid,)](
                idx,
                inv,
                src_work,
                out_work,
                J,
                inner,
                alpha,
                src_so,
                src_sj,
                out_oo,
                out_od,
                total,
                USE_INT64=use_int64,
                BLOCK_P=_BLOCK_P_REDUCE,
                BLOCK_I=block_i,
                GRID_SIZE=grid,
                num_warps=_NUM_WARPS,
            )
        else:
            block_j, block_i, total, grid = _tile_layout(J, inner, outer)
            if inner == 1:
                index_add_1d_kernel[(grid,)](
                    idx,
                    src_work,
                    out_work,
                    J,
                    alpha,
                    src_so,
                    out_oo,
                    total,
                    USE_INT64=use_int64,
                    BLOCK_J=block_j,
                    GRID_SIZE=grid,
                    num_warps=_NUM_WARPS,
                )
            else:
                index_add_kernel[(grid,)](
                    idx,
                    src_work,
                    out_work,
                    J,
                    inner,
                    alpha,
                    src_so,
                    src_sj,
                    out_oo,
                    out_od,
                    total,
                    USE_INT64=use_int64,
                    BLOCK_J=block_j,
                    BLOCK_I=block_i,
                    GRID_SIZE=grid,
                    num_warps=_NUM_WARPS,
                )

    if out_work is not out:
        out.copy_(out_work)


def index_add(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ENFLAME INDEX_ADD")
    _check_args(inp, dim, index, src)
    out = inp.clone()
    _index_add_impl(out, index, src, dim, alpha)
    return out


def index_add_(inp, dim, index, src, alpha=1):
    logger.debug("GEMS_ENFLAME INDEX_ADD_")
    _check_args(inp, dim, index, src)
    _index_add_impl(inp, index, src, dim, alpha)
    return inp
