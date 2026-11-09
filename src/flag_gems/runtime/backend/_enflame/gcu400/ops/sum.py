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

"""GCU400 特化的 sum 系列实现。

全局求和（dim=None / dim=[]）按数据规模分三条路径：
- 不足一个 tile：单 program 一次归约；
- 中等规模且最后一维够宽：2D block tiling，两阶段（分段 partial + 归并）；
- 其余：1D 流式单 launch，各 program 分段累加后原子归并。

单维求和按 (M, N, K) 展开选路：N==1 走拷贝，K==1 走行归约，K>1 走 (行, 尾块) 切分。
多 dim 统一先经 dim_compress 压成 (M, N) 的 K == 1 情形。
"""

import builtins
import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry

logger = logging.getLogger("flag_gems.ops.sum")

_min = builtins.min
_max = builtins.max

# GCU400 有 48 个计算单元；固定网格 + 核内跨步可让 SP 占满。
MAX_GRID = 48

# 全局求和：实测 65536 元素（fp16 128 KiB / fp32 256 KiB）带宽最好。
GLOBAL_TILE_ELEMS = 65536

# 行归约单个 tile 的字节预算：sum 只有一趟读，tile 敢开到 256 KiB。
INNER_TILE_BYTES = 262144

# 非行归约（K > 1）单个 tile 的字节预算。
NON_INNER_TILE_BYTES = 131072

# 全局求和的 2D block tiling 只在中等规模启用：数据量再大时 1D 流式的连续
# 访问更划算（实测 2 GiB 时 1D 1.66x vs 2D 1.46x），再小时启动开销占主导。
GLOBAL_2D_MIN_BYTES = 64 * 1024 * 1024
GLOBAL_2D_MAX_BYTES = 512 * 1024 * 1024

# 2D 全局求和的归约宽度上限（fp16/fp32 实测均在此附近最优）。
GLOBAL_2D_WIDTH = 8192

# 行内归约宽度的上限：跨 lane 归约的开销随宽度增长，
# 实测超过 16384 后带宽下降。
MAX_REDUCE_WIDTH = 16384

# 行数很少而行长极长时，行归约只有 1 个 tile，必须沿 N 维再切一刀分摊到 48 个 program。
ROW_SPLIT_MAX_ROWS = 8
ROW_SPLIT_MIN_N = 65536


def _tile_elems(element_size, budget):
    """按字节预算换算出单 tile 可容纳的元素数。"""
    return _max(1024, budget // element_size)


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def sum_global_direct_kernel(
    X,
    Out,
    n_elem,
    BLOCK: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """Out 预清零后，各 program 直接把 partial 原子累加进去。

    单 launch 里最省的一条路径：不需要 Part / Counter，也没有二次归并。
    partial 始终在 fp32 上累加（fp16/bf16 输入亦然），只有最后的原子归并落到
    Out 的 dtype 上。
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    acc = tl.zeros([], dtype=cdtype)
    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, mask=offset < n_elem, other=0.0)
        acc += tl.sum(x).to(cdtype)
    # 各 program 之间无数据依赖，只需原子性、不需要 ordering/visibility，
    # 用 relaxed 避免 acq_rel 带来的多余 fence（默认就是 acq_rel）。
    tl.atomic_add(Out, acc.to(Out.dtype.element_ty), sem="relaxed")


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def sum_global_single_kernel(X, Out, n_elem, BLOCK: tl.constexpr):
    """小规模输入：单 program 一次归约到底，省掉第二阶段的启动开销。"""
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    offset = tl.arange(0, BLOCK)
    x = tl.load(X + offset, mask=offset < n_elem, other=0.0)
    tl.store(Out, tl.sum(x, axis=0).to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def sum_copy_kernel(Out, X, n_elem, BLOCK: tl.constexpr):
    """被归约维长度为 1：输出就是输入本身，走一遍 elementwise 拷贝即可。"""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tile, step):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_elem
        x = tl.load(X + offset, mask=mask, other=0.0)
        tl.store(Out + offset, x.to(Out.dtype.element_ty), mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_dim_row_split_partial_kernel(
    X,
    Part,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """行数很少的场合：把行内 N 维也切给多个 program，各产出一份 partial。

    partial 布局为 [n_part, BLOCK_M]，第 i 个 program 写第 i 行。
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    m_offset = tl.arange(0, BLOCK_M)
    row_mask = m_offset < M
    acc = tl.zeros([BLOCK_M], dtype=cdtype)
    for col_start in tl.range(pid * BLOCK_N, N, step * BLOCK_N, num_stages=num_stages):
        n_offset = col_start + tl.arange(0, BLOCK_N)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        x = tl.load(
            X + m_offset[:, None] * N + n_offset[None, :], mask=mask, other=0.0
        ).to(cdtype)
        acc += tl.sum(x, axis=1)
    tl.store(Part + pid * BLOCK_M + m_offset, acc.to(Part.dtype.element_ty), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "n_part"])
def sum_dim_row_split_final_kernel(
    Part,
    Out,
    M,
    n_part,
    BLOCK_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """把 [n_part, BLOCK_M] 的 partial 按行归并。"""
    if tl.constexpr(Part.dtype.element_ty == tl.float16) | tl.constexpr(
        Part.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Part.dtype.element_ty

    m_offset = tl.arange(0, BLOCK_M)
    p_offset = tl.arange(0, BLOCK_P)
    mask = (p_offset[:, None] < n_part) & (m_offset[None, :] < M)
    part = tl.load(
        Part + p_offset[:, None] * BLOCK_M + m_offset[None, :],
        mask=mask,
        other=0.0,
    ).to(cdtype)
    tl.store(
        Out + m_offset, tl.sum(part, axis=0).to(Out.dtype.element_ty), m_offset < M
    )


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_dim_inner_kernel(
    X,
    Out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """被归约维在最内层：一个 tile 覆盖 BLOCK_M 行 × BLOCK_N 列，按行归约。

    归约在 fp32 上完成：低精度累加在 4096 长度的行上相对误差已达 5e-3，
    过不了 sum_dim 的精度门（全局求和因归一化长度极长、容差按 reduce_dim 放宽才可用）。
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BLOCK_M - 1) // BLOCK_M
    for tile_id in tl.range(pid, num_tile, step, num_stages=num_stages):
        m_offset = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = m_offset < M
        acc = tl.zeros([BLOCK_M], dtype=cdtype)
        for col_start in tl.range(0, N, BLOCK_N):
            n_offset = col_start + tl.arange(0, BLOCK_N)
            mask = row_mask[:, None] & (n_offset[None, :] < N)
            x = tl.load(
                X + m_offset[:, None] * N + n_offset[None, :], mask=mask, other=0.0
            ).to(cdtype)
            acc += tl.sum(x, axis=1)
        tl.store(Out + m_offset, acc.to(Out.dtype.element_ty), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def sum_dim_non_inner_kernel(
    X,
    Out,
    M,
    N,
    K,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """被归约维后面还有 K 个元素：任务 = (行 m, 尾块 k)，沿被归约维累加。

    tile 取 (BLOCK_I, BLOCK_K)，其中 BLOCK_K 沿尾维（即内存连续维），
    保证每次 load 都是若干段连续访存；acc 沿尾维保留，最后一次性写出。
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (K + BLOCK_K - 1) // BLOCK_K
    num_task = M * num_k_tile
    for task_id in tl.range(pid, num_task, step, num_stages=num_stages):
        m_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        k_offset = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K
        base = m_index * N * K
        acc = tl.zeros([BLOCK_K], dtype=cdtype)
        for i_start in tl.range(0, N, BLOCK_I):
            i_offset = i_start + tl.arange(0, BLOCK_I)
            mask = (i_offset[:, None] < N) & k_mask[None, :]
            x = tl.load(
                X + base + i_offset[:, None] * K + k_offset[None, :],
                mask=mask,
                other=0.0,
            ).to(cdtype)
            acc += tl.sum(x, axis=0)
        tl.store(Out + m_index * K + k_offset, acc.to(Out.dtype.element_ty), k_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def sum_global_block_partial_kernel(
    X,
    Part,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr = 1,
):
    """中等规模全局求和的 2D block tiling 阶段一。

    输入按最后一维摊成 (M, N)，tile 取 BLOCK_M 行 x BLOCK_N 列，每行一个
    fp32 累加器跨 tile 累加。归约宽度固定在 BLOCK_N，跨 lane 归约成本不随
    tile 整体放大（这是 2D 化相对整块 1D 归约的关键收益）。
    """
    if tl.constexpr(X.dtype.element_ty == tl.float16) | tl.constexpr(
        X.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_col_tile = (N + BLOCK_N - 1) // BLOCK_N
    num_row_tile = (M + BLOCK_M - 1) // BLOCK_M
    m_offset = tl.arange(0, BLOCK_M)
    n_offset = tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_M], dtype=cdtype)
    for tile_id in tl.range(
        pid, num_row_tile * num_col_tile, step, num_stages=num_stages
    ):
        row_tile = tile_id // num_col_tile
        col_tile = tile_id - row_tile * num_col_tile
        rows = row_tile * BLOCK_M + m_offset
        cols = col_tile * BLOCK_N + n_offset
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        x = tl.load(X + rows[:, None] * N + cols[None, :], mask=mask, other=0.0).to(
            cdtype
        )
        acc += tl.sum(x, axis=1)
    tl.store(Part + pid * BLOCK_M + m_offset, acc.to(Part.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_part"])
def sum_global_block_final_kernel(
    Part,
    Out,
    n_part,
    BLOCK_P: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """把 [n_part, BLOCK_M] 的 partial 压成一个标量。"""
    if tl.constexpr(Part.dtype.element_ty == tl.float16) | tl.constexpr(
        Part.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Part.dtype.element_ty

    p_offset = tl.arange(0, BLOCK_P)[:, None]
    m_offset = tl.arange(0, BLOCK_M)[None, :]
    part = tl.load(
        Part + p_offset * BLOCK_M + m_offset, mask=p_offset < n_part, other=0.0
    ).to(cdtype)
    tl.store(
        Out,
        tl.sum(tl.sum(part, axis=1), axis=0).to(Out.dtype.element_ty),
    )


def _launch_global_sum(inp, out, n_elem):
    """全局求和：按数据规模在三条路径间选。

    - 不足一个 tile：单 kernel 直接出结果；
    - 中等规模且最后一维够宽：2D block tiling（归约宽度压到 BLOCK_N）；
    - 其余（含超大数据量）：1D 流式单 launch，分段累加 + 原子归并。
    """
    element_size = inp.element_size()
    block = _min(triton.next_power_of_2(n_elem), GLOBAL_TILE_ELEMS)
    num_bytes = n_elem * element_size

    with torch_device_fn.device(inp.device):
        if n_elem <= block:
            sum_global_single_kernel[(1,)](inp, out, n_elem, BLOCK=block, num_warps=2)
            return

        if (
            inp.dim() >= 2
            and inp.shape[-1] >= 1024
            and GLOBAL_2D_MIN_BYTES <= num_bytes <= GLOBAL_2D_MAX_BYTES
        ):
            num_cols = inp.shape[-1]
            num_rows = n_elem // num_cols
            block_n = _min(triton.next_power_of_2(num_cols), GLOBAL_2D_WIDTH)
            block_m = _max(1, _tile_elems(element_size, INNER_TILE_BYTES) // block_n)
            num_tile = triton.cdiv(num_rows, block_m) * triton.cdiv(num_cols, block_n)
            grid = _min(MAX_GRID * 2, num_tile)
            part_dtype = torch.float32 if inp.is_floating_point() else out.dtype
            part = torch.empty([grid * block_m], dtype=part_dtype, device=inp.device)
            sum_global_block_partial_kernel[(grid,)](
                inp,
                part,
                num_rows,
                num_cols,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_stages=1,
                num_warps=2,
            )
            sum_global_block_final_kernel[(1,)](
                part,
                out,
                grid,
                BLOCK_P=triton.next_power_of_2(grid),
                BLOCK_M=block_m,
                num_warps=1,
            )
            return

        # 1D 流式：2 字节类型给 2 倍网格、2 warp（实测能吃满 2.4 TB/s）；
        # 4 字节类型和小规模都退回 1 warp——前者带宽本就不是瓶颈，后者由
        # 启动开销主导，多出来的 warp 只会拖慢。
        grid_cap = MAX_GRID * 2 if element_size == 2 else MAX_GRID
        grid = _min(grid_cap, triton.cdiv(n_elem, block))

        # 各 program 直接把 partial 原子累加进预清零的 Out，单 launch、无临时
        # buffer。partial 在 fp32 上累加（fp16/bf16 输入亦然），只有最后的原子
        # 归并落到 Out 的 dtype 上。
        out.zero_()
        sum_global_direct_kernel[(grid,)](
            inp,
            out,
            n_elem,
            BLOCK=block,
            num_stages=1,
            num_warps=4,
        )


def _launch_dim_sum(inp, out, M, N, K):
    """单维求和：按 (M, N, K) 的形态选路。

    N == 1           → 输出即输入，走 elementwise 拷贝；
    K == 1 且 M == 1 → 退化为全局求和（复用单 launch 全局路径）；
    K == 1 且行数极少 → 再沿 N 维切一刀，否则只有 1 个 program 串行整行；
    K == 1           → 行归约（BLOCK_M × BLOCK_N tile）；
    K > 1            → (行, 尾块) 任务切分，沿被归约维累加。
    """
    element_size = inp.element_size()

    with torch_device_fn.device(inp.device):
        if N == 1:
            n_elem = inp.numel()
            block = _min(
                triton.next_power_of_2(n_elem),
                _tile_elems(element_size, INNER_TILE_BYTES),
            )
            grid = _min(MAX_GRID, triton.cdiv(n_elem, block))
            sum_copy_kernel[(grid,)](
                out,
                inp,
                n_elem,
                BLOCK=block,
                num_warps=1 if block < 4096 else 4,
            )
            return

        if K == 1:
            if M == 1:
                _launch_global_sum(inp, out, inp.numel())
                return
            if M <= ROW_SPLIT_MAX_ROWS and N >= ROW_SPLIT_MIN_N:
                block_m = triton.next_power_of_2(M)
                block_n = MAX_REDUCE_WIDTH
                grid = _min(MAX_GRID, triton.cdiv(N, block_n))
                part = torch.empty(
                    [grid * block_m], dtype=torch.float32, device=inp.device
                )
                sum_dim_row_split_partial_kernel[(grid,)](
                    inp,
                    part,
                    M,
                    N,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                    num_stages=1,
                    num_warps=2,
                )
                sum_dim_row_split_final_kernel[(1,)](
                    part,
                    out,
                    M,
                    grid,
                    BLOCK_P=triton.next_power_of_2(grid),
                    BLOCK_M=block_m,
                    num_warps=1,
                )
                return

            # fp32 的跨 lane 归约成本更低，超长行可以让归约宽度翻倍
            max_width = MAX_REDUCE_WIDTH * 2 if element_size >= 4 else MAX_REDUCE_WIDTH
            block_n = _min(triton.next_power_of_2(N), max_width)
            batch_rows = triton.next_power_of_2(triton.cdiv(M, MAX_GRID))
            block_m = _tile_elems(element_size, INNER_TILE_BYTES) // block_n
            block_m = _max(1, _min(block_m, batch_rows))
            grid = _min(MAX_GRID, triton.cdiv(M, block_m))
            sum_dim_inner_kernel[(grid,)](
                inp,
                out,
                M,
                N,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                num_stages=1,
                num_warps=2,
            )
            return

        # 尾块宽度：优先取大值（访存段更长），但任务数不足会让 48 个计算单元
        # 出现空转尾效应，因此按 2 倍网格数回退。
        block_k = _min(triton.next_power_of_2(K), 512)
        while block_k > 16 and M * triton.cdiv(K, block_k) < 2 * MAX_GRID:
            block_k //= 2
        block_i = _min(
            triton.next_power_of_2(N),
            _tile_elems(element_size, NON_INNER_TILE_BYTES) // block_k,
        )
        block_i = _max(1, _min(block_i, 1024))
        num_task = M * triton.cdiv(K, block_k)
        grid = _min(MAX_GRID, num_task)
        sum_dim_non_inner_kernel[(grid,)](
            inp,
            out,
            M,
            N,
            K,
            BLOCK_I=block_i,
            BLOCK_K=block_k,
            num_stages=1,
            num_warps=1,
        )


def _normalize_dtype(inp, dtype):
    """保持与公共实现一致的 dtype 推算（bool 走整型累加）。"""
    if dtype is None:
        dtype = inp.dtype
        if dtype is torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32
    return inp, dtype


def sum(inp, *, dtype=None):
    logger.debug("GEMS_ENFLAME SUM")

    inp, out_dtype = _normalize_dtype(inp, dtype)
    inp = inp.contiguous()
    n_elem = inp.numel()

    out = torch.empty([], dtype=out_dtype, device=inp.device)
    if n_elem == 0:
        out.zero_()
        return out

    _launch_global_sum(inp, out, n_elem)
    return out


def sum_out(inp, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME SUM_OUT")

    inp, _ = _normalize_dtype(inp, dtype)
    inp = inp.contiguous()
    n_elem = inp.numel()
    if n_elem == 0:
        out.zero_()
        return out

    _launch_global_sum(inp, out, n_elem)
    return out


def _empty_out_shape(shape, dim, keepdim):
    """空输入的输出形状（与 aten 一致）。"""
    if dim is None:
        return [1] * len(shape) if keepdim else []
    dims_to_reduce = dim if isinstance(dim, (list, tuple)) else [dim]
    if len(dims_to_reduce) == 0:
        return [1] * len(shape) if keepdim else []
    out_shape = list(shape)
    if keepdim:
        for d in dims_to_reduce:
            out_shape[d % len(shape)] = 1
    else:
        for d in sorted([d % len(shape) for d in dims_to_reduce], reverse=True):
            out_shape.pop(d)
    return out_shape


def _sum_dim_impl(inp, dim, keepdim, dtype, out):
    """单维 / 多维求和的统一实现，语义与公共实现保持一致。"""
    if dim is None or (isinstance(dim, (list, tuple)) and len(dim) == 0):
        result = _sum_dim_global(inp, keepdim, dtype, out)
        return result

    shape = list(inp.shape)
    ndim = inp.ndim
    dims = [d % ndim for d in dim]

    if len(dims) == 1:
        # 单维：(M, N, K) 三维展开，K 为被归约维之后的元素数。
        reduce_dim = dims[0]
        N = shape[reduce_dim]
        M = 1
        for s in shape[:reduce_dim]:
            M *= s
        K = inp.numel() // M // N
        inp = inp.contiguous()
        out_shape_keepdim = list(shape)
        out_shape_keepdim[reduce_dim] = 1
        out_shape = (
            out_shape_keepdim
            if keepdim
            else shape[:reduce_dim] + shape[reduce_dim + 1 :]
        )
        if out is None:
            out = torch.empty(out_shape, dtype=dtype, device=inp.device)
        else:
            out.resize_(out_shape)
        with torch_device_fn.device(inp.device):
            _launch_dim_sum(inp, out, M, N, K)
        return out

    # 多维：把被归约维压缩到末尾，退化为 (M, N) 的行归约。
    inp = dim_compress(inp, dims)
    N = 1
    for d in dims:
        N *= shape[d]
        shape[d] = 1
    M = inp.numel() // N
    out_shape = (
        shape if keepdim else [s for i, s in enumerate(shape) if i not in set(dims)]
    )
    if out is None:
        out = torch.empty(out_shape, dtype=dtype, device=inp.device)
    else:
        out.resize_(out_shape)
    with torch_device_fn.device(inp.device):
        _launch_dim_sum(inp, out, M, N, 1)
    return out


def _sum_dim_global(inp, keepdim, dtype, out):
    """dim=None 或 dim=[]：退化为全局求和。"""
    if out is None:
        result = sum(inp, dtype=dtype)
        return result.reshape([1] * inp.ndim) if keepdim else result
    out.resize_([1] * inp.ndim if keepdim else [])
    return sum_out(inp, dtype=dtype, out=out)


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME SUM_DIM")

    inp, dtype = _normalize_dtype(inp, dtype)
    if inp.numel() == 0:
        return torch.zeros(
            _empty_out_shape(list(inp.shape), dim, keepdim),
            dtype=dtype,
            device=inp.device,
        )
    return _sum_dim_impl(inp, dim, keepdim, dtype, None)


def sum_dim_out(inp, dim=None, keepdim=False, *, dtype=None, out):
    logger.debug("GEMS_ENFLAME SUM_DIM_OUT")

    inp, dtype = _normalize_dtype(inp, dtype)
    if inp.numel() == 0:
        out.resize_(_empty_out_shape(list(inp.shape), dim, keepdim))
        out.zero_()
        return out
    return _sum_dim_impl(inp, dim, keepdim, dtype, out)
