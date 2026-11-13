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

"""GCU400 特化的 prod 系列实现。

结构对齐同目录 sum.py 的 persistent 固定网格 + tile 字节预算体系，并针对
乘法归约做了三处关键替换：

1. 乘法没有 atomic 原语（atomic_add 归并不了 partial 积），跨 program
   归并改用「last-block 收尾」单 launch 模式：各 program 写 partial 后
   acq_rel 自增计数器，最后一个 program 在同一 launch 内归并全部
   partial（对齐 masked_scatter_backward 的 barrier 先例）。
2. 乘法归约树的成本远高于元素级向量乘（实测约一个数量级），主路全部
   采用 2D 元素级累加器：被归约维按 chunk 展开成向量乘 acc *= x，树只
   在每 program 收尾做一次（inner/global），或退化为按 chunk 的窄树
   （non-inner，树深 = BI 沿被归约维）。
3. 主循环免 mask（尾块用 constexpr 分离），power-of-2 形状零谓词开销。

实测（L600/48SP，官方 bench core 形状 kernel 模式 median w1000/r100，
job j20261111113840_b395bdf9，两轮 fresh geomean 1.355/1.309×，改前
0.405×）：
- inner (4096,4096): 233→37µs(fp16)/233→42µs(fp32)；
- non-inner (1024,1024,512): 2224→440µs(fp16)/2393→965µs(fp32)；
- global (1048576,): 61→15µs（单 launch + per-call zeros 计数器）。

单维求和按 (M, N, K) 展开选路：N==1 走拷贝；K==1 且 M==1 退化为全局
归约；K==1 且行数极少再沿 N 切一刀；K==1 走行归约；K>1 走 (行, 尾块)
任务切分。
"""

import builtins
import logging
from functools import reduce as _reduce

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger("flag_gems.ops.prod")

_min = builtins.min
_max = builtins.max

# GCU400 有 48 个计算单元；固定网格 + 核内跨步可让 SP 占满。
MAX_GRID = 48

# 全局归约：小规模单 program 一趟出结果（对齐 sum.py 的实测阈值）。
GLOBAL_TILE_ELEMS = 65536

# persistent 全局归约的 tile / 累加器形状：元素级 acc 跨 tile 累乘，
# 归约树只在收尾做一次。
GLOBAL_R = 8
GLOBAL_W = 1024

# 行归约 acc2d 的元素预算：fp32 下 128KB，控制 DSM 占用（实测最优档）。
INNER_ACC_ELEMS = 32768
INNER_MAX_BN = 4096

# 非行归约 tile：BI 沿被归约维（树深），BK 沿内存连续维。
NON_INNER_BI = 128
NON_INNER_BK = 512
NON_INNER_ACC_ELEMS = 65536

# 行数很少而行长极长时，沿 N 维再切一刀分摊到 48 个 program。
ROW_SPLIT_MAX_ROWS = 8
ROW_SPLIT_MIN_N = 65536
ROW_SPLIT_BN = 4096


@triton.jit
def _reduce_mul(a, b):
    return a * b


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_copy_kernel(Out, X, n_elem, BLOCK: tl.constexpr):
    """被归约维长度为 1：输出就是输入本身，走一遍 elementwise 拷贝。"""
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        mask = offset < n_elem
        x = tl.load(X + offset, mask=mask, other=0)
        tl.store(Out + offset, x.to(Out.dtype.element_ty), mask)


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_global_single_kernel(X, Out, n_elem, BLOCK: tl.constexpr):
    """小规模全局乘积：单 program 一趟归约，无 partial 无计数器。"""
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    offset = tl.arange(0, BLOCK)
    x = tl.load(X + offset, mask=offset < n_elem, other=1).to(cdtype)
    tl.store(Out, tl.reduce(x, 0, _reduce_mul).to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["n_elem"])
def prod_global_kernel(
    X,
    Part,
    Counter,
    Out,
    n_elem,
    R: tl.constexpr,
    W: tl.constexpr,
    NP_BLOCK: tl.constexpr,
):
    """persistent 流式全局乘积：元素级 acc [R, W] 跨 tile 累乘，收尾树一次；
    last-block（acq_rel barrier）在同一 launch 内归并全部 partial。

    Counter 是 fp32 标量（与 Part 同一块 buffer 的尾元素），计数到 48 之内
    在 fp32 上精确；Out 由最后一个 program 独占写出。
    """
    BLOCK: tl.constexpr = R * W
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (n_elem + BLOCK - 1) // BLOCK
    acc = tl.full([R, W], 1, cdtype)
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        offset = tile_id * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(X + offset, mask=offset < n_elem, other=1).to(cdtype)
        acc *= tl.reshape(x, (R, W))
    tl.store(Part + pid, tl.reduce(tl.reduce(acc, 1, _reduce_mul), 0, _reduce_mul))
    barrier = tl.atomic_add(Counter, 1.0, sem="acq_rel")
    if barrier == (tl.num_programs(0) - 1).to(tl.float32):
        offs = tl.arange(0, NP_BLOCK)
        part = tl.load(Part + offs, mask=offs < tl.num_programs(0), other=1.0).to(
            tl.float32
        )
        tl.store(Out, tl.reduce(part, 0, _reduce_mul).to(Out.dtype.element_ty))


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def prod_dim_inner_kernel(
    X,
    Out,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """被归约维在最内层：persistent 行批 tile [BM, BN]，acc2d 元素级累乘，
    每个行 tile 收尾树一次；主循环免 mask（尾块 constexpr 分离）。"""
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_tile = (M + BM - 1) // BM
    for tile_id in tl.range(pid, num_tile, step, num_stages=1):
        m_offset = tile_id * BM + tl.arange(0, BM)
        acc = tl.full([BM, BN], 1, cdtype)
        n_floor = (N // BN) * BN
        for n_start in tl.range(0, n_floor, BN, num_stages=1):
            n_offset = n_start + tl.arange(0, BN)
            if EVEN_M:
                x = tl.load(X + m_offset[:, None] * N + n_offset[None, :])
            else:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=m_offset[:, None] < M,
                    other=1,
                )
            acc *= x.to(cdtype)
        if not EVEN_N:
            n_offset = n_floor + tl.arange(0, BN)
            tail_mask = n_offset[None, :] < N
            if EVEN_M:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=tail_mask,
                    other=1,
                )
            else:
                x = tl.load(
                    X + m_offset[:, None] * N + n_offset[None, :],
                    mask=tail_mask & (m_offset[:, None] < M),
                    other=1,
                )
            acc *= x.to(cdtype)
        out = tl.reduce(acc, 1, _reduce_mul)
        if EVEN_M:
            tl.store(Out + m_offset, out.to(Out.dtype.element_ty))
        else:
            tl.store(Out + m_offset, out.to(Out.dtype.element_ty), mask=m_offset < M)


@libentry()
@triton.jit(do_not_specialize=["M", "N"])
def prod_dim_row_split_partial_kernel(
    X,
    Part,
    M,
    N,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    """行数很少的场合：把行内 N 维也切给多个 program，各产出一份 partial。

    acc2d [BM, BN] 元素级累乘跨 chunk，每 program 收尾树一次；
    partial 布局为 [n_part, BM]，第 i 个 program 写第 i 行。
    """
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    m_offset = tl.arange(0, BM)
    row_mask = m_offset < M
    acc = tl.full([BM, BN], 1, cdtype)
    for col_start in tl.range(pid * BN, N, step * BN, num_stages=1):
        n_offset = col_start + tl.arange(0, BN)
        mask = row_mask[:, None] & (n_offset[None, :] < N)
        x = tl.load(
            X + m_offset[:, None] * N + n_offset[None, :], mask=mask, other=1
        ).to(cdtype)
        acc *= x
    out = tl.reduce(acc, 1, _reduce_mul)
    tl.store(Part + pid * BM + m_offset, out.to(Part.dtype.element_ty), row_mask)


@libentry()
@triton.jit(do_not_specialize=["M", "n_part"])
def prod_dim_row_split_final_kernel(
    Part,
    Out,
    M,
    n_part,
    BLOCK_P: tl.constexpr,
    BM: tl.constexpr,
):
    """把 [n_part, BM] 的 partial 按行归并（乘法）。"""
    if tl.constexpr(Part.dtype.element_ty == tl.float16) | tl.constexpr(
        Part.dtype.element_ty == tl.bfloat16
    ):
        cdtype = tl.float32
    else:
        cdtype = Part.dtype.element_ty

    p_offset = tl.arange(0, BLOCK_P)[:, None]
    m_offset = tl.arange(0, BM)[None, :]
    mask = (p_offset < n_part) & (m_offset < M)
    part = tl.load(Part + p_offset * BM + m_offset, mask=mask, other=1).to(cdtype)
    out = tl.reduce(part, 0, _reduce_mul)
    m_flat = tl.arange(0, BM)
    tl.store(Out + m_flat, out.to(Out.dtype.element_ty), m_flat < M)


@libentry()
@triton.jit(do_not_specialize=["M", "N", "K"])
def prod_dim_non_inner_kernel(
    X,
    Out,
    M,
    N,
    K,
    BI: tl.constexpr,
    BK: tl.constexpr,
    EVEN_I: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    """被归约维后面还有 K 个元素：任务 = (行 m, 尾块 k)，沿被归约维累乘。

    tile 取 (BI, BK)，BK 沿尾维（内存连续维）保证 load 是若干段连续访存；
    每 chunk 一次 BI 深的窄树（列方向全并行），acc [BK] 沿尾维保留。
    注意：树必须在 fp32（或宽整型）上做，cast 要在 reduce 之前。"""
    if (
        tl.constexpr(X.dtype.element_ty == tl.float16)
        | tl.constexpr(X.dtype.element_ty == tl.bfloat16)
        | tl.constexpr(X.dtype.element_ty == tl.float64)
    ):
        cdtype = tl.float32
    else:
        cdtype = X.dtype.element_ty

    pid = tl.program_id(0)
    step = tl.num_programs(0)
    num_k_tile = (K + BK - 1) // BK
    num_task = M * num_k_tile
    for task_id in tl.range(pid, num_task, step, num_stages=1):
        m_index = task_id // num_k_tile
        k_tile = task_id % num_k_tile
        k_offset = k_tile * BK + tl.arange(0, BK)
        base = m_index * N * K
        acc = tl.full([BK], 1, cdtype)
        i_floor = (N // BI) * BI
        for i_start in tl.range(0, i_floor, BI, num_stages=1):
            i_offset = i_start + tl.arange(0, BI)[:, None]
            if EVEN_K:
                x = tl.load(X + base + i_offset * K + k_offset[None, :])
            else:
                x = tl.load(
                    X + base + i_offset * K + k_offset[None, :],
                    mask=k_offset[None, :] < K,
                    other=1,
                )
            acc *= tl.reduce(x.to(cdtype), 0, _reduce_mul)
        if not EVEN_I:
            i_offset = i_floor + tl.arange(0, BI)[:, None]
            tail_mask = i_offset < N
            if EVEN_K:
                x = tl.load(
                    X + base + i_offset * K + k_offset[None, :], mask=tail_mask, other=1
                )
            else:
                x = tl.load(
                    X + base + i_offset * K + k_offset[None, :],
                    mask=tail_mask & (k_offset[None, :] < K),
                    other=1,
                )
            acc *= tl.reduce(x.to(cdtype), 0, _reduce_mul)
        if EVEN_K:
            tl.store(Out + m_index * K + k_offset, acc.to(Out.dtype.element_ty))
        else:
            tl.store(
                Out + m_index * K + k_offset,
                acc.to(Out.dtype.element_ty),
                mask=k_offset < K,
            )


def _launch_global_prod(inp, out, n_elem):
    """全局乘积：小规模单 program；其余 persistent 流式 + last-block 单 launch。

    Counter 与 Part 合并进同一块 zeros buffer（计数器占尾元素），一次
    memset 完成 partial 清零与计数器清零。"""
    with torch_device_fn.device(inp.device):
        if n_elem <= GLOBAL_TILE_ELEMS:
            block = triton.next_power_of_2(n_elem)
            prod_global_single_kernel[(1,)](inp, out, n_elem, BLOCK=block, num_warps=2)
            return

        block = GLOBAL_R * GLOBAL_W
        num_tile = triton.cdiv(n_elem, block)
        grid = _min(MAX_GRID, num_tile)
        buf = torch.zeros(grid + 1, dtype=torch.float32, device=inp.device)
        prod_global_kernel[(grid,)](
            inp,
            buf[:grid],
            buf[grid:],
            out,
            n_elem,
            R=GLOBAL_R,
            W=GLOBAL_W,
            NP_BLOCK=triton.next_power_of_2(grid),
            num_warps=1,
        )


def _launch_dim_prod(inp, out, M, N, K):
    """单维乘积：按 (M, N, K) 形态选路。

    N == 1           → 输出即输入，走 elementwise 拷贝；
    K == 1 且 M == 1 → 退化为全局乘积（复用单 launch 全局路径）；
    K == 1 且行数极少 → 再沿 N 维切一刀，否则只有 1 个 program 串行整行；
    K == 1           → 行归约（persistent [BM, BN] acc2d tile）；
    K > 1            → (行, 尾块) 任务切分，沿被归约维窄树累乘。
    """
    with torch_device_fn.device(inp.device):
        if N == 1:
            n_elem = inp.numel()
            block = _min(
                triton.next_power_of_2(n_elem),
                _max(1024, 262144 // inp.element_size()),
            )
            grid = _min(MAX_GRID, triton.cdiv(n_elem, block))
            prod_copy_kernel[(grid,)](
                out, inp, n_elem, BLOCK=block, num_warps=1 if block < 4096 else 4
            )
            return

        if K == 1:
            if M == 1:
                _launch_global_prod(inp, out, inp.numel())
                return
            if M <= ROW_SPLIT_MAX_ROWS and N >= ROW_SPLIT_MIN_N:
                block_m = triton.next_power_of_2(M)
                block_n = ROW_SPLIT_BN
                grid = _min(MAX_GRID, triton.cdiv(N, block_n))
                part = torch.empty(
                    [grid * block_m], dtype=torch.float32, device=inp.device
                )
                prod_dim_row_split_partial_kernel[(grid,)](
                    inp,
                    part,
                    M,
                    N,
                    BM=block_m,
                    BN=block_n,
                    num_warps=2,
                )
                prod_dim_row_split_final_kernel[(1,)](
                    part,
                    out,
                    M,
                    grid,
                    BLOCK_P=triton.next_power_of_2(grid),
                    BM=block_m,
                    num_warps=1,
                )
                return

            # 行归约：acc2d 元素预算封顶，行批随行数摊到 48 个 program。
            block_n = _min(triton.next_power_of_2(N), INNER_MAX_BN)
            block_m = _max(
                1,
                _min(
                    INNER_ACC_ELEMS // block_n,
                    triton.next_power_of_2(triton.cdiv(M, MAX_GRID)),
                ),
            )
            grid = _min(MAX_GRID, triton.cdiv(M, block_m))
            prod_dim_inner_kernel[(grid,)](
                inp,
                out,
                M,
                N,
                BM=block_m,
                BN=block_n,
                EVEN_M=(M % block_m == 0),
                EVEN_N=(N % block_n == 0),
                num_warps=4 if block_m * block_n >= 8192 else 2,
            )
            return

        # 尾块宽度固定 512：实测 [BI, BK] tile 的 tl.reduce(axis=0) 在
        # BK <= 128 时落入 ~55 Melem/s 的慢路径（30×）、BK = 256 间歇性
        # 病态，只有 BK >= 512 稳定在快路径；宁可任务数少于 48 也不再
        # 向下折半（折半只会制造病态档位）。
        block_k = _min(triton.next_power_of_2(K), NON_INNER_BK)
        block_i = _max(
            1,
            _min(
                triton.next_power_of_2(N),
                NON_INNER_ACC_ELEMS // block_k,
                NON_INNER_BI,
            ),
        )
        num_task = M * triton.cdiv(K, block_k)
        grid = _min(MAX_GRID, num_task)
        prod_dim_non_inner_kernel[(grid,)](
            inp,
            out,
            M,
            N,
            K,
            BI=block_i,
            BK=block_k,
            EVEN_I=(N % block_i == 0),
            EVEN_K=(K % block_k == 0),
            num_warps=4,
        )


def _normalize_dtype(inp, dtype):
    """保持与 sum.py 一致的 dtype 归一（bool 走整型）。"""
    if dtype is None:
        dtype = inp.dtype
        if dtype == torch.bool:
            inp = inp.to(torch.int32)
            dtype = torch.int32
    return inp, dtype


def prod(inp, *, dtype=None):
    logger.debug("GEMS_ENFLAME PROD")

    inp, dtype = _normalize_dtype(inp, dtype)
    inp = inp.contiguous()
    n_elem = inp.numel()

    out = torch.empty([], dtype=dtype, device=inp.device)
    if n_elem == 0:
        out.fill_(1)
        return out

    _launch_global_prod(inp, out, n_elem)
    return out


def prod_dim(inp, dim=None, keepdim=False, *, dtype=None):
    logger.debug("GEMS_ENFLAME PROD_DIM")

    if not (-inp.ndim <= dim < inp.ndim):
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-inp.ndim}, {inp.ndim - 1}])"
        )
    inp, dtype = _normalize_dtype(inp, dtype)

    shape = list(inp.shape)
    d = dim % inp.ndim
    N = shape[d]
    M = _reduce(lambda x, y: x * y, shape[:d], 1)
    K = _reduce(lambda x, y: x * y, shape[d + 1 :], 1)
    shape[d] = 1
    out = torch.empty(shape, dtype=dtype, device=inp.device)

    if M == 0 or K == 0:
        # A spectator dimension is empty: the output is a valid empty tensor.
        if not keepdim:
            out = torch.squeeze(out, d)
        return out
    if N == 0:
        # The product over an empty dimension is the identity, 1 (unlike max/min
        # which have no identity). Fill the output rather than launching.
        out.fill_(1)
        if not keepdim:
            out = torch.squeeze(out, d)
        return out

    inp = inp.contiguous()
    with torch_device_fn.device(inp.device):
        _launch_dim_prod(inp, out, M, N, K)
    if not keepdim:
        out = torch.squeeze(out, d)
    return out
