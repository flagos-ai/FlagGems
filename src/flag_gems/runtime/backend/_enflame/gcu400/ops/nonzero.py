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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# 单 block 路径的最大元素数(小于等于该值走单 kernel 方案)
SINGLE_BLOCK_LIMIT = 65536
# 单 block 路径的最小 BLOCK(避免过小向量触发编译器缺陷)
MIN_BLOCK = 1024


@libentry()
@triton.jit
def nonzero_small_kernel_1d(
    inp,
    out,
    total,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz
    out_off = out_pos.to(tl.int64)
    tl.store(out + out_off, offs, mask=(nz == 1))
    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit(do_not_specialize=["d0", "d1"])
def nonzero_small_kernel_2d(
    inp,
    out,
    total,
    n_elements,
    d0,
    d1,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz
    out_off = out_pos.to(tl.int64) * 2

    idx_flat = offs
    r1 = idx_flat % d1
    idx_flat //= d1
    r0 = idx_flat
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))
    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit
def nonzero_small_kernel_2d_pow2(
    inp,
    out,
    total,
    n_elements,
    d1_log2,
    d1_mask,
    BLOCK_SIZE: tl.constexpr,
):
    """1-block 2D 特化: d1 为 2 的幂, 移位/掩码替代 div/mod。"""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz
    out_off = out_pos.to(tl.int64) * 2

    idx_flat = offs
    r0 = idx_flat >> d1_log2
    r1 = idx_flat & d1_mask
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))
    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit(do_not_specialize=["d0", "d1", "d2"])
def nonzero_small_kernel_3d(
    inp,
    out,
    total,
    n_elements,
    d0,
    d1,
    d2,
    BLOCK_SIZE: tl.constexpr,
):
    """1-block 3D 特化: 标量 d0/d1/d2 替代 shape tensor。"""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz
    out_off = out_pos.to(tl.int64) * 3

    idx_flat = offs
    r2 = idx_flat % d2
    idx_flat //= d2
    r1 = idx_flat % d1
    idx_flat //= d1
    r0 = idx_flat
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))
    tl.store(out + out_off + 2, r2, mask=(nz == 1))
    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit(do_not_specialize=["d0", "d1", "inv_d1"])
def nonzero_small_kernel_2d_fp(
    inp,
    out,
    total,
    n_elements,
    d0,
    d1,
    inv_d1,
    BLOCK_SIZE: tl.constexpr,
):
    """1-block 2D 特化: d1 非 2 的幂, 用 f32 倒数近似除法(GCU 软件整数除法慢)。
    要求 n_elements < 2^24(f32 精确性), 带商/余数修正。"""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz
    out_off = out_pos.to(tl.int64) * 2

    idx_flat = offs
    q = (idx_flat.to(tl.float32) * inv_d1).to(tl.int32)
    r = idx_flat - q * d1
    q = tl.where(r < 0, q - 1, q)
    r = tl.where(r < 0, r + d1, r)
    q = tl.where(r >= d1, q + 1, q)
    r = tl.where(r >= d1, r - d1, r)
    tl.store(out + out_off + 0, q, mask=(nz == 1))
    tl.store(out + out_off + 1, r, mask=(nz == 1))
    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit(do_not_specialize=["d1_log2", "d1_mask", "d2_log2", "d2_mask"])
def nonzero_small_kernel_3d_pow2(
    inp,
    out,
    total,
    n_elements,
    d1_log2,
    d1_mask,
    d2_log2,
    d2_mask,
    BLOCK_SIZE: tl.constexpr,
):
    """1-block 3D 特化: d1/d2 均为 2 的幂, 移位/掩码替代 div/mod。"""
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz
    out_off = out_pos.to(tl.int64) * 3

    idx_flat = offs
    r2 = idx_flat & d2_mask
    idx_flat >>= d2_log2
    r1 = idx_flat & d1_mask
    idx_flat >>= d1_log2
    r0 = idx_flat
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))
    tl.store(out + out_off + 2, r2, mask=(nz == 1))
    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit
def nonzero_small_kernel(
    inp,
    out,
    total,
    n_elements,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    out_pos = local - nz

    idx_flat = offs
    out_off = out_pos.to(tl.int64) * ndim
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + out_off + dim, remainder, mask=(nz == 1))

    tl.store(total, tl.sum(nz))


@libentry()
@triton.jit
def nonzero_count_kernel(
    inp,
    inp_i32,
    counts,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    # 同时输出 int32 化的非零标记, 供 compact kernel 使用(避免 host 二次转换)
    tl.store(inp_i32 + offs, nz, mask=mask)
    cnt = tl.sum(nz)
    tl.store(counts + pid, cnt)


@libentry()
@triton.jit
def nonzero_block_offset_kernel(
    counts,
    offsets,
    n_blocks,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < n_blocks
    c = tl.load(counts + offs, mask=mask, other=0)
    p = tl.cumsum(c, axis=0)
    # 排他前缀和(每个 block 的输出起始位置)
    tl.store(offsets + offs, p - c, mask=mask)


@libentry()
@triton.jit
def nonzero_compact_kernel_1d(
    inp,
    offsets,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    # 排他局部前缀和: 第 k 个非零元素的输出行号
    out_pos = local - nz + base
    out_off = out_pos.to(tl.int64)
    tl.store(out + out_off, offs, mask=(nz == 1))


@libentry()
@triton.jit(do_not_specialize=["d0", "d1"])
def nonzero_compact_kernel_2d(
    inp,
    offsets,
    out,
    n_elements,
    d0,
    d1,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    out_pos = local - nz + base
    out_off = out_pos.to(tl.int64) * 2

    idx_flat = offs
    r1 = idx_flat % d1
    idx_flat //= d1
    r0 = idx_flat
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))


@libentry()
@triton.jit
def nonzero_compact_kernel_2d_pow2(
    inp,
    offsets,
    out,
    n_elements,
    d1_log2,
    d1_mask,
    BLOCK_SIZE: tl.constexpr,
):
    """2D compact 特化: 最后一维 d1 为 2 的幂, 用移位/掩码替代运行时 div/mod(GCU 无硬件除法器)。"""
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    out_pos = local - nz + base
    out_off = out_pos.to(tl.int64) * 2

    idx_flat = offs
    r0 = idx_flat >> d1_log2
    r1 = idx_flat & d1_mask
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))


@libentry()
@triton.jit(do_not_specialize=["d0", "d1", "d2"])
def nonzero_compact_kernel_3d(
    inp,
    offsets,
    out,
    n_elements,
    d0,
    d1,
    d2,
    BLOCK_SIZE: tl.constexpr,
):
    """3D compact 特化: 标量 d0/d1/d2 替代 shape tensor。"""
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    out_pos = local - nz + base
    out_off = out_pos.to(tl.int64) * 3

    idx_flat = offs
    r2 = idx_flat % d2
    idx_flat //= d2
    r1 = idx_flat % d1
    idx_flat //= d1
    r0 = idx_flat
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))
    tl.store(out + out_off + 2, r2, mask=(nz == 1))


@libentry()
@triton.jit(do_not_specialize=["d0", "d1", "inv_d1"])
def nonzero_compact_kernel_2d_fp(
    inp,
    offsets,
    out,
    n_elements,
    d0,
    d1,
    inv_d1,
    BLOCK_SIZE: tl.constexpr,
):
    """2D compact 特化: d1 非 2 的幂, 用 f32 倒数近似除法(GCU 软件整数除法慢)。
    要求 n_elements < 2^24(f32 精确性), 带商/余数修正。"""
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    out_pos = local - nz + base
    out_off = out_pos.to(tl.int64) * 2

    idx_flat = offs
    q = (idx_flat.to(tl.float32) * inv_d1).to(tl.int32)
    r = idx_flat - q * d1
    q = tl.where(r < 0, q - 1, q)
    r = tl.where(r < 0, r + d1, r)
    q = tl.where(r >= d1, q + 1, q)
    r = tl.where(r >= d1, r - d1, r)
    tl.store(out + out_off + 0, q, mask=(nz == 1))
    tl.store(out + out_off + 1, r, mask=(nz == 1))


@libentry()
@triton.jit(do_not_specialize=["d1_log2", "d1_mask", "d2_log2", "d2_mask"])
def nonzero_compact_kernel_3d_pow2(
    inp,
    offsets,
    out,
    n_elements,
    d1_log2,
    d1_mask,
    d2_log2,
    d2_mask,
    BLOCK_SIZE: tl.constexpr,
):
    """3D compact 特化: d1/d2 均为 2 的幂, 移位/掩码替代 div/mod。"""
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    out_pos = local - nz + base
    out_off = out_pos.to(tl.int64) * 3

    idx_flat = offs
    r2 = idx_flat & d2_mask
    idx_flat >>= d2_log2
    r1 = idx_flat & d1_mask
    idx_flat >>= d1_log2
    r0 = idx_flat
    tl.store(out + out_off + 0, r0, mask=(nz == 1))
    tl.store(out + out_off + 1, r1, mask=(nz == 1))
    tl.store(out + out_off + 2, r2, mask=(nz == 1))


@libentry()
@triton.jit
def nonzero_compact_kernel(
    inp,
    offsets,
    out,
    n_elements,
    shape,
    ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = ext.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp + offs, mask=mask, other=0)
    nz = (x != 0).to(tl.int32)
    local = tl.cumsum(nz, axis=0)
    base = tl.load(offsets + pid)
    # 排他局部前缀和: 第 k 个非零元素的输出行号
    out_pos = local - nz + base

    idx_flat = offs
    out_off = out_pos.to(tl.int64) * ndim
    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape + dim)
        remainder = idx_flat % dim_size
        idx_flat //= dim_size
        tl.store(out + out_off + dim, remainder, mask=(nz == 1))


def nonzero(inp, *, as_tuple=False):
    logger.debug("GEMS_ENFLAME NONZERO")

    inp_ndim = inp.ndim

    inp = inp.contiguous()
    n_elements = inp.numel()
    inp_view = inp.view(n_elements)

    if n_elements == 0:
        out = torch.empty((0, inp_ndim), dtype=torch.int64, device=inp.device)
        if as_tuple:
            # torch 语义: 每个维度一个索引张量
            return torch.unbind(out, dim=1)
        return out

    is_1d = inp_ndim == 1
    is_2d = inp_ndim == 2
    is_3d = inp_ndim == 3
    shape = (
        None
        if is_1d or is_2d or is_3d
        else torch.tensor(inp.shape, dtype=torch.int32, device=inp.device)
    )
    out = torch.empty(n_elements, inp_ndim, dtype=torch.int64, device=inp.device)

    with torch_device_fn.device(inp.device):
        if n_elements <= SINGLE_BLOCK_LIMIT:
            # small path: host 端转 int32(GCU 编译器对非 int32 输入 + cumsum 有缺陷)
            inp_bool = inp_view
            if inp_view.dtype != torch.bool:
                inp_bool = inp_view != 0
            inp_i32 = inp_bool.to(torch.int32)
            total = torch.empty(1, dtype=torch.int32, device=inp.device)
            block = max(MIN_BLOCK, triton.next_power_of_2(n_elements))
            if is_1d:
                nonzero_small_kernel_1d[(1,)](
                    inp_i32, out, total, n_elements, block, num_warps=4
                )
            elif is_2d:
                d1 = inp.shape[1]
                if (d1 & (d1 - 1)) == 0:
                    nonzero_small_kernel_2d_pow2[(1,)](
                        inp_i32,
                        out,
                        total,
                        n_elements,
                        d1.bit_length() - 1,
                        d1 - 1,
                        block,
                        num_warps=4,
                    )
                else:
                    nonzero_small_kernel_2d_fp[(1,)](
                        inp_i32,
                        out,
                        total,
                        n_elements,
                        inp.shape[0],
                        d1,
                        1.0 / d1,
                        block,
                        num_warps=4,
                    )
            elif is_3d:
                d1, d2 = inp.shape[1], inp.shape[2]
                if ((d1 & (d1 - 1)) == 0) and ((d2 & (d2 - 1)) == 0):
                    nonzero_small_kernel_3d_pow2[(1,)](
                        inp_i32,
                        out,
                        total,
                        n_elements,
                        d1.bit_length() - 1,
                        d1 - 1,
                        d2.bit_length() - 1,
                        d2 - 1,
                        block,
                        num_warps=4,
                    )
                else:
                    nonzero_small_kernel_3d[(1,)](
                        inp_i32,
                        out,
                        total,
                        n_elements,
                        inp.shape[0],
                        d1,
                        d2,
                        block,
                        num_warps=4,
                    )
            else:
                nonzero_small_kernel[(1,)](
                    inp_i32, out, total, n_elements, shape, inp_ndim, block, num_warps=4
                )
            num_nonzeros = int(total.item())
        else:
            # 按元素数自适应 BLOCK_SIZE / num_warps(实测 tune 结论)
            # 3D 坐标计算多, 小 shape 用更大 BLOCK 收益明显
            if is_3d and n_elements <= 1048576:
                block, num_warps = 8192, 2
            elif n_elements <= 262144:
                block, num_warps = 4096, 4
            elif n_elements <= 1048576:
                block, num_warps = 8192, 2
            else:
                block, num_warps = 16384, 4
            n_blocks = triton.cdiv(n_elements, block)
            inp_i32 = torch.empty(n_elements, dtype=torch.int32, device=inp.device)
            counts = torch.empty(n_blocks, dtype=torch.int32, device=inp.device)
            offsets = torch.empty(n_blocks, dtype=torch.int32, device=inp.device)
            grid = (n_blocks,)
            nonzero_count_kernel[grid](
                inp_view, inp_i32, counts, n_elements, block, num_warps=num_warps
            )
            nonzero_block_offset_kernel[(1,)](
                counts, offsets, n_blocks, triton.next_power_of_2(n_blocks), num_warps=1
            )
            if is_1d:
                nonzero_compact_kernel_1d[grid](
                    inp_i32, offsets, out, n_elements, block, num_warps=num_warps
                )
            elif is_2d:
                d1 = inp.shape[1]
                if (d1 & (d1 - 1)) == 0:
                    nonzero_compact_kernel_2d_pow2[grid](
                        inp_i32,
                        offsets,
                        out,
                        n_elements,
                        d1.bit_length() - 1,
                        d1 - 1,
                        block,
                        num_warps=num_warps,
                    )
                else:
                    if n_elements < (1 << 24):
                        # f32 倒数近似除法(仅当元素数 < 2^24 保证 f32 精确)
                        nonzero_compact_kernel_2d_fp[grid](
                            inp_i32,
                            offsets,
                            out,
                            n_elements,
                            inp.shape[0],
                            d1,
                            1.0 / d1,
                            block,
                            num_warps=num_warps,
                        )
                    else:
                        nonzero_compact_kernel_2d[grid](
                            inp_i32,
                            offsets,
                            out,
                            n_elements,
                            inp.shape[0],
                            d1,
                            block,
                            num_warps=num_warps,
                        )
            elif is_3d:
                d1, d2 = inp.shape[1], inp.shape[2]
                if ((d1 & (d1 - 1)) == 0) and ((d2 & (d2 - 1)) == 0):
                    nonzero_compact_kernel_3d_pow2[grid](
                        inp_i32,
                        offsets,
                        out,
                        n_elements,
                        d1.bit_length() - 1,
                        d1 - 1,
                        d2.bit_length() - 1,
                        d2 - 1,
                        block,
                        num_warps=num_warps,
                    )
                else:
                    nonzero_compact_kernel_3d[grid](
                        inp_i32,
                        offsets,
                        out,
                        n_elements,
                        inp.shape[0],
                        d1,
                        d2,
                        block,
                        num_warps=num_warps,
                    )
            else:
                nonzero_compact_kernel[grid](
                    inp_i32,
                    offsets,
                    out,
                    n_elements,
                    shape,
                    inp_ndim,
                    block,
                    num_warps=num_warps,
                )
            num_nonzeros = int((offsets[n_blocks - 1] + counts[n_blocks - 1]).item())

    out = out[0:num_nonzeros]

    if as_tuple:
        # torch 语义: 每个维度一个索引张量(out 为 (nnz, ndim))
        return torch.unbind(out, dim=1)
    else:
        return out
