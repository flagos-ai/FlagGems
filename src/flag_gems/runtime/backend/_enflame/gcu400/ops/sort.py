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

from flag_gems.ops.topk import _get_finfo_val, _get_iinfo_val, argsort
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from . import sort_mradix as _sort_mradix

logger = logging.getLogger(__name__)


def unwrap_if_constexpr(o):
    return o.value if isinstance(o, tl.constexpr) else o


@tl.constexpr
def get_int_t(num_bits: tl.constexpr, signed: tl.constexpr) -> tl.dtype:
    num_bits = unwrap_if_constexpr(num_bits)
    signed = unwrap_if_constexpr(signed)
    return tl.core.get_int_dtype(num_bits, signed)


@tl.constexpr
def one_zeros(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return 1 << (num_bits - 1)


@tl.constexpr
def zero_ones(num_bits: tl.constexpr) -> int:
    num_bits = unwrap_if_constexpr(num_bits)
    return (1 << (num_bits - 1)) - 1


@triton.jit
def uint_to_uint(x, descending: tl.constexpr = False):
    out = ~x if descending else x
    return out


@triton.jit
def int_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    udtype = get_int_t(num_bits, False)
    ux = tl.cast(x, udtype, bitcast=True)
    if descending:
        # 0111111....1
        bit_mask: tl.constexpr = zero_ones(num_bits)
        bit_mask_tensor = tl.full((), value=bit_mask, dtype=udtype)
        out = ux ^ bit_mask_tensor
    else:
        # 1000000...0
        sign_bit_mask: tl.constexpr = one_zeros(num_bits)
        sign_bit_mask_tensor = tl.full((), value=sign_bit_mask, dtype=udtype)
        out = ux ^ sign_bit_mask_tensor
    return out


@triton.jit
def floating_to_uint(x, descending: tl.constexpr = False):
    num_bits: tl.constexpr = x.dtype.primitive_bitwidth
    sdtype = get_int_t(num_bits, True)
    udtype = get_int_t(num_bits, False)
    sx = x.to(sdtype, bitcast=True)
    ux = x.to(udtype, bitcast=True)

    sign_bit_mask_v: tl.constexpr = one_zeros(num_bits)
    sign_bit_mask = tl.full((), value=sign_bit_mask_v, dtype=udtype)
    # mind the dtype, right_shift for signed is arithmetic right shift
    # Fix for triton 3.1 or else `sx >> rshift_bits` is promoted to int32
    rshift_bits = tl.full((), value=num_bits - 1, dtype=sdtype)
    mask = sign_bit_mask | (sx >> rshift_bits).to(udtype, bitcast=True)
    tl.static_assert(mask.dtype == udtype, "type mismatch")
    # 1000000000...0 for positive
    # 1111111111...1 for negative
    if descending:
        out = ux ^ (~mask)
    else:
        out = ux ^ mask
    return out.to(udtype, bitcast=True)


@triton.jit
def convert_to_uint_preverse_order(x: tl.tensor, descending: tl.constexpr = False):
    if x.dtype.is_floating():
        out = floating_to_uint(x, descending)
    elif x.dtype.is_int_signed():
        out = int_to_uint(x, descending)
    elif x.dtype.is_int_unsigned():
        out = uint_to_uint(x, descending)
    return out


@triton.jit
def compute_global_hist_kernel(
    arr_ptr,
    out_ptr,
    num_passes,
    m,
    n,
    tiles_n_per_cta,
    start_pass,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    num_bits_per_pass: tl.constexpr,
    descending: tl.constexpr,
):
    # arr_ptr: (m, n)
    # out_ptr: per-CTA partial 直方图 (grid_n, m, num_passes, r)
    #
    # start_pass: 本次统计的起始 pass 号。LSD radix 每个 pass 都会重排元素，
    # 块内各 bin 的计数随排列变化，因此直方图必须针对当前排列、逐 pass 重算，
    # 不能一次算完所有 pass 复用。num_passes=1 时即为单 pass 直方图。
    #
    # 无原子化说明：cur_m 由 pid_m_base 按 grid_y_m 步进遍历，行与行天然互斥，
    # 因此同一 (cur_m, p, bin) 的写者只可能来自 pid_n 维度。每个 pid_n 写自己的
    # partial 槽位即可，跨 CTA 的累加改用 reduce_sum（在 sweep 内对 partial 求和，
    # 替代 atomic_add）。
    pid_n = tl.program_id(0)
    pid_m_base = tl.program_id(1)
    grid_y_m = tl.num_programs(1)

    r: tl.constexpr = 2**num_bits_per_pass
    bfe_mask: tl.constexpr = (1 << num_bits_per_pass) - 1  # a.k.a. 2 ** k_bits - 1
    CTA_TILE_N: tl.constexpr = TILE_N * tiles_n_per_cta
    cta_n_start = CTA_TILE_N * pid_n
    cta_n_end = tl.minimum(cta_n_start + CTA_TILE_N, n)

    for cur_m in range(pid_m_base, m, grid_y_m):
        for p in range(0, num_passes):  # parallel
            bit_offset = (p + start_pass) * num_bits_per_pass
            for r_start in range(0, r, TILE_R):  # parallel
                bin_indices = r_start + tl.arange(0, TILE_R)
                # int32 计数即可（count <= n < 2**30），比 int64 省一半 DSM
                acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int32)
                for n_start in range(cta_n_start, cta_n_end, TILE_N):  # sequantial
                    n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                    mask = n_offsets < cta_n_end
                    arr = tl.load(arr_ptr + cur_m * n + n_offsets, mask=mask)
                    arr = convert_to_uint_preverse_order(arr, descending)
                    key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                    matches = (bin_indices[:, None] == key) & mask[None, :]
                    acc += matches.to(tl.int32)
                local_sum = tl.sum(acc, axis=1)
                hist_offset = (pid_n * m + cur_m) * num_passes * r + p * r
                tl.store(out_ptr + hist_offset + bin_indices, local_sum)


@triton.jit
def block_base_kernel(
    hist_ptr, base_ptr, m, r: tl.constexpr, grid_n, GRID_P: tl.constexpr
):
    # hist_ptr: (grid_n, m, 1, r) 展平后元素 (g, row, bin) 下标为 (g * m + row) * r + bin
    # base_ptr: 同 shape，写出每个 (块, 行, bin) 在输出中的全局起始位置：
    #           base[g, row, b] = sum_{b' < b} total(row, b')
    #                           + sum_{g' < g} hist[g', row, b]
    #
    # 跨 CTA 归约用 reduce_sum（对 partial 直方图按块求和）替代 atomic_add；
    # 与"每行只算一次"的向量化实现：row_total 以 (r,) 向量一次归约整行全部 bin，
    # 各行之间无依赖，天然并行。不使用 torch.cumsum（flag_gems 的 cumsum 在
    # M=1、K > 255 时会以 grid.z = K 启动而超硬件上限，见 Section 9 条目 31）。
    pid = tl.program_id(0)
    bin_all = tl.arange(0, r)

    for row in range(pid, m, GRID_P):
        # reduce_sum：整行每 bin 的总数 = 所有 CTA 块 partial 之和
        row_total = tl.zeros((r,), dtype=tl.int32)
        for g in range(0, grid_n):
            row_total += tl.load(hist_ptr + (g * m + row) * r + bin_all)
        bin_prefix = tl.cumsum(row_total, axis=0) - row_total  # 跨 bin 排他前缀

        # 跨 CTA 块的排他前缀（同一 (row, bin) 在 pid_n 方向上的累计）
        running = tl.zeros((r,), dtype=tl.int32)
        for g in range(0, grid_n):
            hist_row = tl.load(hist_ptr + (g * m + row) * r + bin_all)
            tl.store(base_ptr + (g * m + row) * r + bin_all, bin_prefix + running)
            running += hist_row


@triton.jit
def sweep(
    arr_ptr,
    associate_arr_ptr,  # inputs: (key & value)
    out_ptr,
    associate_out_ptr,  # outputs: (key & value)
    block_base_ptr,  # (grid_n, m, 1, r)：每 (CTA 块, 行, bin) 的全局起始位置；SINGLE_BLOCK 时为 None
    bit_offset,
    m,
    N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
):
    # r: num_bins = 2 ** k_bits

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # block_base_ptr: (grid_n, m, 1, r)，由 block_base_kernel 预计算

    # grid: (grid_n, grid_y_m, grid_r)
    #
    # 原实现的 decoupled lookback 会让所有 CTA 沿 pid_n 串行等待，实测是大 shape 的
    # 主要瓶颈；原子累加又被标量原子 0.39 G elem/s 卡死。本实现彻底去掉这两者：
    #   - 跨 CTA 计数归约用 reduce_sum（partial 直方图 + block_base_kernel 求和）
    #     替代 atomic_add
    #   - block_base 由 block_base_kernel 预计算，kernel 内只做块内排他前缀 + scatter
    #   - SINGLE_BLOCK（grid_n == 1）时块内计数即全局计数，边计数边排前缀，
    #     连 hist kernel 都不需要，每 pass 只剩一次 launch

    pid_n = tl.program_id(0)
    pid_m_base = tl.program_id(1)
    pid_r = tl.program_id(2)
    grid_y_m = tl.num_programs(1)

    bfe_mask: tl.constexpr = (1 << k_bits) - 1  # a.k.a. 2 ** k_bits - 1

    r: tl.constexpr = 2**k_bits
    cta_r_start = pid_r * TILE_R
    cta_r_end = tl.minimum(cta_r_start + TILE_R, r)

    # n_offsets only depend on pid_n, shared across m iterations
    n_offsets = pid_n * TILE_N + tl.arange(0, TILE_N)  # (TILE_N, )

    for cur_m in range(pid_m_base, m, grid_y_m):
        mask = n_offsets < N
        arr = tl.load(arr_ptr + cur_m * N + n_offsets, mask=mask)
        arr_u = convert_to_uint_preverse_order(arr, descending)
        key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )
        if associate_arr_ptr is not None:
            associate_arr = tl.load(
                associate_arr_ptr + cur_m * N + n_offsets, mask=mask
            )

        if SINGLE_BLOCK:
            # 整行只属于本 CTA：按 bin 递增顺序累计即可得到排他前缀
            running_prefix = tl.zeros((), dtype=tl.int32)

        # since triton can only use scalar as condition, loop by bin_index
        for bin_index in range(cta_r_start, cta_r_end):
            matches = tl.where(mask, key == bin_index, False)  # (TILE_N, ) bool
            matches_i = matches.to(tl.int32)
            local_ex_cumsum = tl.cumsum(matches_i, axis=0) - matches_i  # (TILE_N, )

            # 本 (CTA 块, 行, bin) 在输出中的全局起始位置
            if SINGLE_BLOCK:
                block_base = running_prefix
                running_prefix += tl.sum(matches_i, axis=0)
            else:
                block_base = tl.load(
                    block_base_ptr + (pid_n * m + cur_m) * r + bin_index
                )  # scalar
            pos = block_base + local_ex_cumsum  # (TILE_N, )

            # scatter
            tl.store(out_ptr + cur_m * N + pos, arr, mask=matches)
            if associate_arr_ptr is not None:
                tl.store(
                    associate_out_ptr + cur_m * N + pos, associate_arr, mask=matches
                )


@triton.jit
def sort_small_kernel(
    arr_ptr,  # (m, N) 初始输入，同时作为 ping-pong 的 buffer A
    idx_ptr,  # (m, N) 初始索引，buffer A
    tmp_arr_ptr,  # (m, N) ping-pong buffer B
    tmp_idx_ptr,  # (m, N) ping-pong buffer B
    out_ptr,  # (m, N) 最终结果
    out_idx_ptr,  # (m, N) 最终索引
    m,
    N,
    n_passes: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # SINGLE_BLOCK（grid_n == 1，整行由一个 CTA 独占）专用：把 LSD radix 的
    # n_passes 轮全部融合进一次 launch，消除 8 次 kernel 启动开销。
    #
    # 每 pass 用两个全局 buffer 做 ping-pong：src 读、dst 写（按 bin 排他前缀
    # scatter），下一个 pass 交换。整个过程仍是纯 Triton kernel，tmp buffer 只是
    # 算子内部的暂存（与 R6 的 arr_in/arr_out/indices_in/indices_out 同量）。
    # tl.debug_barrier() 保证同一 CTA 内上一 pass 的 store 对下一 pass 的 load 可见。
    r: tl.constexpr = 2**k_bits
    bfe_mask: tl.constexpr = (1 << k_bits) - 1
    offs = tl.arange(0, TILE_N)
    mask = offs < N

    for row in range(tl.program_id(0), m, tl.num_programs(0)):
        base = row * N
        for p in tl.static_range(0, n_passes):
            if p % 2 == 0:
                src = arr_ptr
                sidx = idx_ptr
                dst = tmp_arr_ptr
                didx = tmp_idx_ptr
            else:
                src = tmp_arr_ptr
                sidx = tmp_idx_ptr
                dst = arr_ptr
                didx = idx_ptr
            if p == n_passes - 1:
                dst = out_ptr
                didx = out_idx_ptr

            arr = tl.load(src + base + offs, mask=mask, other=0)
            idx = tl.load(sidx + base + offs, mask=mask, other=0)
            arr_u = convert_to_uint_preverse_order(arr, descending)
            key = (arr_u >> (p * k_bits)) & bfe_mask  # (TILE_N, )

            # 块内计数即全局计数（整行只有一个 CTA），按 bin 递增累计得到排他前缀
            running = tl.zeros((), dtype=tl.int32)
            for b in tl.static_range(0, r):
                matches = mask & (key == b)
                matches_i = matches.to(tl.int32)
                local_ex = tl.cumsum(matches_i, axis=0) - matches_i
                pos = running + local_ex
                tl.store(dst + base + pos, arr, mask=matches)
                tl.store(didx + base + pos, idx, mask=matches)
                running += tl.sum(matches_i, axis=0)
            tl.debug_barrier()


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    # hist 与 sweep 必须使用相同的 CTA 块边界 CTA_TILE_N，sweep 内 reduce_sum
    # 归约出的 block_base 才能与 sweep 的 pid_n 一一对应。
    TILE_N_HIST = 1024
    TILE_N_SWEEP = 2048
    while triton.cdiv(n, TILE_N_SWEEP) > 65535:
        TILE_N_SWEEP *= 2
    CTA_TILE_N = TILE_N_SWEEP
    tiles_n_per_cta = max(1, CTA_TILE_N // TILE_N_HIST)

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    # hist kernel 的 bin 维必须与实际 num_bins 一致：k_bits=2 时若仍用固定 16，
    # bin_indices 会写出 (r=16) 而 buffer 只有 r=4 宽，造成越界写。
    TILE_R_HIST = num_bins
    TILE_R_SWEEP = 8

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_y_m = min(m, 255)
    # grid_n == 1 时整行都在一个 CTA 内，走 sort_small_kernel 一次 launch 完成全部 pass
    single_block = grid_n == 1
    tile_r_sweep = TILE_R_SWEEP

    with torch_device_fn.device(arr.device):
        # sort
        arr_in = torch.clone(arr)
        # NOTE: indices 必须用 int32 传给 kernel。GCU400 上 kernel 参数为 int64 Tensor 时
        # 需要 TORCH_GCU_ENABLE_INT64_AND_UINT64=1 才能正确按 int64 寻址，否则索引错乱
        # (实测 idx_match_ratio 仅 0.4%)。n < 2**30 已由上方 assert 保证，故用 int32 传输，
        # 最终结果再转回 int64。同时也把 associate 的每 pass 访存流量减半。
        indices_in = (
            torch.arange(0, n, dtype=torch.int32, device=arr_in.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        if single_block:
            # 整行一个 CTA：一次 kernel 内 ping-pong 完成全部 n_passes 轮，
            # 把 8 次 kernel 启动压缩成 1 次（小 shape 由 launch 开销主导）。
            sorted_arr = torch.empty_like(arr)
            sorted_idx = torch.empty_like(indices_in)
            sort_small_kernel[(min(m, 1024),)](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                sorted_arr,
                sorted_idx,
                m,
                n,
                n_passes,
                k_bits,
                descending,
                TILE_N_SWEEP,
            )
            return sorted_arr, sorted_idx.to(torch.int64)

        # 多块路径：每 pass = hist(独占写) + reduce_sum 前缀 + sweep scatter
        hist_blocks = torch.empty(
            (grid_n, m, 1, num_bins), device=arr.device, dtype=torch.int32
        )
        block_base_buf = torch.empty(
            (grid_n, m, 1, num_bins), device=arr.device, dtype=torch.int32
        )
        grid_r = triton.cdiv(num_bins, TILE_R_SWEEP)
        grid_for_sweep = (grid_n, grid_y_m, grid_r)

        for i in range(0, n_passes):
            bit_offset = i * k_bits
            # 1) 用当前排列重算单 pass 的 per-CTA 直方图（每块独占写，无 atomic_add）。
            #    不能一次算完所有 pass 再复用：元素在每个 pass 后都会被重排，
            #    块内各 bin 的计数随之变化，复用会导致后续 pass 的 block_base 错位。
            compute_global_hist_kernel[(grid_n, grid_y_m, 1)](
                arr_in,
                hist_blocks,
                1,
                m,
                n,
                tiles_n_per_cta,
                i,
                TILE_N_HIST,
                TILE_R_HIST,
                k_bits,
                descending,
            )
            # 2) reduce_sum 归约 partial 直方图 + 排他前缀，得到每 (块, 行, bin)
            #    的全局起始位置（替代 atomic_add 与 decoupled lookback）
            block_base_kernel[(min(m, 1024),)](
                hist_blocks, block_base_buf, m, num_bins, grid_n, 1024
            )
            # 3) sweep：块内排他前缀 + scatter
            sweep[grid_for_sweep](
                arr_in,
                indices_in,
                arr_out,
                indices_out,
                block_base_buf,
                bit_offset,
                m,
                n,
                TILE_N_SWEEP,
                tile_r_sweep,
                k_bits,
                descending,
                False,
            )
            arr_in, arr_out = arr_out, arr_in
            indices_in, indices_out = indices_out, indices_in

    return arr_in, indices_in.to(torch.int64)


@libentry()
@triton.jit()
def sort_kernel(
    in_ptr,
    out_ptr,
    out_index_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DESCENDING: tl.constexpr,
    IS_FLOAT: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offset = tl.program_id(0) * N + cols
    in_ptr += offset
    out_ptr += offset
    out_index_ptr += offset

    if IS_FLOAT:
        mask_val = _get_finfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)
    else:
        mask_val = _get_iinfo_val(in_ptr.dtype.element_ty, return_max=not DESCENDING)
        in_val = tl.load(in_ptr, mask=mask, other=mask_val)

    index_val = tl.arange(0, BLOCK_SIZE)

    sorted_in_val, sorted_index_val = argsort(
        in_val, index_val, 0, descending=DESCENDING
    )
    tl.store(out_ptr, sorted_in_val, mask=mask)
    tl.store(out_index_ptr, sorted_index_val, mask=mask)


def sort(inp, dim=-1, descending=False):
    # We only implement stable radix sort here
    logger.debug("GEMS_ENFLAME SORT")
    return sort_stable(inp, stable=False, dim=dim, descending=descending)


def sort_stable(inp, *, stable, dim=-1, descending=False):
    logger.debug("GEMS_ENFLAME SORT_STABLE")
    # We only implement stable radix sort here
    _ = stable
    sort_elem_cnt = inp.shape[dim]
    if sort_elem_cnt == 1:
        return inp, torch.zeros_like(inp, dtype=torch.int64)

    if dim < 0:
        dim = dim + inp.ndim
    if dim != inp.ndim - 1:
        inp = torch.movedim(inp, dim, -1).contiguous()
    else:
        inp = inp.contiguous()

    dtype = inp.dtype
    # MRADIX path: one hw thread per row, LSD radix on the hardware unit, i.e.
    # one count scan + one scatter per digit instead of the per-bin
    # compare/cumsum sweeps of radix_sort. Same stable LSD contract, so it
    # produces the torch.sort(stable=True) reference order. Shapes/dtypes
    # outside its VDMEM envelope fall through to the generic radix below.
    if _sort_mradix.sort_mradix_supported(inp):
        out, out_index32 = _sort_mradix.sort_mradix(inp, inp.shape, descending)
        out_index = out_index32.to(torch.int64)
    else:
        # radix 每 pass 的成本 ∝ num_bins（逐 bin 比较 + cumsum），总计算量 ∝
        # 2**k * num_bits/k：k=4 → 128 单位，k=2 → 64 单位（计算减半），代价是
        # pass 数翻倍。
        #   - n <= 2048 走 single_block 融合 kernel：多 pass 全在 1 次 launch 内完成，
        #     计算减半是净收益（实测 2.11x）。
        #   - n > 2048 走 multi-block：pass 翻倍意味着 hist 每 pass 重读全数组、
        #     launch 翻倍，实测反而劣化 0.76x，故保持 k=4。
        if dtype == torch.bool:
            num_bits_per_pass = 1
        elif sort_elem_cnt <= 2048:
            num_bits_per_pass = 2
        else:
            # k=3（r=8, 11 passes）实测大 shape 劣化 0.38x：pass 数增加带来的
            # hist 全数组重读 + launch 增加，远超每 pass 计算减半的收益，已否决。
            num_bits_per_pass = 4
        out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
