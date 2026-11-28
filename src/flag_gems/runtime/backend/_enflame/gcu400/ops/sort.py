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
    FULL_TILE: tl.constexpr = False,
):
    # arr_ptr: (m, n)
    # out_ptr: per-CTA partial histogram (grid_n, m, num_passes, r)
    #
    # start_pass: first pass covered by this measurement. Every LSD radix pass
    # permutes the elements, so the per-bin counts inside a block change with the
    # permutation; the histogram must be recomputed per pass for the current
    # permutation rather than computed once and reused. num_passes=1 means the
    # single-pass histogram.
    #
    # Why no atomics: cur_m is walked by pid_m_base in steps of grid_y_m, so rows
    # are disjoint by construction and the writers of one (cur_m, p, bin) can only
    # come from the pid_n dimension. Each pid_n writes its own partial slot, and the
    # cross-CTA accumulation moves to reduce_sum (summing the partials inside sweep
    # instead of using atomic_add).
    #
    # FULL_TILE: when n is a multiple of CTA_TILE_N every tile is full, so the mask
    # comparison can be dropped entirely (the (TILE_R, TILE_N) mask broadcast plus &
    # is one of the most expensive scalar paths in hist). Measured on fp32
    # 1024x65536: single pass 2.72ms -> 1.78ms (tile=4096).
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
                # int32 counts suffice (count <= n < 2**30) and halve the DSM use
                acc = tl.zeros((TILE_R, TILE_N), dtype=tl.int32)
                if FULL_TILE:
                    n_end = cta_n_start + CTA_TILE_N
                else:
                    n_end = cta_n_end
                for n_start in range(cta_n_start, n_end, TILE_N):  # sequantial
                    n_offsets = n_start + tl.arange(0, TILE_N)  # (TILE_N, )
                    if FULL_TILE:
                        arr = tl.load(arr_ptr + cur_m * n + n_offsets)
                    else:
                        mask = n_offsets < cta_n_end
                        arr = tl.load(arr_ptr + cur_m * n + n_offsets, mask=mask)
                    arr = convert_to_uint_preverse_order(arr, descending)
                    key = (arr >> bit_offset) & bfe_mask  # (TILE_N, )
                    matches = bin_indices[:, None] == key
                    if not FULL_TILE:
                        matches = matches & mask[None, :]
                    acc += matches.to(tl.int32)
                local_sum = tl.sum(acc, axis=1)
                hist_offset = (pid_n * m + cur_m) * num_passes * r + p * r
                tl.store(out_ptr + hist_offset + bin_indices, local_sum)


@triton.jit
def block_base_kernel(
    hist_ptr, base_ptr, m, r: tl.constexpr, grid_n, GRID_P: tl.constexpr
):
    # hist_ptr: (grid_n, m, 1, r); when flattened, element (g, row, bin) sits at
    # (g * m + row) * r + bin
    # base_ptr: same shape; holds the global start offset of every (block, row, bin)
    # in the output:
    #           base[g, row, b] = sum_{b' < b} total(row, b')
    #                           + sum_{g' < g} hist[g', row, b]
    #
    # The cross-CTA reduction uses reduce_sum (summing the partial histograms per
    # block) instead of atomic_add; it is also vectorized to touch each row only
    # once: row_total reduces all bins of a row in one (r,) vector and the rows are
    # independent, so this parallelizes naturally. torch.cumsum is avoided because
    # flag_gems' cumsum launches with grid.z = K for M=1 and K > 255, which exceeds
    # the hardware limit (see Section 9 entry 31).
    #
    # Replacing the per-block scalar reads with a (CHUNK, r) 2D tile plus an axis=0
    # cumsum was tried in order to shorten the memory latency of the dependency
    # chain, and measured slower (fp32 1024x65536: 147.5ms -> 151.6ms overall); the
    # 2D cumsum costs more than the latency it saves, so the serial per-block form
    # stays.
    pid = tl.program_id(0)
    bin_all = tl.arange(0, r)

    for row in range(pid, m, GRID_P):
        # reduce_sum: the total of each bin in a row is the sum over all CTA blocks
        row_total = tl.zeros((r,), dtype=tl.int32)
        for g in range(0, grid_n):
            row_total += tl.load(hist_ptr + (g * m + row) * r + bin_all)
        bin_prefix = (
            tl.cumsum(row_total, axis=0) - row_total
        )  # exclusive prefix over bins

        # Exclusive prefix over CTA blocks (one (row, bin) accumulated along pid_n)
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
    block_base_ptr,  # (grid_n, m, 1, r): global start of each (CTA block, row, bin); None for SINGLE_BLOCK
    bit_offset,
    m,
    N,
    TILE_N: tl.constexpr,
    TILE_R: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
    SINGLE_BLOCK: tl.constexpr,
    FULL_TILE: tl.constexpr = False,
):
    # r: num_bins = 2 ** k_bits

    # arr_ptr: (m, N)
    # out_ptr: (m, N)
    # block_base_ptr: (grid_n, m, 1, r), precomputed by block_base_kernel

    # grid: (grid_n, grid_y_m, grid_r)
    #
    # The previous implementation's decoupled lookback made every CTA wait serially
    # along pid_n, which measured as the main bottleneck for large shapes, and the
    # atomic accumulation was capped by scalar atomics at 0.39 G elem/s. Both are
    # removed here:
    #   - the cross-CTA count reduction uses reduce_sum (partial histograms summed by
    #     block_base_kernel) instead of atomic_add
    #   - block_base is precomputed by block_base_kernel, so the kernel only does the
    #     in-block exclusive prefix and the scatter
    #   - for SINGLE_BLOCK (grid_n == 1) the in-block count is the global count, so the
    #     prefix is built while counting, no hist kernel is needed, and each pass
    #     costs a single launch

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
        # FULL_TILE: every CTA tile lies completely inside the row (N is a multiple
        # of TILE_N), so both the load mask and the where mask are always true and the
        # whole block is dropped. The wide-row N values (65536 / 131072 / 262144) all
        # satisfy this condition.
        if FULL_TILE:
            arr = tl.load(arr_ptr + cur_m * N + n_offsets)
        else:
            mask = n_offsets < N
            arr = tl.load(arr_ptr + cur_m * N + n_offsets, mask=mask)
        arr_u = convert_to_uint_preverse_order(arr, descending)
        key = (arr_u >> bit_offset) & bfe_mask  # (TILE_N, )
        if associate_arr_ptr is not None:
            if FULL_TILE:
                associate_arr = tl.load(associate_arr_ptr + cur_m * N + n_offsets)
            else:
                associate_arr = tl.load(
                    associate_arr_ptr + cur_m * N + n_offsets, mask=mask
                )

        if SINGLE_BLOCK:
            # The whole row belongs to this CTA: accumulating bins in increasing
            # order gives the exclusive prefix
            running_prefix = tl.zeros((), dtype=tl.int32)

        # since triton can only use scalar as condition, loop by bin_index
        for bin_index in range(cta_r_start, cta_r_end):
            if FULL_TILE:
                matches = key == bin_index
            else:
                matches = tl.where(mask, key == bin_index, False)  # (TILE_N, ) bool
            matches_i = matches.to(tl.int32)
            local_ex_cumsum = tl.cumsum(matches_i, axis=0) - matches_i  # (TILE_N, )

            # Global start offset of this (CTA block, row, bin) in the output
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
    arr_ptr,  # (m, N) initial input, also buffer A of the ping-pong
    idx_ptr,  # (m, N) initial indices, buffer A
    tmp_arr_ptr,  # (m, N) ping-pong buffer B
    tmp_idx_ptr,  # (m, N) ping-pong buffer B
    out_ptr,  # (m, N) final values
    out_idx_ptr,  # (m, N) final indices
    m,
    N,
    n_passes: tl.constexpr,
    k_bits: tl.constexpr,
    descending: tl.constexpr,
    TILE_N: tl.constexpr,
):
    # SINGLE_BLOCK only (grid_n == 1, one CTA owns the whole row): all n_passes
    # rounds of the LSD radix are fused into a single launch, removing the overhead
    # of 8 kernel launches.
    #
    # Two global buffers ping-pong per pass: src is read and dst is written by the
    # bin exclusive-prefix scatter, then they swap for the next pass. Everything
    # stays inside Triton kernels; the tmp buffers are internal scratch only (the
    # same amount as R6's arr_in/arr_out/indices_in/indices_out).
    # tl.debug_barrier() makes the stores of one pass visible to the loads of the
    # next pass within the same CTA.
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

            # The in-block count is the global count (one CTA owns the row), so
            # accumulating bins in increasing order gives the exclusive prefix
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


# Tiling parameters for multi-block radix. _SWEEP_BINS_PER_CTA determines how many
# bins each CTA handles during sweep: grid_r = num_bins / _SWEEP_BINS_PER_CTA, and bins are processed serially within
# each CTA.
_SWEEP_BINS_PER_CTA = 4
# Column-wise tile for sweep / hist. hist and sweep must share the same CTA block boundaries, otherwise
# the pid_n-to-block_base correspondence will be misaligned.
#
# Tile width is the biggest lever on the wide-row path: sweep runs one cumsum over
# the whole tile per bin, the sync steps of a cumsum scale with log2(tile) while the
# elements per tile scale with tile, so the per-element cumsum cost scales with
# log2(tile)/tile and widening the tile dilutes it directly; hist's (TILE_R, tile)
# 2D compare+accumulate behaves the same way. Measured single-pass cost (fp32,
# 1024x65536 / 16x131072):
#   hist 1024 + sweep 2048 : 5.01+9.99 = 15.00ms  / 0.209+0.391 = 0.600ms
#   hist 2048 + sweep 8192 : 2.81+7.26 = 10.07ms  / 0.090+0.262 = 0.352ms
#   hist 2048 + sweep 16384: 2.47+8.02 = 10.49ms  / 0.076+0.287 = 0.363ms
# i.e. 1.5~1.7x overall for the 16-pass wide rows (1024x65536 fp32: 257ms -> 169ms).
# Pushing sweep to 16384 regresses again because the per-CTA tile data grows, so
# 8192 is used.
_SWEEP_TILE_N = 8192
# The hist tile and the num_bins (TILE_R) dimension together decide the size of the
# compare tensor inside a CTA. With the mask, 2048 is optimal (at 8192 the (4, 8192)
# int32 accumulation tensor makes hist 4x slower); with full tiles (FULL_TILE, the
# wide rows are all multiples) and no mask, 4096 is optimal: 1024x65536 single pass
# 2.72ms(2048) -> 1.78ms(4096), 16x131072 0.088 -> 0.071ms.
_HIST_TILE_N = 4096
# For n <= this value (a single CTA can cover the row) the fused sort_small_kernel
# path is used; its tile must span the whole row, so it is decoupled from the
# multi-block tile above and keeps its historical value.
_SINGLE_BLOCK_TILE_N = 2048
# Radix bits per pass for the multi-block path. The cost of a sweep is num_bins times "full-row compare +
# cumsum + masked store", so work per pass ∝ 2**k and the number of passes = num_bits/k,
# total work ∝ 2**k * num_bits/k: k=4 → 128 units, k=3 → 88, k=2 → 64. On wide rows, measurements show
# k=2 is 1.35~1.50x faster than k=4 (int16 1024x65536: 103.8ms → 68.8ms), indicating this is
# store-bandwidth-bound rather than launch-bound, so use k=2. (The single-block fused path for n <= 2048 already
# uses k=2.)
_LARGE_ROW_BITS_PER_PASS = 2


def radix_sort(arr, k_bits=8, descending=False):
    n = arr.shape[-1]
    m = arr.numel() // n
    assert n < (1 << 30), "we have not implemented 2**30 per launch"
    dtype = arr.dtype
    num_bits = 1 if dtype == torch.bool else (arr.itemsize * 8)

    # hist and sweep must share the same CTA block boundary CTA_TILE_N so that the
    # block_base reduced by reduce_sum inside sweep maps one-to-one to sweep's pid_n.
    #
    # The single-block path (one CTA per row) is chosen from the row width alone and
    # is decoupled from the multi-block tile width: otherwise a wider multi-block
    # tile would pull rows with n in (2048, 8192] into the fused kernel and change
    # the behavior of those shapes (the fused kernel needs the tile to span the row).
    single_block = n <= _SINGLE_BLOCK_TILE_N
    TILE_N_HIST = _HIST_TILE_N if not single_block else _SINGLE_BLOCK_TILE_N
    TILE_N_SWEEP = _SINGLE_BLOCK_TILE_N if single_block else _SWEEP_TILE_N
    while triton.cdiv(n, TILE_N_SWEEP) > 65535:
        TILE_N_SWEEP *= 2
    CTA_TILE_N = TILE_N_SWEEP
    tiles_n_per_cta = max(1, CTA_TILE_N // TILE_N_HIST)

    num_bins = 2**k_bits
    n_passes = triton.cdiv(num_bits, k_bits)
    # The bin dimension of the hist kernel must match the real num_bins: with
    # k_bits=2 and a fixed 16, bin_indices would write (r=16) into a buffer that is
    # only r=4 wide, i.e. an out-of-bounds write.
    TILE_R_HIST = num_bins
    TILE_R_SWEEP = _SWEEP_BINS_PER_CTA

    grid_n = triton.cdiv(n, CTA_TILE_N)
    grid_y_m = min(m, 255)
    tile_r_sweep = TILE_R_SWEEP
    # When the row width is a multiple of the CTA tile, every hist / sweep CTA
    # processes a full tile, so the per-lane bound mask is always true and the whole
    # block can be dropped (see the FULL_TILE notes of both kernels).
    full_tile = n % CTA_TILE_N == 0

    with torch_device_fn.device(arr.device):
        # sort
        arr_in = torch.clone(arr)
        # NOTE: indices must be passed to the kernel as int32. On GCU400 an int64
        # Tensor argument only addresses correctly with
        # TORCH_GCU_ENABLE_INT64_AND_UINT64=1, otherwise the indices are scrambled
        # (measured idx_match_ratio of just 0.4%). n < 2**30 is guaranteed by the
        # assert above, so int32 is used on the wire and the result is converted back
        # to int64. This also halves the per-pass memory traffic of associate.
        indices_in = (
            torch.arange(0, n, dtype=torch.int32, device=arr_in.device)
            .broadcast_to(arr.shape)
            .contiguous()
        )
        arr_out = torch.empty_like(arr)
        indices_out = torch.empty_like(indices_in)

        if single_block:
            # One CTA per row: all n_passes rounds ping-pong inside a single kernel,
            # compressing 8 kernel launches into 1 (small shapes are launch-bound).
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

        # Multi-block path: each pass = hist (exclusive writes) + reduce_sum prefix +
        # sweep scatter
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
            # 1) Recompute the per-CTA histogram of one pass for the current
            #    permutation (each block writes exclusively, no atomic_add). It cannot
            #    be computed once for all passes and reused: the elements are permuted
            #    after every pass, so the per-bin counts inside a block change and
            #    reusing them would misplace block_base in later passes.
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
                full_tile,
            )
            # 2) reduce_sum reduces the partial histograms and builds the exclusive
            #    prefix, giving the global start of every (block, row, bin)
            #    (replacing atomic_add and decoupled lookback)
            block_base_kernel[(min(m, 1024),)](
                hist_blocks, block_base_buf, m, num_bins, grid_n, 1024
            )
            # 3) sweep: in-block exclusive prefix + scatter
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
                full_tile,
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
        # The per-pass cost of radix scales with num_bins (per-bin compare + cumsum)
        # and the total work scales with 2**k * num_bits/k, while the pass count
        # num_bits/k decides how often hist re-reads the whole array:
        #   - n <= 2048 takes the fused single_block kernel, where all passes finish
        #     in one launch and k=2 is a net win (measured 2.11x).
        #   - n > 2048 takes multi-block, where k comes from
        #     _LARGE_ROW_BITS_PER_PASS; on wide rows k=2 measures 1.35~1.50x faster
        #     than k=4.
        if dtype == torch.bool:
            num_bits_per_pass = 1
        elif sort_elem_cnt <= 2048:
            num_bits_per_pass = 2
        else:
            num_bits_per_pass = _LARGE_ROW_BITS_PER_PASS
        out, out_index = radix_sort(inp, num_bits_per_pass, descending)

    if dim != inp.ndim - 1:
        out = torch.movedim(out, -1, dim)
        out_index = torch.movedim(out_index, -1, dim)
    return out, out_index
