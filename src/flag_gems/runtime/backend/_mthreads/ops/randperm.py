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

from flag_gems.ops.topk import argsort
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.random_utils import philox_backend_seed_offset

device_ = device
logger = logging.getLogger(__name__)

V3_RADIX_BLOCK = 1024
V3_RADIX_BITS = 4
V3_RADIX_BINS = 16

_MAX_INT16_VAL = tl.constexpr(torch.iinfo(torch.int16).max)
_PAD_KEY_VAL = tl.constexpr(0x7FFFFFFF)
_MAX_INT64_VAL = tl.constexpr(torch.iinfo(torch.int64).max)


@triton.jit
def _philox_u32(philox_seed, philox_offset, idx):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    idx_u32 = idx.to(tl.uint32)
    z = c0 * 0
    r0, _, _, _ = tl.philox(philox_seed, c0 + idx_u32, c1, z, z)
    return r0


# =====================================================================
# 路径 1: n <= 512 极轻量单 CTA Bitonic Argsort (针对 64, 100 等极小规模)
# =====================================================================
@libentry()
@triton.jit(do_not_specialize=["n_elements", "philox_seed", "philox_offset"])
def ultra_slim_bitonic_kernel(
    out_ptr,
    n_elements,
    philox_seed,
    philox_offset,
    BLOCK_SIZE: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_SIZE)
    mask = (cols < n_elements).to(tl.int1)

    raw = _philox_u32(philox_seed, philox_offset, cols)
    key = (raw & 0x7FFF).to(tl.int16)
    key = tl.where(mask, key, _MAX_INT16_VAL)

    vals = cols.to(tl.int32)
    _, sorted_indices = argsort(key, vals, 0, descending=False)
    tl.store(out_ptr + cols, sorted_indices.to(out_ptr.dtype.element_ty), mask=mask)


# =====================================================================
# 路径 2: 512 < n <= 10000 均匀分桶快速排序 (含 1024 / 2048 / 4096 / 10000)
# =====================================================================
@libentry()
@triton.jit(do_not_specialize=["n_elements", "philox_seed", "philox_offset"])
def bucket_scatter_kernel(
    bucket_keys_ptr,
    bucket_vals_ptr,
    bucket_counts_ptr,
    n_elements,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    idx = pid.to(tl.int64) * BLOCK + cols
    mask = (idx < n_elements.to(tl.int64)).to(tl.int1)

    raw = _philox_u32(philox_seed, philox_offset, idx)
    key = (raw & 0x7FFFFFFF).to(tl.int32)

    bucket_id = (key >> 27) & 15
    slot = tl.atomic_add(bucket_counts_ptr + bucket_id, 1, mask=mask)

    dest = bucket_id * 1024 + slot
    # 随机打散写入采用 .cg 绕过 L1，保护高频缓存行
    tl.store(bucket_keys_ptr + dest, key, mask=mask, cache_modifier=".cg")
    tl.store(bucket_vals_ptr + dest, idx.to(tl.int32), mask=mask, cache_modifier=".cg")


@libentry()
@triton.jit
def bucket_sort_compact_kernel(
    bucket_keys_ptr,
    bucket_vals_ptr,
    bucket_counts_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)

    all_counts = tl.load(bucket_counts_ptr + tl.arange(0, 16))
    is_prev = tl.arange(0, 16) < b
    out_base = tl.sum(tl.where(is_prev, all_counts, 0))

    my_count = tl.load(bucket_counts_ptr + b)

    cols = tl.arange(0, BLOCK)
    in_mask = (cols < my_count).to(tl.int1)
    key = tl.load(bucket_keys_ptr + b * BLOCK + cols, mask=in_mask, other=_PAD_KEY_VAL)
    val = tl.load(bucket_vals_ptr + b * BLOCK + cols, mask=in_mask, other=0)

    _, sorted_vals = argsort(key, val, 0, descending=False)

    out_mask = (cols < my_count).to(tl.int1)
    tl.store(out_ptr + out_base + cols, sorted_vals.to(out_ptr.dtype.element_ty), mask=out_mask)


def _launch_uniform_bucket_sort(out, n, generator):
    device = out.device
    num_buckets = 16
    bucket_cap = 1024

    total_buf_size = num_buckets * bucket_cap * 2 + num_buckets
    buf = torch.empty(total_buf_size, dtype=torch.int32, device=device)
    bucket_keys = buf[: num_buckets * bucket_cap]
    bucket_vals = buf[num_buckets * bucket_cap : num_buckets * bucket_cap * 2]
    bucket_counts = buf[num_buckets * bucket_cap * 2 :]
    bucket_counts.zero_()

    seed, offset = philox_backend_seed_offset(n, generator=generator)
    scatter_grid = (triton.cdiv(n, bucket_cap),)

    with torch_device_fn.device(device):
        bucket_scatter_kernel[scatter_grid](
            bucket_keys, bucket_vals, bucket_counts, n, seed, offset, bucket_cap, num_warps=8
        )
        bucket_sort_compact_kernel[(num_buckets,)](
            bucket_keys, bucket_vals, bucket_counts, out, bucket_cap, num_warps=8
        )


# =====================================================================
# 路径 3: 10000 < n <= 65536 快速 Radix 排序 (加入 .cg 写入优化)
# =====================================================================
@libentry()
@triton.jit(do_not_specialize=["n_elements", "bit_offset"])
def tile_digit_hist_kernel(key_ptr, hist_ptr, n_elements, bit_offset, NUM_BINS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = (offs < n_elements.to(tl.int64)).to(tl.int1)
    key = tl.load(key_ptr + offs, mask=mask, other=0).to(tl.uint32)
    digit = (key >> bit_offset) & (NUM_BINS - 1)
    hist_row = hist_ptr + pid.to(tl.int64) * NUM_BINS
    for b in range(NUM_BINS):
        cnt = tl.sum(((digit == b) & mask).to(tl.int32))
        tl.store(hist_row + b, cnt)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "philox_seed", "philox_offset"])
def fused_keygen_hist_kernel(key_ptr, hist_ptr, n_elements, philox_seed, philox_offset, NUM_BINS: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = (offs < n_elements.to(tl.int64)).to(tl.int1)

    raw = _philox_u32(philox_seed, philox_offset, offs)
    key = (raw & 0x7FFF).to(tl.int32)
    tl.store(key_ptr + offs, key, mask=mask)

    digit = key & (NUM_BINS - 1)
    hist_row = hist_ptr + pid.to(tl.int64) * NUM_BINS
    for b in range(NUM_BINS):
        cnt = tl.sum(((digit == b) & mask).to(tl.int32))
        tl.store(hist_row + b, cnt)


@libentry()
@triton.jit(do_not_specialize=["tiles"])
def scan_hist_le64k_kernel(tile_hist_ptr, global_off_ptr, tiles, MAX_TILES: tl.constexpr):
    t_idx = tl.arange(0, MAX_TILES)
    t_mask = t_idx < tiles

    running_bin_base = 0
    for b in range(16):
        ptrs = tile_hist_ptr + t_idx * 16 + b
        counts = tl.load(ptrs, mask=t_mask, other=0)
        inc = tl.cumsum(counts)
        exc = inc - counts
        total = tl.sum(counts)
        tl.store(global_off_ptr + t_idx * 16 + b, running_bin_base + exc, mask=t_mask)
        running_bin_base += total


@libentry()
@triton.jit(do_not_specialize=["n_elements", "bit_offset"])
def radix_scatter_kernel(
    key_in_ptr, val_in_ptr, key_out_ptr, val_out_ptr, global_off_ptr,
    n_elements, bit_offset, HAS_VAL_IN: tl.constexpr, NUM_BINS: tl.constexpr, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = (offs < n_elements.to(tl.int64)).to(tl.int1)

    key = tl.load(key_in_ptr + offs, mask=mask, other=0)
    if HAS_VAL_IN:
        val = tl.load(val_in_ptr + offs, mask=mask, other=0)
    else:
        val = offs.to(tl.int32)

    digit = (key.to(tl.uint32) >> bit_offset) & (NUM_BINS - 1)
    base = global_off_ptr + pid.to(tl.int64) * NUM_BINS

    for b in range(NUM_BINS):
        m = (digit == b) & mask
        rank = tl.cumsum(m.to(tl.int32)) - 1
        dest = tl.load(base + b) + rank.to(tl.int64)
        # .cg 写入优化：全局离散写回绕过 L1，显著降低 Cache Thrashing
        tl.store(key_out_ptr + dest, key, mask=m, cache_modifier=".cg")
        tl.store(val_out_ptr + dest, val.to(val_out_ptr.dtype.element_ty), mask=m, cache_modifier=".cg")


def _launch_fast_radix_mid(out, n, generator):
    bits = V3_RADIX_BITS
    num_bins = V3_RADIX_BINS
    passes = 4
    block = V3_RADIX_BLOCK
    tiles = triton.cdiv(n, block)
    device = out.device

    key_a = torch.empty(n, dtype=torch.int32, device=device)
    key_b = torch.empty(n, dtype=torch.int32, device=device)
    val_a = torch.empty(n, dtype=torch.int32, device=device)
    val_b = torch.empty(n, dtype=torch.int32, device=device)
    tile_hist = torch.empty((tiles, num_bins), dtype=torch.int32, device=device)
    global_off = torch.empty((tiles, num_bins), dtype=torch.int32, device=device)

    seed, offset = philox_backend_seed_offset(n, generator=generator)
    grid = (tiles,)

    with torch_device_fn.device(device):
        k_in, k_out = key_a, key_b

        for p in range(passes):
            bit_offset = p * bits
            if p == 0:
                fused_keygen_hist_kernel[grid](key_a, tile_hist, n, seed, offset, num_bins, block, num_warps=4)
            else:
                tile_digit_hist_kernel[grid](k_in, tile_hist, n, bit_offset, num_bins, block, num_warps=4)

            scan_hist_le64k_kernel[(1,)](tile_hist, global_off, tiles, MAX_TILES=64, num_warps=4)

            has_val = 1 if p > 0 else 0
            if p == 0:
                val_in = val_a
                val_out = val_b
            elif p == 1:
                val_in = val_b
                val_out = val_a
            elif p == 2:
                val_in = val_a
                val_out = val_b
            else:
                val_in = val_b
                val_out = out

            radix_scatter_kernel[grid](
                k_in, val_in, k_out, val_out, global_off,
                n, bit_offset, has_val, num_bins, block, num_warps=8
            )
            k_in, k_out = k_out, k_in


# =====================================================================
# 路径 4: 大 n (n > 65536) 跨块前缀和 Radix (加入 .cg 写入优化)
# =====================================================================
@libentry()
@triton.jit(do_not_specialize=["tiles"])
def scan_reduce_16_kernel(
    tile_hist_ptr,
    chunk_sums_ptr,
    tiles,
    CHUNK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    b = tl.program_id(1)

    offs = (c * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)).to(tl.int64)
    mask = offs < tiles.to(tl.int64)
    counts = tl.load(tile_hist_ptr + offs * 16 + b, mask=mask, other=0)
    c_sum = tl.sum(counts)
    tl.store(chunk_sums_ptr + c * 16 + b, c_sum)


@libentry()
@triton.jit(do_not_specialize=["num_chunks"])
def scan_spine_16_kernel(
    chunk_sums_ptr,
    chunk_bases_ptr,
    num_chunks,
    MAX_CHUNKS: tl.constexpr,
):
    c_idx = tl.arange(0, MAX_CHUNKS)
    c_mask = c_idx < num_chunks

    running_bin_base = 0
    for b in range(16):
        sums = tl.load(chunk_sums_ptr + c_idx * 16 + b, mask=c_mask, other=0)
        inc = tl.cumsum(sums)
        exc = inc - sums
        total = tl.sum(sums)

        tl.store(chunk_bases_ptr + c_idx * 16 + b, running_bin_base + exc, mask=c_mask)
        running_bin_base += total


@libentry()
@triton.jit(do_not_specialize=["tiles"])
def scan_apply_16_kernel(
    tile_hist_ptr,
    chunk_bases_ptr,
    global_off_ptr,
    tiles,
    CHUNK_SIZE: tl.constexpr,
):
    c = tl.program_id(0)
    b = tl.program_id(1)

    offs = (c * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)).to(tl.int64)
    mask = offs < tiles.to(tl.int64)
    counts = tl.load(tile_hist_ptr + offs * 16 + b, mask=mask, other=0)
    inc = tl.cumsum(counts)
    exc = inc - counts

    base = tl.load(chunk_bases_ptr + c * 16 + b)
    tl.store(global_off_ptr + offs * 16 + b, base + exc, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["n_elements", "philox_seed", "philox_offset"])
def gen_keys_kernel(key_ptr, n_elements, philox_seed, philox_offset, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = (offs < n_elements.to(tl.int64)).to(tl.int1)
    key = _philox_u32(philox_seed, philox_offset, offs)
    tl.store(key_ptr + offs, key.to(tl.int32), mask=mask)


def _launch_fast_radix_large(out, n, generator):
    bits = 4
    num_bins = 16
    passes = 8
    block = V3_RADIX_BLOCK
    tiles = triton.cdiv(n, block)
    device = out.device

    key_a = torch.empty(n, dtype=torch.int32, device=device)
    key_b = torch.empty(n, dtype=torch.int32, device=device)
    val_a = torch.empty(n, dtype=torch.int32, device=device)
    val_b = torch.empty(n, dtype=torch.int32, device=device)
    tile_hist = torch.empty((tiles, num_bins), dtype=torch.int32, device=device)
    global_off = torch.empty((tiles, num_bins), dtype=torch.int32, device=device)

    chunk_size = 2048
    num_chunks = triton.cdiv(tiles, chunk_size)
    chunk_sums = torch.empty((num_chunks, num_bins), dtype=torch.int32, device=device)
    chunk_bases = torch.empty((num_chunks, num_bins), dtype=torch.int32, device=device)

    seed, offset = philox_backend_seed_offset(n, generator=generator)
    grid = (tiles,)

    with torch_device_fn.device(device):
        gen_keys_kernel[grid](key_a, n, seed, offset, block, num_warps=4)
        k_in, k_out = key_a, key_b

        for p in range(passes):
            bit_offset = p * bits
            tile_digit_hist_kernel[grid](k_in, tile_hist, n, bit_offset, num_bins, block, num_warps=4)
            scan_reduce_16_kernel[(num_chunks, num_bins)](tile_hist, chunk_sums, tiles, CHUNK_SIZE=chunk_size, num_warps=4)
            scan_spine_16_kernel[(1,)](chunk_sums, chunk_bases, num_chunks, MAX_CHUNKS=128, num_warps=4)
            scan_apply_16_kernel[(num_chunks, num_bins)](tile_hist, chunk_bases, global_off, tiles, CHUNK_SIZE=chunk_size, num_warps=4)

            has_val = 1 if p > 0 else 0
            if p == 0:
                val_in = val_a
                val_out = val_b
            elif p % 2 == 1:
                val_in = val_b
                val_out = val_a if p < passes - 1 else out
            else:
                val_in = val_a
                val_out = val_b if p < passes - 1 else out

            radix_scatter_kernel[grid](
                k_in, val_in, k_out, val_out, global_off,
                n, bit_offset, has_val, num_bins, block, num_warps=8
            )
            k_in, k_out = k_out, k_in


# =====================================================================
# 主入口
# =====================================================================
def randperm(
    n,
    *,
    generator=None,
    out=None,
    dtype=torch.int64,
    layout=torch.strided,
    device=None,
    requires_grad=False,
    pin_memory=False,
):
    logger.debug("GEMS_MTHREADS RANDPERM")
    assert dtype == torch.int16 or dtype == torch.int32 or dtype == torch.int64
    assert n <= _MAX_INT64_VAL, "n exceeds maximum int64"

    if device is None:
        device = torch.device(device_.name)
    elif isinstance(device, str):
        device = torch.device(device)

    if out is None:
        out = torch.empty(n, dtype=dtype, device=device)

    # 1. 极小规模 (n <= 512): 单 CTA 全展开 bitonic argsort (64, 100 保持 4x~5x 极速)
    if n <= 512:
        block_size = triton.next_power_of_2(n)
        seed, offset = philox_backend_seed_offset(n, generator=generator)
        num_warps = 4
        with torch_device_fn.device(device):
            ultra_slim_bitonic_kernel[(1,)](
                out, n, seed, offset, BLOCK_SIZE=block_size, num_warps=num_warps,
            )
        return out

    # 2. 中小规模 (512 < n <= 10000): 
    # 将 1024 纳入桶排序，彻底消除 1024 单 block 展开指令臃肿的问题（预计从 0.78x 升至 1.1x+）
    if n <= 10000:
        _launch_uniform_bucket_sort(out, n, generator)
        return out

    # 3. 中等规模 (10000 < n <= 65536): 保持 v17 稳定 Radix + .cg 写缓存优化
    if n <= 65536:
        _launch_fast_radix_mid(out, n, generator)
        return out

    # 4. 大规模 (n > 65536): 保持 v17 稳定分块 Radix + .cg 写缓存优化
    if n < (1 << 31):
        _launch_fast_radix_large(out, n, generator)
        return out

    raise NotImplementedError("n >= 2**31 is not supported yet on MUSA.")