# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA stable K2048 radix TopK; other descriptors use the existing kernel."""

import logging
import os

import torch
import triton
import triton.language as tl
from flag_gems.fused.indexer_topk_key import topk_stable_fp32_key
from flag_gems.ops.topk import topk as _generic_topk
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import HAS_TLE

logger = logging.getLogger(__name__)
_TOPK_RADIX_STABLE_K2048_ENABLED = os.getenv("FLAGGEMS_TOPK_K2048_TLE_V2", "0") == "1"
# Preserve the qualified reduction/radix order rather than retuning a migration.
_TOPK_RADIX_STABLE_K2048_CONFIG = (1024, 4, 8)

if HAS_TLE:
    import triton.experimental.tle.language as tle_gpu

    @libentry()
    @triton.jit
    def topk_kernel_radix_stable_k2048_tle(
        X,
        Yv,
        Yi,
        stride_xm,
        stride_xn,
        stride_ym,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        RADIX_BITS: tl.constexpr,
    ):
        # Input domain: 2D fp32, finite values and +/-inf (no NaN), K=2048. Order:
        # value descending, then original column ascending for numeric ties.
        tl.static_assert(N >= 4096)
        tl.static_assert(N <= 34880)
        tl.static_assert(RADIX_BITS == 4 or RADIX_BITS == 8)
        K: tl.constexpr = 2048
        RADIX_SIZE: tl.constexpr = 1 << RADIX_BITS
        RADIX_MASK: tl.constexpr = RADIX_SIZE - 1
        pid = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        bins = tl.arange(0, RADIX_SIZE)
        desired = tl.full((), 0, tl.uint32)
        desired_mask = tl.full((), 0, tl.uint32)
        k_to_find = tl.full((), K, tl.int32)

        # Find the exact Kth-largest numeric value key. Built-in masked
        # histograms avoid one shared atomic reservation per equal-valued item.
        for digit_pos in tl.static_range(32 - RADIX_BITS, -1, -RADIX_BITS):
            counts = tl.zeros([RADIX_SIZE], tl.int32)
            for start in tl.range(0, N, BLOCK_N):
                offsets = start + cols
                valid = offsets < N
                x = tl.load(
                    X + pid * stride_xm + offsets * stride_xn,
                    mask=valid,
                    other=float("-inf"),
                )
                key = topk_stable_fp32_key(x)
                matches = (key & desired_mask) == desired
                digit = ((key >> digit_pos) & RADIX_MASK).to(tl.int32)
                counts += tl.histogram(digit, RADIX_SIZE, valid & matches)

            cumulative = tl.cumsum(counts, axis=0, reverse=True)
            greater = cumulative - counts
            selected = (cumulative >= k_to_find) & (greater < k_to_find)
            selected_digit = tl.sum(tl.where(selected, bins, 0), axis=0)
            count_gt = tl.sum(tl.where(selected, greater, 0), axis=0)
            desired |= selected_digit.to(tl.uint32) << digit_pos
            desired_mask |= tl.full((), RADIX_MASK, tl.uint32) << digit_pos
            k_to_find -= count_gt

        # k_to_find is now exactly K - count(key > desired). Take all greater
        # keys and only the first k_to_find threshold ties in column order.
        # Prefix scans create unique deterministic slots even for an all--inf
        # row. There are exactly K selected items; no padding slots are sorted.
        selected_buf = tle_gpu.gpu.alloc(
            [K],
            dtype=tl.uint64,
            layout=None,
            scope=tle_gpu.gpu.smem,
            nv_mma_shared_layout=False,
        )
        equal_seen = tl.full((), 0, tl.int32)
        output_seen = tl.full((), 0, tl.int32)
        for start in tl.range(0, N, BLOCK_N):
            offsets = start + cols
            valid = offsets < N
            x = tl.load(
                X + pid * stride_xm + offsets * stride_xn,
                mask=valid,
                other=float("-inf"),
            )
            key = topk_stable_fp32_key(x)
            equal = valid & (key == desired)
            equal_rank = equal_seen + tl.cumsum(equal.to(tl.int32), axis=0)
            take = valid & ((key > desired) | (equal & (equal_rank <= k_to_find)))
            positions = output_seen + tl.cumsum(take.to(tl.int32), axis=0) - 1
            # Masked-off local pointers also stay in bounds.
            safe_positions = tl.where(take, positions, 0)
            pointers = tle_gpu.gpu.local_ptr(selected_buf, (safe_positions,))
            # N <= 34880 fits the low 16 bits. Larger packed keys sort first;
            # inverse column makes lower original columns sort first on ties.
            packed = (key.to(tl.uint64) << 16) | (N - offsets).to(tl.uint64)
            tl.store(pointers, packed, mask=take)
            equal_seen += tl.sum(equal.to(tl.int32), axis=0)
            output_seen += tl.sum(take.to(tl.int32), axis=0)

        # Required ordering between cross-thread shared scatter and gather.
        tl.debug_barrier()
        out_cols = tl.arange(0, K)
        selected_ptrs = tle_gpu.gpu.local_ptr(selected_buf, (out_cols,))
        packed_sorted = tl.sort(tl.load(selected_ptrs), dim=0, descending=True)
        out_indices = N - (packed_sorted & 0xFFFF).to(tl.int32)
        out_values = tl.load(X + pid * stride_xm + out_indices * stride_xn)
        tl.store(Yv + pid * stride_ym + out_cols, out_values)
        tl.store(Yi + pid * stride_ym + out_cols, out_indices.to(tl.int64))


def topk(x, k, dim=-1, largest=True, sorted=True):
    logger.debug("GEMS_NVIDIA TOPK")
    if not (
        _TOPK_RADIX_STABLE_K2048_ENABLED
        and HAS_TLE
        and isinstance(x, torch.Tensor)
        and x.ndim == 2
        and dim in (-1, 1)
        and largest
        and sorted
        and x.is_cuda
        and x.dtype == torch.float32
        and k == 2048
        and 1 <= x.shape[0] <= 1024
        and 4096 <= x.shape[1] <= 34880
        and x.stride(0) >= 0
        and x.stride(1) > 0
        and (x.shape[0] - 1) * x.stride(0)
        + (triton.cdiv(x.shape[1], 1024) * 1024 - 1) * x.stride(1)
        <= 0x7FFFFFFF
    ):
        return _generic_topk(x, k, dim=dim, largest=largest, sorted=sorted)
    values = torch.empty((x.shape[0], k), device=x.device, dtype=x.dtype)
    indices = torch.empty((x.shape[0], k), device=x.device, dtype=torch.int64)
    block_n, radix_bits, num_warps = _TOPK_RADIX_STABLE_K2048_CONFIG
    with torch_device_fn.device(x.device):
        topk_kernel_radix_stable_k2048_tle[(x.shape[0],)](
            x,
            values,
            indices,
            x.stride(0),
            x.stride(1),
            values.stride(0),
            N=x.shape[1],
            BLOCK_N=block_n,
            RADIX_BITS=radix_bits,
            num_warps=num_warps,
            num_stages=1,
        )
    return values, indices


__all__ = ["topk"]
