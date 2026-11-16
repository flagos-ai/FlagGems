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

logger = logging.getLogger(__name__)


@triton.jit
def _load_f32_1d(ptr, mask, BF16: tl.constexpr):
    # bf16 strided/gather loads on XPU3 go through a vmerge float de-interleave
    # that mis-maps lanes; loading the raw int16 bits and rebuilding the fp32
    # value via (bits << 16) is bit-exact and avoids the buggy path.
    if BF16:
        raw = tl.load(ptr, mask=mask, other=0)
        return ((raw.to(tl.int32) & 0xFFFF) << 16).to(tl.float32, bitcast=True)
    return tl.load(ptr, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def _store_f32_1d(ptr, val, mask, BF16: tl.constexpr):
    # Round fp32 -> bf16 bits with round-to-nearest-even and store through an
    # int16 view (mixing an int16-view load with a bf16-typed store trips the
    # XPU3 vectorizer).
    if BF16:
        rb = val.to(tl.int32, bitcast=True)
        rounded = (rb + 0x7FFF + ((rb >> 16) & 1)) >> 16
        tl.store(ptr, rounded.to(tl.int16), mask=mask)
    else:
        tl.store(ptr, val.to(ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _adaptive_max_pool1d_flat_kernel(
    input,
    output,
    indices,
    total,
    L: tl.constexpr,
    OW: tl.constexpr,
    MAX_KW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    # One output element per lane. The adaptive window length is bounded by
    # MAX_KW (constexpr) so the tap scan is a static unroll -- that keeps the
    # kernel free of a runtime loop and lets the tile stay wide. Taps are loaded
    # unconditionally from clamped in-range addresses (compound masked loads are
    # slow and `other=` is unreliable on XPU3); window membership is settled
    # value-wise with tl.where.
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    store_mask = offs < total
    q = tl.where(store_mask, offs, 0)

    ow = q % OW
    nc = q // OW
    # Adaptive window: start = floor(o * L / OW), end = ceil((o + 1) * L / OW).
    iw_start = (ow * L) // OW
    iw_end = ((ow + 1) * L + OW - 1) // OW

    base = nc * L
    acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    idx = tl.full((BLOCK,), -1, dtype=tl.int32)
    # Keep the first tap that attains the maximum and the last NaN, matching
    # ATen's adaptive_max_pool1d.
    for kw in tl.static_range(MAX_KW):
        iw = iw_start + kw
        active = iw < iw_end
        safe = tl.minimum(iw, iw_end - 1)
        v = _load_f32_1d(input + base + safe, store_mask, BF16)
        is_new = active & ((v > acc) | (v != v) | (idx < 0))
        acc = tl.where(is_new, v, acc)
        idx = tl.where(is_new, safe, idx)

    _store_f32_1d(output + offs, acc, store_mask, BF16)
    tl.store(indices + offs, idx.to(tl.int64), mask=store_mask)


@libentry()
@triton.jit
def _adaptive_max_pool1d_loop_kernel(
    input,
    output,
    indices,
    total,
    kmax,
    L: tl.constexpr,
    OW: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16: tl.constexpr,
):
    # Large adaptive windows: the tap scan must stay a *runtime* loop, otherwise
    # a static unroll of 16+ taps makes the XPU compiler hang. A runtime loop
    # pins the tile to <=128 lanes; these windows only occur for tiny output
    # sizes (e.g. global pooling) so throughput does not matter here.
    pid = tl.program_id(0)
    offs = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int32)
    store_mask = offs < total
    q = tl.where(store_mask, offs, 0)

    ow = q % OW
    nc = q // OW
    iw_start = (ow * L) // OW
    iw_end = ((ow + 1) * L + OW - 1) // OW

    base = nc * L
    acc = tl.full((BLOCK,), float("-inf"), dtype=tl.float32)
    idx = tl.full((BLOCK,), -1, dtype=tl.int32)
    for kw in range(0, kmax):
        iw = iw_start + kw
        active = iw < iw_end
        safe = tl.minimum(iw, iw_end - 1)
        v = _load_f32_1d(input + base + safe, store_mask, BF16)
        is_new = active & ((v > acc) | (v != v) | (idx < 0))
        acc = tl.where(is_new, v, acc)
        idx = tl.where(is_new, safe, idx)

    _store_f32_1d(output + offs, acc, store_mask, BF16)
    tl.store(indices + offs, idx.to(tl.int64), mask=store_mask)


def _parse_output_size(output_size):
    if isinstance(output_size, int):
        return output_size
    if isinstance(output_size, (list, tuple)) and len(output_size) == 1:
        return output_size[0]
    raise ValueError(f"Invalid output_size for adaptive_max_pool1d: {output_size}")


def _max_window(in_size: int, out_size: int) -> int:
    # Exact longest adaptive window instead of the loose cdiv(in, out) + 1 bound:
    # every saved tap is a saved load per output lane.
    longest = 1
    for o in range(out_size):
        start = (o * in_size) // out_size
        end = -((-(o + 1) * in_size) // out_size)
        longest = max(longest, end - start)
    return longest


def _pick_block(numel: int, max_kw: int) -> int:
    # Wide tiles are only safe while the static-unrolled tap set stays small
    # (each tap is a live BLOCK-sized fp32 vector). Benchmark windows are 2 taps
    # so they get the full 1024-lane tile; wider windows step down.
    if max_kw <= 2:
        block = 1024
    elif max_kw <= 4:
        block = 512
    else:
        block = 256
    if numel < block:
        # Round the tile up to the next power of two, but never below 64 lanes.
        block = min(1024 if max_kw <= 2 else block, 1 << (numel - 1).bit_length())
        block = max(64, block)
    return block


# Static unroll is fast but the XPU compiler hangs for large tap counts; beyond
# this many taps switch to the runtime-loop kernel (capped at 128 lanes).
STATIC_TAP_LIMIT = 8
LOOP_BLOCK = 128


def adaptive_max_pool1d(input: torch.Tensor, output_size):
    """Adaptive max pooling over the innermost 1D axis.

    The generic implementation delegates to ``adaptive_max_pool2d``, which on
    XPU compiles to a 2D tiled kernel with runtime nested loops: it is stuck at
    a 128-lane tile, does per-output gather at the gapped-access cliff, and
    fails to compile for bf16. This override computes the 1D reduction directly
    with a flat per-lane kernel and a static-unrolled adaptive window.
    """
    logger.debug("GEMS_KUNLUNXIN ADAPTIVE_MAX_POOL1D")

    assert input.ndim == 3, f"adaptive_max_pool1d expects 3D input, got {input.ndim}D"

    out_size = _parse_output_size(output_size)

    input = input.contiguous()
    n, c, lin = input.shape
    output = torch.empty((n, c, out_size), dtype=input.dtype, device=input.device)
    indices = torch.empty((n, c, out_size), dtype=torch.int64, device=input.device)

    if output.numel() == 0 or lin == 0:
        return output, indices

    numel = output.numel()
    max_kw = _max_window(lin, out_size)

    is_bf16 = input.dtype == torch.bfloat16
    flat_input = input.view(torch.int16) if is_bf16 else input
    flat_output = output.view(torch.int16) if is_bf16 else output

    with torch_device_fn.device(input.device):
        if max_kw <= STATIC_TAP_LIMIT:
            block = _pick_block(numel, max_kw)
            grid = (triton.cdiv(numel, block),)
            _adaptive_max_pool1d_flat_kernel[grid](
                flat_input,
                flat_output,
                indices,
                numel,
                L=lin,
                OW=out_size,
                MAX_KW=max_kw,
                BLOCK=block,
                BF16=is_bf16,
                num_warps=max(1, block // 256),
            )
        else:
            if numel >= LOOP_BLOCK:
                block = LOOP_BLOCK
            else:
                block = max(64, min(LOOP_BLOCK, 1 << (numel - 1).bit_length()))
            grid = (triton.cdiv(numel, block),)
            _adaptive_max_pool1d_loop_kernel[grid](
                flat_input,
                flat_output,
                indices,
                numel,
                max_kw,
                L=lin,
                OW=out_size,
                BLOCK=block,
                BF16=is_bf16,
                num_warps=1,
            )

    return output, indices
