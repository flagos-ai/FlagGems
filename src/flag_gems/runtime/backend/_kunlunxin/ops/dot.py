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
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.libentry import libentry

logger = logging.getLogger(__name__)

# XPU reduction safety: tl.sum is only numerically reliable up to 8192 lanes
# (verified with fp64 ground truth); masked (tail) loads are unreliable in
# general, so tails are handled with exact power-of-two unmasked tiles that
# never read out of bounds.
SMALL_BLOCK = 512  # single reliable tile for the trivial case
WIDE_BLOCK = 8192   # reliable tl.sum width for bulk reduction
SMALL_N = 16384      # N <= this uses the historical 512-lane tree
# Batched bulk reduction: one program reduces DOT_UNROLL x WIDE_BLOCK lanes
# (multiple independent 8192-lane tl.sum, accumulated in fp32). Measured on
# XPU6: 16M -15/-21/-17% (fp16/fp32/bf16), 2^28 -4~-6%, 655M -3~-5% vs the
# single-tile kernel, with fp64-verified results; UNROLL=32 regresses
# (register pressure), so 8 is the retained sweet spot.
DOT_UNROLL = 8


@libentry()
@triton.jit
def dot_kernel(x_ptr, y_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr, tl.sum(x * y))


@libentry()
@triton.jit
def dot_kernel_1(x_ptr, y_ptr, mid_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


@libentry()
@triton.jit
def dot_sum_kernel(in_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = ext.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(in_ptr + offsets, mask=offsets < N, other=0.0)
    tl.store(out_ptr + pid, tl.sum(values))


@libentry()
@triton.jit
def dot_kernel_2(mid_ptr, out_ptr, M, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mask = offset < M
    mid_val = tl.load(mid_ptr + offset, mask=mask, other=0.0)
    tl.store(out_ptr, tl.sum(mid_val))


# Unmasked bulk kernels: one 8192-lane load+reduce per program.
@libentry()
@triton.jit
def dot_kernel_wide(x_ptr, y_ptr, mid_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


# Batched bulk kernel: one program reduces DOT_UNROLL independent 8192-lane
# unmasked tiles, storing ONE mid slot per tile (mid layout identical to the
# unbatched kernel, so the reduction tree below is unchanged).
@libentry()
@triton.jit
def dot_kernel_wide_batch(x_ptr, y_ptr, mid_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = ext.program_id(0)
    base_off = pid * (BLOCK * UNROLL)
    for j in tl.static_range(UNROLL):
        offs = base_off + j * BLOCK + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs).to(tl.float32)
        y = tl.load(y_ptr + offs).to(tl.float32)
        tl.store(mid_ptr + pid * UNROLL + j, tl.sum(x * y))


@libentry()
@triton.jit
def dot_kernel_512(x_ptr, y_ptr, mid_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(mid_ptr + pid, tl.sum(x * y))


# Exact power-of-two unmasked tile: never reads past the end, never masks.
@libentry()
@triton.jit
def dot_kernel_tile(x_ptr, y_ptr, out_ptr, TILE: tl.constexpr):
    offs = tl.arange(0, TILE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = tl.load(y_ptr + offs).to(tl.float32)
    tl.store(out_ptr, tl.sum(x * y))


@libentry()
@triton.jit
def sum_kernel_wide(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(v))


@libentry()
@triton.jit
def sum_kernel_wide_batch(in_ptr, out_ptr, BLOCK: tl.constexpr, UNROLL: tl.constexpr):
    pid = ext.program_id(0)
    base_off = pid * (BLOCK * UNROLL)
    for j in tl.static_range(UNROLL):
        offs = base_off + j * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(in_ptr + offs).to(tl.float32)
        tl.store(out_ptr + pid * UNROLL + j, tl.sum(v))


@libentry()
@triton.jit
def sum_kernel_512(in_ptr, out_ptr, BLOCK: tl.constexpr):
    pid = ext.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(v))


@libentry()
@triton.jit
def sum_kernel_tile(in_ptr, out_ptr, TILE: tl.constexpr):
    offs = tl.arange(0, TILE)
    v = tl.load(in_ptr + offs).to(tl.float32)
    tl.store(out_ptr, tl.sum(v))


def _pow2_decomp(v):
    """Exact power-of-two aligned slices covering [0, v) with no overlap."""
    out = []
    off = 0
    while v:
        size = 1 << (v.bit_length() - 1)
        out.append((off, size))
        off += size
        v -= size
    return out


def _dot_large(x, y, out, N):
    """Mask-free reduction: 8192 bulk + 512 tail + exact pow2 residue, tree."""
    full, rem = divmod(N, WIDE_BLOCK)
    t512, tiny = divmod(rem, SMALL_BLOCK)
    t_tiles = _pow2_decomp(tiny)
    count = full + t512 + len(t_tiles)
    mid = torch.empty((count,), dtype=torch.float32, device=x.device)
    if full:
        b, r = divmod(full, DOT_UNROLL)
        if b:
            dot_kernel_wide_batch[(b,)](x, y, mid, WIDE_BLOCK, DOT_UNROLL)
        if r:
            dot_kernel_wide[(r,)](
                x[b * WIDE_BLOCK * DOT_UNROLL:],
                y[b * WIDE_BLOCK * DOT_UNROLL:],
                mid[b * DOT_UNROLL:], WIDE_BLOCK)
    if t512:
        dot_kernel_512[(t512,)](x[full * WIDE_BLOCK:], y[full * WIDE_BLOCK:],
                                mid[full:], SMALL_BLOCK)
    base = full + t512
    for j, (off, size) in enumerate(t_tiles):
        dot_kernel_tile[(1,)](
            x[full * WIDE_BLOCK + t512 * SMALL_BLOCK + off:],
            y[full * WIDE_BLOCK + t512 * SMALL_BLOCK + off:],
            mid[base + j:],
            size,
        )
    mid_size = count
    while mid_size > SMALL_BLOCK:
        full2, rem2 = divmod(mid_size, WIDE_BLOCK)
        t512b, tiny2 = divmod(rem2, SMALL_BLOCK)
        t2 = _pow2_decomp(tiny2)
        cnt2 = full2 + t512b + len(t2)
        nm = torch.empty((cnt2,), dtype=torch.float32, device=x.device)
        if full2:
            b2, r2 = divmod(full2, DOT_UNROLL)
            if b2:
                sum_kernel_wide_batch[(b2,)](mid, nm, WIDE_BLOCK, DOT_UNROLL)
            if r2:
                sum_kernel_wide[(r2,)](mid[b2 * WIDE_BLOCK * DOT_UNROLL:],
                                       nm[b2 * DOT_UNROLL:], WIDE_BLOCK)
        if t512b:
            sum_kernel_512[(t512b,)](mid[full2 * WIDE_BLOCK:], nm[full2:],
                                     SMALL_BLOCK)
        b2 = full2 + t512b
        for j, (off, size) in enumerate(t2):
            sum_kernel_tile[(1,)](mid[full2 * WIDE_BLOCK + t512b * SMALL_BLOCK + off:],
                                  nm[b2 + j:], size)
        mid = nm
        mid_size = cnt2
    acc = torch.zeros((), dtype=torch.float32, device=x.device)
    for (off, size) in _pow2_decomp(mid_size):
        t = torch.empty((), dtype=torch.float32, device=x.device)
        sum_kernel_tile[(1,)](mid[off:], t, size)
        acc += t
    out.copy_(acc)


def _dot_small(x, y, out, N):
    """Historical 512-lane masked-tree path (N <= SMALL_N)."""
    if N <= SMALL_BLOCK:
        block_size = triton.next_power_of_2(N) if N else 1
        dot_kernel[(1,)](x, y, out, N, block_size)
        return
    mid_size = triton.cdiv(N, SMALL_BLOCK)
    mid = torch.empty((mid_size,), dtype=torch.float32, device=x.device)
    dot_kernel_1[(mid_size,)](x, y, mid, N, SMALL_BLOCK)
    while mid_size > SMALL_BLOCK:
        next_size = triton.cdiv(mid_size, SMALL_BLOCK)
        next_mid = torch.empty((next_size,), dtype=torch.float32, device=x.device)
        dot_sum_kernel[(next_size,)](mid, next_mid, mid_size, SMALL_BLOCK)
        mid = next_mid
        mid_size = next_size
    dot_kernel_2[(1,)](mid, out, mid_size, triton.next_power_of_2(mid_size))


def dot(x, y):
    logger.debug("GEMS_KUNLUNXIN DOT")

    assert x.shape == y.shape, "Input vectors must have the same shape"
    assert x.dim() == 1, "Input must be 1D tensors"

    N = x.shape[0]
    out = torch.empty([], dtype=x.dtype, device=x.device)

    with torch_device_fn.device(x.device):
        if N == 0:
            out.zero_()
        elif N <= SMALL_N:
            _dot_small(x, y, out, N)
        else:
            _dot_large(x, y, out, N)

    return out