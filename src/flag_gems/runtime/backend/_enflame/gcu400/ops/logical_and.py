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
import os  # noqa: F401

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_INT32_MAX = torch.iinfo(torch.int32).max
_FAST_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32)

# GCU400 override of the generic pointwise implementation (logical_not /
# logical_xor family playbook, 2026-11-16).  The generic wrapper lowers the
# bool result through the i1 lane (~35% off the byte-stream floor) on top of
# the block-pointer tile / GSL structure tax; the flat grid-stride kernels
# below classify both operands with the branch-free sign trick
# `v = ((x | -x) & (y | -y)) >> (bits-1)) & 1` (x | -x has the sign bit set
# iff x != 0 for any two's-complement width; per-element, so no cross-lane
# borrow) and store the 0/1 result through an int8 view of the bool output.
# Bool tensors additionally get a packed 4:1 word lane: the borrow-free
# 0x7f trick classifies each byte, and (tx & ty) bit 7 per byte is exactly
# and-of-truthiness.  The traffic is bit-identical to logical_xor (2 reads
# + 1 write per element; only the single combining ALU op differs), so the
# xor-measured configs carry over: NS software pipelining on the stride
# loop (3 for int16, 2 for int32/byte lanes), BLOCK ladder int16 65536 /
# int32 32768 / packed 32768, grid 48, num_warps 2.  Measured on the same
# traffic (GCU400, official bench kernel mode, do_bench median w1000/r100,
# 2026-11-16 logical_xor probes):
#
#   dtype  arm                     2^24 elems        2^30 elems      floor(2r1w)
#   int16  flat s3 B65536 g48 nw2   37.8us (2.22TB/s) 1.951ms (2.75)  1.954ms
#   int32  flat s2 B32768 g48 nw2   64.0us (2.36TB/s) 3.786ms (2.55)  3.909ms
#   bool   packed B32768 g48 nw2    24.0us (2.09TB/s) 1.061ms (3.04)  1.062ms
#
# Swept and rejected on the identical traffic (xor campaign): cmp form
# `(x!=0)&(y!=0)` (+7-14%, i1 widening machinery), nw4 (memory-bound,
# sometimes much worse), grid 16/24/32/64/96 (48 stays best incl.
# exact-division grids), chunked per-CTA contiguous spans, bool 8:1 i64
# packing, pointwise CodeGenConfig tiers (only int32 large ties the flat
# arm).  Native int16/int32 large shapes run within 1-6% of the
# non-foldable 2r1w traffic floor, so the remaining gap there is
# memory-wall, not kernel overhead.  Exotic inputs (floats -- NaN/-0.0
# need IEEE semantics, int64, complex, strided, broadcast, > int32 range)
# fall back to the backend pointwise path unchanged.


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def logical_and_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_total,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
    NS: tl.constexpr,
):
    o8 = out_ptr.to(tl.pointer_type(tl.int8))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        if EVEN:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
            if x_ptr.dtype.element_ty == tl.int16:
                v = (((x | -x) & (y | -y)) >> 15) & 1
            elif x_ptr.dtype.element_ty == tl.int8:
                v = (((x | -x) & (y | -y)) >> 7) & 1
            elif x_ptr.dtype.element_ty == tl.uint8:
                v = (((x | (0 - x)) & (y | (0 - y))) >> 7) & 1
            else:
                v = (((x | -x) & (y | -y)) >> 31) & 1
            tl.store(o8 + off, v.to(tl.int8))
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
            if x_ptr.dtype.element_ty == tl.int16:
                v = (((x | -x) & (y | -y)) >> 15) & 1
            elif x_ptr.dtype.element_ty == tl.int8:
                v = (((x | -x) & (y | -y)) >> 7) & 1
            elif x_ptr.dtype.element_ty == tl.uint8:
                v = (((x | (0 - x)) & (y | (0 - y))) >> 7) & 1
            else:
                v = (((x | -x) & (y | -y)) >> 31) & 1
            tl.store(o8 + off, v.to(tl.int8), mask=mask)


# Bool fast lane: pack four bools per i32 word through in-kernel pointer
# bitcasts.  t = (w & 0x7f7f7f7f) + 0x7f7f7f7f | w sets bit 7 of every byte
# iff that byte is nonzero (the masked add cannot carry across bytes), so
# (tx & ty) bit 7 per byte is exactly and-of-truthiness and the & 0x01
# extraction yields valid 0/1 bool bytes.  NOTE: the 5-op `(~(w | -w) >> 7)`
# shortcut is INVALID on packed words -- the word-wide two's-complement
# borrow chain crosses byte boundaries and misclassifies zero bytes above a
# nonzero byte (value-probe precedent in logical_not).  Contiguous word
# loads only; non-divisible/misaligned sizes fall back to the byte-lane
# flat kernel above (measured tied with packed at mid/large).
@libentry()
@triton.jit(do_not_specialize=["N_words"])
def logical_and_bool_packed_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    N_words,
    BLOCK: tl.constexpr,
    EVEN: tl.constexpr,
):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    y32 = y_ptr.to(tl.pointer_type(tl.int32))
    o32 = out_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_words + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EVEN:
            wx = tl.load(x32 + off)
            wy = tl.load(y32 + off)
            tx = ((wx & 0x7F7F7F7F) + 0x7F7F7F7F) | wx
            ty = ((wy & 0x7F7F7F7F) + 0x7F7F7F7F) | wy
            v = ((tx & ty) >> 7) & 0x01010101
            tl.store(o32 + off, v)
        else:
            mask = off < N_words
            wx = tl.load(x32 + off, mask=mask, other=0)
            wy = tl.load(y32 + off, mask=mask, other=0)
            tx = ((wx & 0x7F7F7F7F) + 0x7F7F7F7F) | wx
            ty = ((wy & 0x7F7F7F7F) + 0x7F7F7F7F) | wy
            v = ((tx & ty) >> 7) & 0x01010101
            tl.store(o32 + off, v, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")])
@triton.jit
def logical_and_func(x, y):
    return x.to(tl.int1).logical_and(y.to(tl.int1))


def _flat_block(n_total, itemsize):
    # do_bench(median) ladder reswept for the same 2-read / 1-write traffic
    # (logical_xor, 2026-11-16): tiny shapes want the exact pow2 tile,
    # int16 peaks at BLOCK 65536, int32 at 32768, 1-byte lanes keep 32768
    # through the balance band and only gain 131072 past 2^28.
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 2)
    if n_total <= 65536:
        return 8192
    if itemsize == 4:
        return 32768 if n_total > 262144 else 8192
    if n_total <= (1 << 21):
        return 32768
    if itemsize == 2:
        return 65536
    if n_total > (1 << 28):
        return 131072
    return 65536


def _packed_block(n_words):
    if n_words <= 2048:
        return max(triton.next_power_of_2(n_words), 2)
    if n_words <= 16384:
        return 8192
    return 32768


def _flat_config(n_total, itemsize):
    block = _flat_block(n_total, itemsize)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    return (
        block,
        (min(triton.cdiv(n_total, block), NUM_SIPS * 2),),
        n_total % block == 0,
    )


def _packed_config(n_words):
    block = _packed_block(n_words)
    if triton.cdiv(n_words, block) * block > _INT32_MAX:
        return None
    return (
        block,
        (min(triton.cdiv(n_words, block), NUM_SIPS * 2),),
        n_words % block == 0,
    )


def _launch(kern, grid, args, **kwargs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (logical_not / logical_xor precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kern[grid](*args, **kwargs)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "logical_and flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _fast_pair(A, B):
    # NOTE: check the device *type*, not Tensor.is_cuda -- torch_gcu only
    # flips is_cuda on gcu tensors after the first Triton launch completes
    # its lazy patching, so an is_cuda gate would wrongly route the very
    # first op call in a process to the pointwise fallback.
    return (
        isinstance(A, torch.Tensor)
        and isinstance(B, torch.Tensor)
        and A.device.type in ("cuda", "gcu")
        and B.device.type == A.device.type
        and A.dtype == B.dtype
        and A.dtype in _FAST_DTYPES
        and A.is_contiguous()
        and B.is_contiguous()
        and A.shape == B.shape
        and A.numel() <= _INT32_MAX
    )


def logical_and(A, B):
    # With the global 64-bit switch always on, the int64 narrow-storage compatibility branch has no trigger condition
    # and converges to one kernel call.
    logger.debug("GEMS_ENFLAME LOGICAL_AND")
    if _fast_pair(A, B):
        out = torch.empty_like(A, dtype=torch.bool)
        n_total = out.numel()
        if n_total == 0:
            return out
        if (
            A.dtype == torch.bool
            and n_total % 4 == 0
            and A.data_ptr() % 4 == 0
            and B.data_ptr() % 4 == 0
            and out.data_ptr() % 4 == 0
        ):
            n_words = n_total // 4
            config = _packed_config(n_words)
            if config is not None:
                block, grid, even = config
                _launch(
                    logical_and_bool_packed_kernel,
                    grid,
                    (A, B, out, n_words),
                    BLOCK=block,
                    EVEN=even,
                    num_warps=2,
                )
                return out
        if A.dtype == torch.bool:
            x_view = A.view(torch.int8)
            y_view = B.view(torch.int8)
        else:
            x_view = A
            y_view = B
        config = _flat_config(n_total, x_view.element_size())
        if config is not None:
            block, grid, even = config
            _launch(
                logical_and_flat_kernel,
                grid,
                (x_view, y_view, out, n_total),
                BLOCK=block,
                EVEN=even,
                NS=3 if A.dtype == torch.int16 else 2,
                num_warps=2,
            )
            return out
    return logical_and_func(A, B)


@triton.jit(do_not_specialize=["N"])
def _logical_and_inplace_kernel(A, B, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    element_offsets = tl.arange(0, BLOCK)
    for tile_index in tl.range(pid, (N + BLOCK - 1) // BLOCK, num_programs):
        offset = tile_index * BLOCK + element_offsets
        mask = offset < N
        lhs = tl.load(A + offset, mask=mask, other=0)
        rhs = tl.load(B + offset, mask=mask, other=0)
        result = tl.where((lhs != 0) & (rhs != 0), 1, 0)
        tl.store(A + offset, result, mask=mask)


def logical_and_(A, B):
    from flag_gems.runtime import torch_device_fn

    logger.debug("GEMS_ENFLAME LOGICAL_AND_")
    if A.numel() == 0:
        return A
    # Broadcast to A's shape and expand into a contiguous tensor so the kernel can use 1D flat offsets.
    expanded_other = B.expand_as(A)
    if not expanded_other.is_contiguous():
        expanded_other = expanded_other.contiguous()
    target = A if A.is_contiguous() else A.contiguous()
    num_elements = target.numel()
    block_size = 8192
    grid_size = min(48, triton.cdiv(num_elements, block_size))
    with torch_device_fn.device(target.device):
        _logical_and_inplace_kernel[(grid_size,)](
            target,
            expanded_other,
            num_elements,
            BLOCK=block_size,
            num_warps=4,
        )
    if target is not A:
        A.copy_(target)
    return A
