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

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

NUM_SIPS = 24
_INT32_MAX = torch.iinfo(torch.int32).max
_FAST_DTYPES = (torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32)
# Dtypes whose elements can share a 32-bit lane pairwise on the scalar path.
_PACKED_PAIR_DTYPES = (torch.int16,)


# GCU400 override of the generic pointwise_dynamic implementation.
#
# The generic path is a block-pointer kernel (grid capped at 48, i1 loads for
# bool, and -- for the scalar variants -- the Python scalar wrapped into a 0-dim
# device tensor whose stride-0 broadcast is materialized per tile).  Measured on
# the core bench shapes (L600) it costs bool 0.60-0.85x and mid-size shapes
# 0.82-0.93x vs torch native, and for the (bool tensor, int scalar) mix it even
# produces wrong values: torch promotes that mix to an int64 result, the
# generated kernel stores 8 bytes per element into the fake-int64 (4-byte-slot)
# output buffer, so readback is interleaved garbage.
#
# The flat grid-stride kernels below saturate the memory floor instead:
#   * tensor pairs: 2-read/1-write flat stream, bool handled through an int8
#     lane view (0/1 bytes AND identically, and the i1 lowering loses ~35%
#     bandwidth);
#   * scalar variants: the scalar rides as a kernel argument (no 0-dim tensor,
#     no broadcast materialization);
#   * (bool tensor, int scalar): torch promotes to int64; the int64 output is
#     a fake container whose value channel is one 32-bit word per element, so
#     the kernel bitcasts the output pointer to int32 and stores one word per
#     element (kthvalue precedent).
# Everything exotic (broadcasting, strided inputs, mixed tensor dtypes,
# out-of-int32-range sizes, int64 tensors) falls back to the generic kernels.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def bitwise_and_flat_kernel(
    x_ptr, y_ptr, out_ptr, N_total, BLOCK: tl.constexpr, HAS_TAIL: tl.constexpr
):
    # 2026-11-16 round-2 restore: HAS_TAIL dual arm.  The FULL arm drops the
    # per-element predicate and must pair with num_stages=1 (FULL+ns2 measured
    # +12% on int32 mid); the MASK arm keeps the pipelined production ns=2.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    if HAS_TAIL:
        for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
            off = block_id * BLOCK + arange
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
            tl.store(out_ptr + off, x & y, mask=mask)
    else:
        for block_id in tl.range(pid, num_blocks, num_pids, num_stages=1):
            off = block_id * BLOCK + arange
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
            tl.store(out_ptr + off, x & y)


@libentry()
@triton.jit(do_not_specialize=["N_total", "y"])
def bitwise_and_scalar_flat_kernel(x_ptr, out_ptr, y, N_total, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        tl.store(out_ptr + off, x & y.to(x_ptr.dtype.element_ty), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "y"])
def bitwise_and_bool_int_flat_kernel(
    x_ptr, out_ptr, y, N_total, BLOCK: tl.constexpr, HAS_TAIL: tl.constexpr
):
    # (bool tensor, int scalar) promotes to int64 and the result buffer holds one
    # 8-byte element per input, so write int64 directly (bitcasting to int32 here
    # would leave the high word of every element untouched -> garbage readback).
    out64 = out_ptr.to(tl.pointer_type(tl.int64))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
        off = block_id * BLOCK + arange
        if HAS_TAIL:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            v = (x & y.to(x_ptr.dtype.element_ty)).to(tl.int64)
            tl.store(out64 + off, v, mask=mask)
        else:
            x = tl.load(x_ptr + off)
            v = (x & y.to(x_ptr.dtype.element_ty)).to(tl.int64)
            tl.store(out64 + off, v)


# 2026-11-16 round-3 (bitwise_xor wide-lane precedent): 1-byte lanes widened 4x
# into int32 words.  AND is lane-width agnostic (bitwise, no carries), so four
# adjacent bool/int8/uint8 elements ride one int32 lane.  In the 2^24..2^28
# element band (tensor) / 2^20..2^28 (scalar) the byte-lane stream is
# element-rate bound; the widened word stream measured -3~-5% (tensor) and
# -14.6% (scalar 1R1W @2^24) on the xor twin.  Outside the band both widths hit
# the same DRAM floor, so the widened path is dispatched only inside the band.
# n must be word-divisible (n & 3 == 0) and 4-byte aligned, else per-element.
@libentry()
@triton.jit(do_not_specialize=["N_lanes"])
def bitwise_and_wide_kernel(
    x_ptr, y_ptr, out_ptr, N_lanes, BLOCK: tl.constexpr, HAS_TAIL: tl.constexpr
):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    y32 = y_ptr.to(tl.pointer_type(tl.int32))
    o32 = out_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_lanes + BLOCK - 1) // BLOCK
    if HAS_TAIL:
        for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
            off = block_id * BLOCK + arange
            mask = off < N_lanes
            x = tl.load(x32 + off, mask=mask)
            y = tl.load(y32 + off, mask=mask)
            tl.store(o32 + off, x & y, mask=mask)
    else:
        for block_id in tl.range(pid, num_blocks, num_pids, num_stages=1):
            off = block_id * BLOCK + arange
            x = tl.load(x32 + off)
            y = tl.load(y32 + off)
            tl.store(o32 + off, x & y)


@libentry()
@triton.jit(do_not_specialize=["N_lanes", "y"])
def bitwise_and_scalar_wide_kernel(
    x_ptr, out_ptr, y, N_lanes, BLOCK: tl.constexpr, HAS_TAIL: tl.constexpr
):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    o32 = out_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_lanes + BLOCK - 1) // BLOCK
    if HAS_TAIL:
        for block_id in tl.range(pid, num_blocks, num_pids, num_stages=2):
            off = block_id * BLOCK + arange
            mask = off < N_lanes
            x = tl.load(x32 + off, mask=mask)
            tl.store(o32 + off, x & y, mask=mask)
    else:
        for block_id in tl.range(pid, num_blocks, num_pids, num_stages=1):
            off = block_id * BLOCK + arange
            x = tl.load(x32 + off)
            tl.store(o32 + off, x & y)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_and_func(x, y):
    return x & y


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_and_func_scalar(x, y):
    return x & y


def _choose_block(n_total, itemsize):
    # do_bench(median, warmup=1000ms/rep=100ms) sweep on GCU400 (L600):
    # num_warps=2 beats 4 on every bucket; tiny shapes want the exact pow2
    # tile, mid shapes peak at 65536 lanes, and only >2^28-element 1/2-byte
    # streams gain from 131072 (num_stages and a 2-block unroll both lose).
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 2)
    if n_total <= 65536:
        return 8192
    if n_total <= 524288:
        return 32768
    if n_total > (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _lane_view(t):
    # Bool tensors ride the int8 lane: 0/1 bytes AND identically and the
    # generic i1 lowering is ~35% off the byte-stream memory floor.
    return t.view(torch.int8) if t.dtype == torch.bool else t


def _packed_lane_view(t):
    # Two adjacent 2-byte lanes ride one 32-bit lane.  Bitwise AND does not care
    # which half a bit lives in, and the packed scalar mask is duplicated into
    # both halves, so the result is endian-independent.
    return t.reshape(-1).view(torch.int32)


def _packed_scalar_value(dtype, scalar):
    value = _scalar_lane_value(dtype, scalar) & 0xFFFF
    packed = value | (value << 16)
    return packed - (1 << 32) if packed >= (1 << 31) else packed


def _launch_flat(kern, grid, args, block, num_warps=8, **constexprs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (rad2deg / atan2_ / sinh precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kern[grid](*args, num_warps=num_warps, BLOCK=block, **constexprs)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "bitwise_and flat kernel launch failed (attempt %d): %s",
                attempt + 1,
                exc,
            )


def _fast_tensor(t):
    return (
        isinstance(t, torch.Tensor)
        and t.is_cuda
        and t.is_contiguous()
        and t.dtype in _FAST_DTYPES
        and t.numel() <= _INT32_MAX
    )


def _flat_config(n_total, itemsize):
    # Scalar / mixed-path ladder (unchanged by the tensor round-2 binning).
    block = _choose_block(n_total, itemsize)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    # GCU400 (L600) sweep: 2-byte lanes already saturate the memory floor with one
    # CTA per SIP, and doubling the CTA count only adds a tail wave (16.7M int16:
    # 29.1us -> 27.3us).  1-byte and 4-byte lanes still need two CTA waves to
    # hide the tail, so keep NUM_SIPS * 2 there.
    grid_cap = NUM_SIPS if itemsize == 2 else NUM_SIPS * 2
    return block, (min(triton.cdiv(n_total, block), grid_cap),)


def _tensor_flat_config(n_total, itemsize):
    # 2026-11-16 round-2 restore: tensor paths bin by cdiv>=1024 x itemsize.
    #   * 1B lanes: FULL (mask-free) arm + nw2 + g48 at every size (bool 1Gi
    #     probe -6%; nw8 under FULL costs +14% on bool mid);
    #   * 2B/4B mid-small (cdiv<1024): FULL + nw2 (2B g48 / 4B g24) -- int16
    #     mid -6.1%, int32 mid -2.6% vs the masked arm;
    #   * 2B/4B large (cdiv>=1024, long grid-stride loops): the masked +
    #     pipelined (ns=2) arm wins -- mask-free costs +1.8% there -- so keep
    #     the production MASK+nw8+g24 config (int16 1Gi bitwise unchanged).
    # FULL must pair with ns=1; non-divisible sizes fall back to the MASK arm
    # via HAS_TAIL.
    block = _choose_block(n_total, itemsize)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    cdiv = triton.cdiv(n_total, block)
    has_tail = (n_total % block) != 0
    if itemsize == 1:
        return block, (min(cdiv, NUM_SIPS * 2),), 2, has_tail
    if cdiv < 1024:
        grid_cap = NUM_SIPS * 2 if itemsize == 2 else NUM_SIPS
        return block, (min(cdiv, grid_cap),), 2, has_tail
    return block, (min(cdiv, NUM_SIPS),), 8, True


# 2026-11-16 round-3 widened-lane band (bitwise_xor probe crossover): the int32
# word stream beats the byte-lane stream only inside these bands; outside both
# hit the same DRAM floor and the per-element ladder stays.
_WIDE_MAX = 1 << 28
_WIDE_TENSOR_MIN = 1 << 24
_WIDE_SCALAR_MIN = 1 << 20
_WIDE_TENSOR_CONFIG = (32768, 2)


def _tensor_wide_config(A, B, out):
    n_total = A.numel()
    if (
        A.element_size() != 1
        or not (_WIDE_TENSOR_MIN <= n_total <= _WIDE_MAX)
        or n_total & 3
        or A.data_ptr() & 3
        or B.data_ptr() & 3
        or out.data_ptr() & 3
    ):
        return None
    block, num_warps = _WIDE_TENSOR_CONFIG
    n_lanes = n_total >> 2
    has_tail = (n_lanes % block) != 0
    return (
        block,
        (min(triton.cdiv(n_lanes, block), NUM_SIPS * 2),),
        num_warps,
        n_lanes,
        has_tail,
    )


def _scalar_wide_config(tensor, out):
    n_total = tensor.numel()
    if (
        tensor.element_size() != 1
        or not (_WIDE_SCALAR_MIN <= n_total <= _WIDE_MAX)
        or n_total & 3
        or tensor.data_ptr() & 3
        or out.data_ptr() & 3
    ):
        return None
    n_lanes = n_total >> 2
    # AND-specific sweep (interleaved A/B, 2^24..2^27): 32768/nw2 beats the
    # xor-twin 8192/nw4 ladder by +12~21% everywhere in-band (nw4 ties nw2;
    # 16384 and 65536 tiles lose).  Sibling configs do not transfer verbatim.
    block, num_warps = 32768, 2
    has_tail = (n_lanes % block) != 0
    return (
        block,
        (min(triton.cdiv(n_lanes, block), NUM_SIPS * 2),),
        num_warps,
        n_lanes,
        has_tail,
    )


def _wide_scalar_word(dtype, scalar):
    # Replicate the wrapped lane byte 4x into one int32 word: AND-ing the word
    # ANDs each byte lane with the scalar byte, byte-for-byte identical to the
    # per-element path (bitwise_xor wide precedent).
    v = _scalar_lane_value(dtype, scalar) & 0xFF
    word = v | (v << 8) | (v << 16) | (v << 24)
    return word - (1 << 32) if word >= (1 << 31) else word


def _scalar_lane_value(dtype, scalar):
    # AND only ever looks at the scalar's low bits, so wrap it into the lane
    # dtype's range on the host (two's complement for signed lanes).
    if dtype == torch.bool:
        return 1 if scalar else 0
    bits = torch.iinfo(dtype).bits
    value = int(scalar) & ((1 << bits) - 1)
    if value >= (1 << (bits - 1)) and dtype.is_signed:
        value -= 1 << bits
    return value


def _is_python_scalar(v):
    return isinstance(v, bool) or isinstance(v, int)


def _and_scalar_impl(tensor, scalar, out=None):
    # Shared by bitwise_and_scalar / bitwise_and_scalar_tensor (AND commutes)
    # and by the in-place variant (out=tensor).  Returns None when the input
    # pair is not eligible for the flat path.
    if not _fast_tensor(tensor) or not _is_python_scalar(scalar):
        return None
    mixed = not isinstance(scalar, bool) and tensor.dtype == torch.bool
    out_dtype = torch.int64 if mixed else tensor.dtype
    if out is None:
        out = torch.empty_like(tensor, dtype=out_dtype)
    elif out.dtype != out_dtype:
        return None
    n_total = tensor.numel()
    if n_total == 0:
        return out
    if mixed:
        config = _flat_config(n_total, tensor.element_size())
        if config is None:
            return None
        block, grid = config
        # (bool, int) promotes to int64; the scalar only contributes its low bit
        # because every input value is 0 or 1.
        _launch_flat(
            bitwise_and_bool_int_flat_kernel,
            grid,
            (_lane_view(tensor), out, int(scalar) & 1, n_total),
            block,
            HAS_TAIL=(n_total % block) != 0,
        )
        return out

    n_lanes = n_total
    x_lanes = _lane_view(tensor)
    out_lanes = _lane_view(out)
    lane_itemsize = tensor.element_size()
    y_value = _scalar_lane_value(tensor.dtype, scalar)
    if lane_itemsize == 1:
        # 1-byte lanes pack 4x into int32 words inside the measured band
        # (bitwise_xor wide precedent: scalar 1R1W @2^24 -14.6%).
        wide = _scalar_wide_config(tensor, out)
        if wide is not None:
            block, grid, num_warps, n_lanes_w, has_tail = wide
            _launch_flat(
                bitwise_and_scalar_wide_kernel,
                grid,
                (
                    x_lanes,
                    out_lanes,
                    _wide_scalar_word(tensor.dtype, scalar),
                    n_lanes_w,
                ),
                block,
                num_warps,
                HAS_TAIL=has_tail,
            )
            return out
    if (
        tensor.dtype in _PACKED_PAIR_DTYPES
        and n_total % 2 == 0
        and tensor.data_ptr() % 4 == 0
        and out.data_ptr() % 4 == 0
    ):
        # 2-byte lanes pack into 4-byte ones: same bytes, half the trip count.
        # Mid-size int16 streams measure ~3% faster (16.7M: 28.2us -> 27.3us)
        # while large ones stay at the memory floor.
        x_lanes = _packed_lane_view(x_lanes)
        out_lanes = _packed_lane_view(out_lanes)
        y_value = _packed_scalar_value(tensor.dtype, scalar)
        n_lanes = n_total // 2
        lane_itemsize = 4
    config = _flat_config(n_lanes, lane_itemsize)
    if config is None:
        return None
    block, grid = config
    _launch_flat(
        bitwise_and_scalar_flat_kernel,
        grid,
        (x_lanes, out_lanes, y_value, n_lanes),
        block,
    )
    return out


def bitwise_and_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND")
    if (
        _fast_tensor(A)
        and _fast_tensor(B)
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.device == B.device
    ):
        out = torch.empty_like(A)
        n_total = out.numel()
        if n_total == 0:
            return out
        wide = _tensor_wide_config(A, B, out)
        if wide is not None:
            block, grid, num_warps, n_lanes, has_tail = wide
            _launch_flat(
                bitwise_and_wide_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(out), n_lanes),
                block,
                num_warps,
                HAS_TAIL=has_tail,
            )
            return out
        config = _tensor_flat_config(n_total, A.element_size())
        if config is not None:
            block, grid, num_warps, has_tail = config
            _launch_flat(
                bitwise_and_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(out), n_total),
                block,
                num_warps,
                HAS_TAIL=has_tail,
            )
            return out
    return bitwise_and_func(A, B)


def bitwise_and_tensor_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_")
    if (
        _fast_tensor(A)
        and _fast_tensor(B)
        and A.dtype == B.dtype
        and A.shape == B.shape
        and A.device == B.device
        and (A is B or A.untyped_storage().data_ptr() != B.untyped_storage().data_ptr())
    ):
        n_total = A.numel()
        if n_total == 0:
            return A
        wide = _tensor_wide_config(A, B, A)
        if wide is not None:
            block, grid, num_warps, n_lanes, has_tail = wide
            _launch_flat(
                bitwise_and_wide_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(A), n_lanes),
                block,
                num_warps,
                HAS_TAIL=has_tail,
            )
            return A
        config = _tensor_flat_config(n_total, A.element_size())
        if config is not None:
            block, grid, num_warps, has_tail = config
            _launch_flat(
                bitwise_and_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(A), n_total),
                block,
                num_warps,
                HAS_TAIL=has_tail,
            )
            return A
    return bitwise_and_func(A, B, out0=A)


def bitwise_and_scalar(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_SCALAR")
    out = _and_scalar_impl(A, B)
    if out is not None:
        return out
    return bitwise_and_func_scalar(A, B)


def bitwise_and_scalar_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_SCALAR_")
    if isinstance(A, torch.Tensor) and _is_python_scalar(B):
        if A.dtype == torch.bool and not isinstance(B, bool):
            # torch raises for the (bool, int) in-place promotion; the generic
            # pointwise path silently truncates instead, so keep aten parity
            # (bitwise_xor precedent).
            raise RuntimeError(
                "result type Long can't be cast to the desired output type Bool"
            )
        out = _and_scalar_impl(A, B, out=A)
        if out is not None:
            return A
    return bitwise_and_func_scalar(A, B, out0=A)


def bitwise_and_scalar_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_AND_SCALAR_TENSOR")
    # AND commutes, so (scalar, tensor) reuses the (tensor, scalar) path.
    if _is_python_scalar(A):
        out = _and_scalar_impl(B, A)
        if out is not None:
            return out
    return bitwise_and_func_scalar(B, A)
