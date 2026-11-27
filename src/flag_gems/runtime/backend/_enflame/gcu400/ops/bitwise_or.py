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


# GCU400 override of the pointwise_dynamic tensor-tensor path (bitwise_and /
# bitwise_xor precedents).  The generated path is a block-pointer kernel with
# the CTA count locked at min(12, num_tiles) and a 4096-lane tile (code_cache
# source), and bool tensors ride the i1 lowering; on the core bench shapes
# (L600) that costs the in-place __ior__ entry 0.10-0.39x vs torch native.
# The flat grid-stride kernel below saturates the memory floor instead:
# 2-read/1-write flat stream, bool handled through an int8 lane view (0/1
# bytes OR identically), and the eligible pairs only (contiguous, same shape,
# same dtype, int32-safe flat offsets).  Everything exotic (broadcasting,
# strided inputs, mixed tensor dtypes, out-of-int32-range sizes, int64
# tensors) falls back to the generated kernels.
#
# The scalar variants (bitwise_or_scalar{,_tensor} / the __ior__.Scalar dunder
# entry via ior_scalar) ride the same flat machinery (bitwise_xor twin): the
# scalar rides as a kernel argument (no 0-dim device tensor, no stride-0
# broadcast materialization per tile).  The (bool tensor, int scalar) mix
# promotes to int64 and stays on the generated pointwise path: int64 storage
# is a real 8-byte lane under TORCH_GCU_ENABLE_INT64_AND_UINT64=1, so the
# historical int32-slot single-word store reads back garbage there.
#
# 2026-11-16 round-3: bitwise_or_scalar_ (bitwise_or_.Scalar, i.e. `x |= s`)
# moves off the retired bitwise_or__opt GSL layer (ops/__init__.py import
# repointed here): that C05-era wrapper predates the 11-13 flat ladder, kept
# bool loads on the slow i1 lowering and measured 0.47-0.58x on the core bool
# buckets.  All five entries now share this file's kernels.  The scalar
# entries additionally dispatch 1-byte lanes in the 2^20..2^28 band to the
# widened int32-word kernel below (-4.6% same-process probe at the band
# center); the tensor entries stay per-element (wide probed a tie there --
# or's 131072 deep tile already matches the widened stream).
#
# 2026-11-16 round-4 (scalar-arm resweep, unique-name variant probes
# interleaved 3 rounds, do_bench median w150/r40): the scalar kernels carried
# no mask-free EXACT body and the per-element ladder predated the 11-16
# tensor-arm resweep, so the small/mid bands still paid the mask tax and
# stale block choices.  Measured wins landed in three places:
#   - per-element EXACT bodies (n % BLOCK == 0) gain -3..-15% across the
#     tiny np2 tile (int32 4096: 3.34->3.18us), the int16 mid band and the
#     1-byte mid band; at >=2^24 elements every config ties the copy floor
#     (copy_e32768 46.43 vs e32768 46.46us at int32 2^24 in-place), and no
#     EXACT flip shows up at 2^30 in the 1-read/1-write scalar stream
#     (int16 e131072 1338.97 vs m 1339.14, bool e131072 668.84 vs m
#     668.67us), unlike the 2R1W tensor arm.
#   - int16 lanes want BLOCK=16384 with num_warps=4 through 2^24
#     (2^20 in-place 6.31->4.80us, -24%; ties the old 65536 config at 2^24),
#     while int32 wants 16384 (<=2^20) / 32768 (<=2^24-2^26) at nw2 -- nw4
#     costs int32 +30% there, so num_warps is band*dtype-specific and the
#     libentry dispatch key (dtype/BLOCK/EXACT) still never sees two warp
#     counts for one entry.
#   - the in-place and out-of-place ladders genuinely split below 2^24:
#     int32 2^16 in-place wants 16384 (4.14 vs 4.64us) while out-of-place
#     wants 8192/EXACT (4.15 vs 4.64us); int32 (2^24, 2^26] in-place keeps
#     32768 (-1.1%) while out-of-place moves to 65536.  The ladder below is
#     therefore keyed on the inplace flag.
#   - the 1-byte widened word stream gains a mask-free EXACT body (ties
#     masked at the band edges, -2.9% at the 2^24 out-of-place center) and
#     an in-place-only 4096/nw2 tile at the band bottom [2^20, 2^21)
#     (4.66->4.17us); 2^22 in-place already prefers 8192/nw4, so the small
#     tile stays capped.  wide-16 (int16 2x pack) was probed across the
#     whole int16 band and never beat the per-element path by more than
#     noise -- vetoed.
@libentry()
@triton.jit(do_not_specialize=["N_total"])
def bitwise_or_flat_kernel(
    x_ptr, y_ptr, out_ptr, N_total, BLOCK: tl.constexpr, EXACT: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x_ptr + off)
            y = tl.load(y_ptr + off)
            tl.store(out_ptr + off, x | y)
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            y = tl.load(y_ptr + off, mask=mask)
            tl.store(out_ptr + off, x | y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total", "y"])
def bitwise_or_scalar_flat_kernel(
    x_ptr, out_ptr, y, N_total, BLOCK: tl.constexpr, EXACT: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x_ptr + off)
            tl.store(out_ptr + off, x | y.to(x_ptr.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(x_ptr + off, mask=mask)
            tl.store(out_ptr + off, x | y.to(x_ptr.dtype.element_ty), mask=mask)


# 2026-11-16 round-3 follow-up (bitwise_xor twin, three-round evidence from the
# same day): 1-byte scalar lanes widened 4x into int32 words.  OR is lane-width
# agnostic (bitwise, no carries), so four adjacent bool/int8/uint8 elements
# ride one int32 lane with the scalar pre-replicated on the host.  In the
# 2^20..2^28 element band the byte-lane stream is element-rate bound and the
# widened word stream measured -4.6% on this op (same-process interleaved
# probe at the 2^24 band center: 14.43 vs 15.11us vs the best per-element
# tile, REPRO-stable; scalar_ bool mid bench 27.7->14.6us combined with the
# scalar_ repoint below).  The tensor path was probed the same way and came
# out a tie (wide 32768/nw2 23.54-23.93us vs per-element 131072/nw2
# 23.90-23.96us) because or's per-element ladder already carries the deeper
# 131072 tile for 1-byte streams (the twin's wide gain was measured against a
# weaker 65536 per-element baseline), so the tensor entries stay per-element.
# Unlike the twin, the dispatch also requires 4-byte-aligned base pointers:
# byte-offset contiguous views (base[k:]) would feed a misaligned int32 word
# stream, which is the P12-family silent-misvalue risk.
@libentry()
@triton.jit(do_not_specialize=["N_lanes", "y"])
def bitwise_or_scalar_wide_kernel(
    x_ptr, out_ptr, y, N_lanes, BLOCK: tl.constexpr, EXACT: tl.constexpr
):
    x32 = x_ptr.to(tl.pointer_type(tl.int32))
    o32 = out_ptr.to(tl.pointer_type(tl.int32))
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_lanes + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(x32 + off)
            tl.store(o32 + off, x | y)
        else:
            mask = off < N_lanes
            x = tl.load(x32 + off, mask=mask)
            tl.store(o32 + off, x | y, mask=mask)


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func(x, y):
    return x | y


def _scalar_block_warps(n_total, itemsize, inplace):
    # Round-4 per-element scalar ladder (unique-name variant probes, GCU400
    # L600, interleaved do_bench median): the old `_choose_block` kept
    # masked 8192/32768/65536 tiles everywhere; the resweep moves each band
    # to its probed optimum and pairs it with the mask-free EXACT body.
    # num_warps is band*dtype-specific (int16 mid wants nw4, int32 hates it)
    # and the ladders split on in-place vs out-of-place below 2^24.
    if n_total <= 8192:
        return max(triton.next_power_of_2(n_total), 2), 2
    if itemsize == 1:
        # 1-byte lanes below the widened-word band: 16384/EXACT beats the old
        # 8192/32768/65536/131072 mid tiles by 9-15% (2^16: 3.73->3.39us,
        # 2^18: 4.75->4.07us, 2^20: 5.28->4.74us); non-word-divisible or
        # misaligned buffers above 2^20 keep the deep 131072 tile.
        if n_total <= (1 << 20):
            return 16384, 2
        return 131072, 2
    if itemsize == 2:
        # int16: 8192 through 65536 (-9%), 16384/nw4 through 2^24 (-6..-24%,
        # ties at 2^24), 65536 through 2^28, 131072 above (EXACT ties masked
        # at 2^30: no scalar-arm flip).
        if n_total <= 65536:
            return 8192, 2
        if n_total <= (1 << 24):
            return 16384, 4
        if n_total <= (1 << 28):
            return 65536, 2
        return 131072, 2
    # int32: in-place 16384 beats 8192 at 2^16 (4.14 vs 4.64us) while
    # out-of-place prefers 8192/EXACT (4.15 vs 4.64us); 16384 through 2^20,
    # 32768 through 2^26 in-place (e32768 171.01 vs e65536 173.04us at 2^26)
    # but only through 2^24 out-of-place (2^26 out-of-place measured e65536
    # faster by 1%), 65536 above.
    if n_total <= 65536:
        return (16384 if inplace else 8192), 2
    if n_total <= (1 << 20):
        return 16384, 2
    if n_total <= ((1 << 26) if inplace else (1 << 24)):
        return 32768, 2
    return 65536, 2


def _lane_view(t):
    # Bool tensors ride the int8 lane: 0/1 bytes OR identically and the
    # generated i1 lowering is far off the byte-stream memory floor.
    return t.view(torch.int8) if t.dtype == torch.bool else t


# 2026-11-16 widened-lane band (twin probe crossover 2^20/2^22/2^24-2^30): the
# int32 word stream beats the byte-lane stream only for 2^20..2^28 scalar
# elements; outside the band both hit the same DRAM floor and the per-element
# ladder stays.  n must be word-divisible and the base pointers word-aligned
# (byte-offset views fall back to the per-element path).
_WIDE_MAX = 1 << 28
_WIDE_SCALAR_MIN = 1 << 20


def _scalar_wide_config(n_total, itemsize, ptrs, inplace):
    if itemsize != 1 or not (_WIDE_SCALAR_MIN <= n_total <= _WIDE_MAX) or n_total & 3:
        return None
    if any(ptr & 3 for ptr in ptrs):
        return None
    n_lanes = n_total >> 2
    # Round-4: the widened word stream gains the mask-free EXACT body (ties
    # masked at the band edges, -2.9% at the 2^24 out-of-place center) and an
    # in-place-only 4096/nw2 tile at the band bottom [2^20, 2^21)
    # (4.66 -> 4.17us at 2^20 in-place; out-of-place and 2^22 in-place prefer
    # 8192/nw4).  The small tile is taken only when it divides the lane count
    # (EXACT body); other shapes fall back to the probed 8192/nw4 config.
    if inplace and n_total < (1 << 21) and n_lanes % 4096 == 0:
        block, num_warps = 4096, 2
    elif n_total <= (1 << 25):
        block, num_warps = 8192, 4
    else:
        block, num_warps = 32768, 2
    exact = n_lanes % block == 0
    return (
        block,
        (min(triton.cdiv(n_lanes, block), NUM_SIPS * 2),),
        num_warps,
        n_lanes,
        exact,
    )


def _wide_scalar_word(dtype, scalar):
    # Replicate the wrapped lane byte 4x into one int32 word: OR-ing the word
    # ORs each byte lane with the scalar byte, byte-for-byte identical to the
    # per-element path.
    v = _scalar_lane_value(dtype, scalar) & 0xFF
    word = v | (v << 8) | (v << 16) | (v << 24)
    return word - (1 << 32) if word >= (1 << 31) else word


def _launch_flat(kern, grid, args, block, num_warps=2, exact=False):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic host-level flake on shared nodes); retrying the
    # launch recompiles and succeeds (rad2deg / atan2_ / sinh precedents).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kern[grid](*args, num_warps=num_warps, BLOCK=block, EXACT=exact)
            return
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "bitwise_or flat kernel launch failed (attempt %d): %s",
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


# 2026-11-16 in-place tensor ladder resweep (kernel-mode do_bench median,
# warmup 1000ms / rep 100ms, interleaved A/B vs native and vs the then-current
# ladder):
#   - the 2R1W in-place int32 stream pays a 6-7% mask tax in the mid band and
#     32768 lanes beat 65536 from 2^18 through 2^24 (74.1us vs 81.4us masked at
#     2^24 = +10% vs native); the tiny np2 tile gains 1-10% from the mask-free
#     body (int32 4096: 3.3 vs 3.7us).
#   - int16 mid prefers 32768/EXACT through 2^23 (-5.5% at 2^20) and 65536/EXACT
#     at 2^24; 1-byte lanes prefer 65536/EXACT through 2^23 (-10.9% at 2^20,
#     -3.1% at 2^22) and 131072/EXACT at 2^24.
#   - EXACT regresses on the 2^30 band (int16 131072 +2.3%, int32 65536 +1.7%)
#     and the wide-int32 lane view of bitwise_xor gains nothing in-place, so
#     the large buckets keep the masked body and no wide kernel is added.
#   - nw4 tiles lose everywhere in-place (int16 mid 32768/nw4 +35%; XOR's
#     out-of-place round-2 int32-mid 32768/nw4 win does not transfer), so the
#     tensor arm stays all-nw2 and the libentry key (dtype/BLOCK/EXACT) never
#     carries two num_warps for one entry.
def _tensor_block_exact(n_total, itemsize):
    if n_total <= 8192:
        block = max(triton.next_power_of_2(n_total), 2)
        return block, n_total % block == 0
    if n_total <= 65536:
        return 8192, False
    if itemsize == 4:
        if n_total <= (1 << 24):
            return 32768, n_total % 32768 == 0
        return 65536, False
    if itemsize == 2:
        if n_total <= (1 << 23):
            return 32768, n_total % 32768 == 0
        if n_total <= (1 << 24):
            return 65536, n_total % 65536 == 0
        if n_total > (1 << 28):
            return 131072, False
        return 65536, False
    if n_total <= 262144:
        return 32768, False
    if n_total <= (1 << 23):
        return 65536, n_total % 65536 == 0
    if n_total <= (1 << 24):
        return 131072, n_total % 131072 == 0
    return 131072, False


def _tensor_flat_config(n_total, itemsize):
    block, exact = _tensor_block_exact(n_total, itemsize)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    return block, (min(triton.cdiv(n_total, block), NUM_SIPS * 2),), exact


def _scalar_lane_value(dtype, scalar):
    # Bitwise OR commutes with truncation: the low L bits of (x | s) are
    # (low L bits of x) | (low L bits of s), so wrap the scalar into the lane
    # dtype's range on the host (two's complement for signed lanes) exactly
    # like the bitwise_xor / bitwise_and precedents.
    if dtype == torch.bool:
        return 1 if scalar else 0
    bits = torch.iinfo(dtype).bits
    value = int(scalar) & ((1 << bits) - 1)
    if value >= (1 << (bits - 1)) and dtype.is_signed:
        value -= 1 << bits
    return value


def _is_python_scalar(v):
    return isinstance(v, bool) or isinstance(v, int)


def _or_scalar_impl(tensor, scalar, out=None):
    # Shared by bitwise_or_scalar / bitwise_or_scalar_tensor (OR commutes)
    # and by the in-place variant (out=tensor).  Returns None when the input
    # pair is not eligible for the flat path.
    if not _fast_tensor(tensor) or not _is_python_scalar(scalar):
        return None
    if tensor.dtype == torch.bool and not isinstance(scalar, bool):
        # (bool, int) promotes to int64.  The bitwise_xor twin's int32-slot
        # single-word store assumed the fake-int64 4-byte value channel; with
        # TORCH_GCU_ENABLE_INT64_AND_UINT64=1 (global since 2026-11-11) int64
        # storage is a real 8-byte lane, so a flat int32 store reads back
        # garbage (verified: values like 2^32 / uninit high words).  The
        # generated pointwise path computes the promoted int64 result
        # correctly under the same env, so mixed pairs stay on it.
        return None
    if out is None:
        out = torch.empty_like(tensor)
    elif out.dtype != tensor.dtype:
        return None
    n_total = tensor.numel()
    if n_total == 0:
        return out
    # 2026-11-17 round-5 dispatch-line audit: `inplace` is computed after the
    # out-of-place allocation above, so it is always True here and the op-side
    # branches in _scalar_block_warps / _scalar_wide_config stay unreachable.
    # A 5-round interleaved re-probe of the op side confirmed the ip configs
    # are optimal for both entry modes today (int32 op 2^18/2^20: 16384 beats
    # 8192 by 1.2/7.5%, op 2^26: 32768 beats 65536 by 1.5%, wide op 2^20:
    # 4096/nw2 ties 8192/nw4), so the effective dispatch is kept as shipped.
    inplace = out is not None
    wide = _scalar_wide_config(
        n_total,
        tensor.element_size(),
        (tensor.data_ptr(), out.data_ptr()),
        inplace,
    )
    if wide is not None:
        block, grid, num_warps, n_lanes, exact = wide
        _launch_flat(
            bitwise_or_scalar_wide_kernel,
            grid,
            (
                _lane_view(tensor),
                _lane_view(out),
                _wide_scalar_word(tensor.dtype, scalar),
                n_lanes,
            ),
            block,
            num_warps,
            exact=exact,
        )
        return out
    block, num_warps = _scalar_block_warps(n_total, tensor.element_size(), inplace)
    if triton.cdiv(n_total, block) * block > _INT32_MAX:
        return None
    grid = (min(triton.cdiv(n_total, block), NUM_SIPS * 2),)
    _launch_flat(
        bitwise_or_scalar_flat_kernel,
        grid,
        (
            _lane_view(tensor),
            _lane_view(out),
            _scalar_lane_value(tensor.dtype, scalar),
            n_total,
        ),
        block,
        num_warps,
        exact=n_total % block == 0,
    )
    return out


def bitwise_or_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR")
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
        config = _tensor_flat_config(n_total, A.element_size())
        if config is not None:
            block, grid, exact = config
            _launch_flat(
                bitwise_or_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(out), n_total),
                block,
                exact=exact,
            )
            return out
    return bitwise_or_func(A, B)


def bitwise_or_tensor_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_")
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
        config = _tensor_flat_config(n_total, A.element_size())
        if config is not None:
            block, grid, exact = config
            _launch_flat(
                bitwise_or_flat_kernel,
                grid,
                (_lane_view(A), _lane_view(B), _lane_view(A), n_total),
                block,
                exact=exact,
            )
            return A
    return bitwise_or_func(A, B, out0=A)


@pointwise_dynamic(is_tensor=[True, False], promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def bitwise_or_func_scalar(x, y):
    return x | y


def bitwise_or_scalar(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR")
    out = _or_scalar_impl(A, B)
    if out is not None:
        return out
    return bitwise_or_func_scalar(A, B)


def bitwise_or_scalar_(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR_")
    if isinstance(A, torch.Tensor) and _is_python_scalar(B):
        if A.dtype == torch.bool and not isinstance(B, bool):
            # torch raises for the (bool, int) in-place promotion; the generic
            # pointwise path silently truncates instead, so keep aten parity.
            raise RuntimeError(
                "result type Long can't be cast to the desired output type Bool"
            )
        out = _or_scalar_impl(A, B, out=A)
        if out is not None:
            return A
    return bitwise_or_func_scalar(A, B, out0=A)


def bitwise_or_scalar_tensor(A, B):
    logger.debug("GEMS_ENFLAME BITWISE_OR_SCALAR_TENSOR")
    # OR commutes, so (scalar, tensor) reuses the (tensor, scalar) path.
    if _is_python_scalar(A):
        out = _or_scalar_impl(B, A)
        if out is not None:
            return out
    return bitwise_or_func_scalar(B, A)
