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

# Kunlunxin (XPU) override for gcd / gcd_out.
#
# NOTE (kunlunxin/XPU): the generic `flag_gems/ops/gcd.py` kernels cannot be
# compiled by the XPU backend:
#   1. `libdevice.ffs` has no XPU lowering (`undefined symbol: Unsupported`
#      at the elfconv link stage).
#   2. The data-dependent `while tl.sum(active) > 0` loop makes
#      `TritonXPUUnrollControl` mis-compile the loop body (`arith.select`
#      failed to verify ... same type) or overflow `uni_sram`.
#   3. Integer `%` (modulo) inside an unrolled loop hits the same pass
#      failure (`arith.cmpi` verify error + `uni_sram` OutOfResources), so
#      Euclid-with-remainder is not viable either.
#
# The kernels below therefore use Stein's binary GCD in the *unsigned*
# domain with a fixed-trip `for` loop and a SWAR-free, division-free
# binary-search ctz. Magnitudes are handled as unsigned so INT_MIN inputs
# need no special case (|INT_MIN| = 2^(bits-1) fits in the unsigned type).
# Empirical worst-case loop trip counts (400k random + Fibonacci chains):
# uint32 <= 33, uint64 <= 64; ITERS adds >20% margin.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_I16_MIN_LUT_CACHE = {}

# Iteration bounds (see module docstring).
# Worst-case loop trips (proven: each active iteration strictly decreases the
# bit width of max(u, v), so trips <= bit-width of the larger magnitude):
#   int16 |other| <= 2^15-1 after initial strip -> <= 15 trips; ITERS_U16 = 20.
#   uint32 magnitudes <= 2^31 -> <= 32 trips (empirical 33); ITERS_U32 = 36.
#   uint64 magnitudes <= 2^63 -> <= 64 trips; ITERS_U64 = 72.
_ITERS_U16 = tl.constexpr(20)
_ITERS_U32 = tl.constexpr(36)
_ITERS_U64 = tl.constexpr(72)

BLOCK_I16 = 512
# NOTE (kunlunxin/XPU perf): int16 hits its throughput sweet spot at
# BLOCK=1024 on large inputs (~1.5-2x vs 512), but tiny inputs regress
# ~25% (fewer programs -> worse latency/launch floor). Dispatch by numel.
BLOCK_I16_LARGE = 1024
_I16_LARGE_MIN_NUMEL = 8192
BLOCK_I32 = 512
BLOCK_I64 = 256
NUM_WARPS = 4


@triton.jit
def _neg_u(x, zu):
    # Modular negation in the unsigned domain: zu = x - x (same unsigned
    # type), so `zu - x` never mixes signed/unsigned operands (mixed
    # signedness promotes to a wider type on XPU and breaks the wraparound
    # that INT_MIN magnitudes rely on).
    return zu - x


@triton.jit
def _ctz_nz32(x, zu):
    # ctz of a nonzero uint32 via float-bitcast: lsb = x & -x is a power of
    # two, and its float32 exponent field is exactly 127 + ctz. XPU verified
    # (tmp/perf/gcd_/probe_ctz.py: exact for all 2^k edges up to 2^31 plus
    # 4M random values; uint32->f32 conversion of powers of two is exact)
    # and ~25% faster than the previous binary-search version.
    lsb = x & (zu - x)
    f = lsb.to(tl.float32)
    e = (f.to(tl.int32, bitcast=True) >> 23) & 0xFF
    return e - 127


@triton.jit
def _ctz_nz64(x, zu):
    # ctz of a nonzero uint64 via float-bitcast (float64 exponent field =
    # 1023 + ctz; XPU verified by tmp/perf/gcd_/probe_ctz64.py, 0/4.19M
    # mismatches incl. 2^k edges up to 2^63).
    lsb = x & (zu - x)
    f = lsb.to(tl.float64)
    e = (f.to(tl.int64, bitcast=True) >> 52) & 0x7FF
    return (e - 1023).to(tl.int32)


@triton.jit
def _unsigned_binary_gcd32(au, bu, ITERS: tl.constexpr):
    # au, bu: uint32 magnitudes (0 allowed). Returns gcd as uint32.
    zu = au - au
    zero_a = au == 0
    zero_b = bu == 0
    # gcd(a, 0) = a, gcd(0, b) = b, gcd(0, 0) = 0
    res0 = tl.where(zero_a, bu, au)
    both = (~zero_a) & (~zero_b)
    one = zu + 1
    common = _ctz_nz32(tl.where(both, au | bu, one), zu)
    # Zero-operand lanes are parked at (u = res0, v = 0), which is a fixed
    # point of the branchless loop body below (z -> vs = u -> swap = false
    # -> u' = u, v' = 0). This removes the per-iteration `act`/`both`
    # guards (~6 fewer int32 ops per iteration, ~25% of kernel time).
    u = tl.where(both, au >> _ctz_nz32(tl.where(both, au, one), zu), res0)
    v = tl.where(both, bu, zu)
    for _ in range(ITERS):
        z = v == 0
        # v != 0 lanes: ctz in [0, 31]; v == 0 lanes discard the shift.
        vs = tl.where(z, u, v >> _ctz_nz32(v, zu))
        swap = u > vs
        small = tl.where(swap, vs, u)
        large = tl.where(swap, u, vs)
        u = small
        v = large - small
    return tl.where(both, u << common, res0)


@triton.jit
def _unsigned_binary_gcd64(au, bu, ITERS: tl.constexpr):
    # au, bu: uint64 magnitudes (0 allowed). Returns gcd as uint64.
    zu = au - au
    zero_a = au == 0
    zero_b = bu == 0
    res0 = tl.where(zero_a, bu, au)
    both = (~zero_a) & (~zero_b)
    one = zu + 1
    common = _ctz_nz64(tl.where(both, au | bu, one), zu)
    u = tl.where(both, au >> _ctz_nz64(tl.where(both, au, one), zu), res0)
    v = tl.where(both, bu, zu)
    for _ in range(ITERS):
        z = v == 0
        vs = tl.where(z, u, v >> _ctz_nz64(v, zu))
        swap = u > vs
        small = tl.where(swap, vs, u)
        large = tl.where(swap, u, vs)
        u = small
        v = large - small
    return tl.where(both, u << common, res0)


@triton.jit
def gcd_kernel_i16(x_ptr, y_ptr, lut_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # int16 magnitudes fit int32. For lanes where an operand is INT16_MIN,
    # the result is looked up from a CPU-precomputed LUT indexed by |other|
    # (0..32767), which reproduces ATen's signed-wraparound gcd bit-exactly
    # (including gcd(INT16_MIN, INT16_MIN) = 32768 -> -32768); the generic
    # implementation uses the same LUT scheme. Non-special lanes use the
    # unsigned binary GCD; gcd(INT16_MIN, 0/-2^k style pairs) agrees with
    # ATen there as well.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    # NOTE (kunlunxin/XPU): masked loads are unreliable (`other` is ignored
    # and even valid lanes can be corrupted). Clamp addresses to be always
    # in-bounds, load unmasked, and mask at register level only.
    safe_offsets = tl.where(mask, offsets, 0)
    x = tl.load(x_ptr + safe_offsets).to(tl.int32)
    y = tl.load(y_ptr + safe_offsets).to(tl.int32)
    xu = x.to(tl.uint32)
    yu = y.to(tl.uint32)
    zu = xu - xu
    au = tl.where(x < 0, _neg_u(xu, zu), xu)
    bu = tl.where(y < 0, _neg_u(yu, zu), yu)
    normal_res = _unsigned_binary_gcd32(au, bu, _ITERS_U16)

    min_value: tl.constexpr = -32768
    min_x = x == min_value
    min_y = y == min_value
    special_mask = mask & (min_x | min_y)
    both_min = special_mask & min_x & min_y
    one_min = special_mask & (~both_min)
    other_abs = tl.where(min_x, tl.abs(y), tl.abs(x))
    # NOTE (kunlunxin/XPU perf): the LUT lookup goes through the slow
    # discrete-gather path for every lane (~60% of kernel time on large
    # shapes) even though special lanes are ~1/32768 of random data. Gate
    # the gather behind a *block-uniform* branch: only programs that
    # actually contain a special lane pay for the gather (a scalar reduce
    # + branch is uniform control flow, no lane divergence).
    n_special = tl.sum(special_mask.to(tl.int32))
    if n_special > 0:
        # one_min lanes have |other| <= 32767; clamp the rest to 0 so the
        # load address is always inside the LUT (no masked load on XPU).
        lut_idx = tl.where(one_min, other_abs, 0)
        lut_val = tl.load(lut_ptr + lut_idx).to(tl.int32)
        special_res = tl.where(both_min, min_value, lut_val)
        out = tl.where(special_mask, special_res, normal_res)
    else:
        out = normal_res
    tl.store(out_ptr + offsets, out.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def gcd_kernel_i32(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # |INT32_MIN| = 2^31 fits uint32, so the unsigned domain needs no
    # INT_MIN special case; the 2^31 result truncates back to INT32_MIN,
    # matching ATen. Known deviation: for pairs where an operand is exactly
    # INT32_MIN and the other has an odd part, ATen's signed-wraparound
    # Euclid occasionally reports the negated magnitude while this kernel
    # reports the true (positive) magnitude. XPU cannot lower integer
    # remainder inside kernel loops (uni_sram overflow + UnrollControl
    # verify failures), so ATen's Euclid-with-`%` cannot be reproduced here;
    # the magnitude itself is always correct.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    safe_offsets = tl.where(mask, offsets, 0)
    x = tl.load(x_ptr + safe_offsets)
    y = tl.load(y_ptr + safe_offsets)
    xu = x.to(tl.uint32)
    yu = y.to(tl.uint32)
    zu = xu - xu
    au = tl.where(x < 0, _neg_u(xu, zu), xu)
    bu = tl.where(y < 0, _neg_u(yu, zu), yu)
    ru = _unsigned_binary_gcd32(au, bu, _ITERS_U32)
    tl.store(out_ptr + offsets, ru.to(out_ptr.type.element_ty), mask=mask)


@triton.jit
def gcd_kernel_i64(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    # |INT64_MIN| = 2^63 fits uint64; the 2^63 result truncates back to
    # INT64_MIN, matching ATen. Same INT_MIN sign deviation as the int32
    # kernel (see note above): the magnitude is always exact, only the sign
    # of ATen's signed-wraparound wart can differ for INT64_MIN pairs with
    # an odd part, because integer remainder is not compilable on XPU.
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    safe_offsets = tl.where(mask, offsets, 0)
    x = tl.load(x_ptr + safe_offsets)
    y = tl.load(y_ptr + safe_offsets)
    xu = x.to(tl.uint64)
    yu = y.to(tl.uint64)
    zu = xu - xu
    au = tl.where(x < 0, _neg_u(xu, zu), xu)
    bu = tl.where(y < 0, _neg_u(yu, zu), yu)
    ru = _unsigned_binary_gcd64(au, bu, _ITERS_U64)
    tl.store(out_ptr + offsets, ru.to(out_ptr.type.element_ty), mask=mask)


def _kernel_meta(dtype, numel):
    if dtype == torch.int16:
        block = BLOCK_I16_LARGE if numel >= _I16_LARGE_MIN_NUMEL else BLOCK_I16
        return gcd_kernel_i16, block, NUM_WARPS
    if dtype == torch.int32:
        return gcd_kernel_i32, BLOCK_I32, NUM_WARPS
    if dtype == torch.int64:
        return gcd_kernel_i64, BLOCK_I64, NUM_WARPS
    raise TypeError(f"unsupported dtype for gcd: {dtype}")


def _get_i16_min_lut(device):
    # Same scheme as the generic implementation: precompute
    # gcd(INT16_MIN, v) for v in [0, 32767] on CPU (ATen-exact) and cache
    # per device. CPU dispatch is unaffected by the gems XPU overrides.
    key = (device.type, device.index)
    lut = _I16_MIN_LUT_CACHE.get(key)
    if lut is None:
        info = torch.iinfo(torch.int16)
        lhs = torch.full((info.max + 1,), info.min, dtype=torch.int16)
        rhs = torch.arange(info.max + 1, dtype=torch.int16)
        lut = torch.gcd(lhs, rhs).to(device=device)
        _I16_MIN_LUT_CACHE[key] = lut
    return lut


def _launch_gcd(lhs, rhs, out):
    numel = out.numel()
    if numel == 0:
        return out

    kernel, block, num_warps = _kernel_meta(out.dtype, numel)
    grid = (triton.cdiv(numel, block),)
    if out.dtype == torch.int16:
        lut = _get_i16_min_lut(out.device)
        kernel[grid](lhs, rhs, lut, out, numel, BLOCK=block, num_warps=num_warps)
    else:
        kernel[grid](lhs, rhs, out, numel, BLOCK=block, num_warps=num_warps)
    return out


def gcd(self, other, *, out=None):
    from flag_gems.ops.gcd import _materialize_inputs

    logger.debug("GEMS_KUNLUNXIN GCD")
    promoted_dtype = torch.promote_types(self.dtype, other.dtype)
    if self.numel() == 0 or other.numel() == 0:
        # Short-circuit empty inputs: the generic _materialize_inputs path
        # goes through the gems broadcast_tensors override, which fails on
        # 0-size tensors. torch.broadcast_shapes raises for incompatible
        # shapes just like ATen.
        shape = torch.broadcast_shapes(self.shape, other.shape)
        result = torch.empty(shape, dtype=promoted_dtype, device=self.device)
        if out is not None:
            out.copy_(result)
            return out
        return result

    lhs, rhs, _ = _materialize_inputs(self, other)
    result = torch.empty_like(lhs, dtype=promoted_dtype)
    _launch_gcd(lhs.reshape(-1), rhs.reshape(-1), result.reshape(-1))
    result = result.view(lhs.shape)
    if out is None:
        return result

    out.copy_(result)
    return out


def gcd_out(self, other, *, out=None):
    logger.debug("GEMS_KUNLUNXIN GCD_OUT")
    if out is None:
        return gcd(self, other)
    return gcd(self, other, out=out)
