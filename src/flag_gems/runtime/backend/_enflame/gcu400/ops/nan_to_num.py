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
INT32_MAX = torch.iinfo(torch.int32).max
_FLAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# bf16's promoted fp32 clamp chain (cvt + clamp + select) is too long to hide
# under the memory stream on very large tensors; the split2 kernel issues both
# half-tile loads before any compute and recovers ~4% there (measured on
# GCU400/L600, 512M-element bucket).  fp16/fp32 gain nothing from it.
SPLIT2_MIN_NUMEL = 1 << 28


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def nan_to_num_flat_kernel(
    x_ptr,
    out_ptr,
    nan,
    posinf,
    neginf,
    N_total,
    BLOCK: tl.constexpr,
    CLAMPABLE: tl.constexpr,
):
    # GCU400 override of the generic pointwise_dynamic path: the codegen kernel
    # lowers nan detection through tl_extra_shim.isnan (libdevice call) and a
    # three-deep tl.where chain, which measured 0.50-0.57x of torch native on
    # the 16M/1G core shapes.  When the replacement bounds lie outside the
    # dtype's finite range (the default finfo.max/min), the infinities are the
    # only values the bounds can touch, so the two where-selects collapse into
    # one clamp; NaN keeps the self-comparison test x != x.  Scalars are
    # narrowed to the element dtype so fp16 stays in native fp16 math (fp16 16M
    # bucket: 1.09x vs 0.89x with fp32 promotion).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    et = out_ptr.dtype.element_ty
    nan_t = nan.to(et)
    posinf_t = posinf.to(et)
    neginf_t = neginf.to(et)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        if CLAMPABLE:
            y = tl.clamp(x, neginf_t, posinf_t)
            y = tl.where(x != x, nan_t, y)
        else:
            y = tl.where(x != x, nan_t, x)
            y = tl.where(x == float("inf"), posinf_t, y)
            y = tl.where(x == -float("inf"), neginf_t, y)
        tl.store(out_ptr + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def nan_to_num_flat_split2_kernel(
    x_ptr, out_ptr, nan, posinf, neginf, N_total, BLOCK: tl.constexpr
):
    # Clamp-form fast path (host only dispatches here when the bounds cannot
    # clip finite values), but each iteration issues both half-tile loads
    # before any compute (manual load hoisting): on bf16 the promoted fp32
    # conversion chain otherwise serialises against the memory stream on very
    # large tensors.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    half: tl.constexpr = BLOCK // 2
    arange = tl.arange(0, half)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    et = out_ptr.dtype.element_ty
    nan_t = nan.to(et)
    posinf_t = posinf.to(et)
    neginf_t = neginf.to(et)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off0 = block_id * BLOCK + arange
        off1 = off0 + half
        m0 = off0 < N_total
        m1 = off1 < N_total
        x0 = tl.load(x_ptr + off0, mask=m0)
        x1 = tl.load(x_ptr + off1, mask=m1)
        y0 = tl.clamp(x0, neginf_t, posinf_t)
        y1 = tl.clamp(x1, neginf_t, posinf_t)
        y0 = tl.where(x0 != x0, nan_t, y0)
        y1 = tl.where(x1 != x1, nan_t, y1)
        tl.store(out_ptr + off0, y0, mask=m0)
        tl.store(out_ptr + off1, y1, mask=m1)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def nan_to_num_flat_default_kernel(
    x_ptr,
    out_ptr,
    nan,
    N_total,
    BLOCK: tl.constexpr,
):
    # bf16 default-bounds fast path (posinf == finfo.max, neginf == finfo.min,
    # i.e. what every posinf/neginf=None call lands on).  The generic clamp arm
    # promotes bf16 through fp32 (cvt + f32 min/max + round back) and that op
    # chain fails to hide under the memory stream: 16M bf16 in-place measured
    # 34.1us vs a 26.6us pure-copy floor (GCU400/L600, do_bench median).
    # In the IEEE bit domain the default-bounds work collapses to: max finite
    # == inf - 1 in bit pattern for BOTH signs (0x7F80 - 1 = 0x7F7F,
    # 0xFF80 - 1 = 0xFF7F) and NaN <=> magnitude above the +inf pattern, so
    # the whole transform is one AND + two compares + one subtract + one
    # select, with zero conversions.  Measured 16M bf16 34.1 -> 31.5us,
    # 256M bf16 412.7 -> 374.0us in-place.  aten semantics never clip finite
    # values, so the bit-domain form is exact; bitwise-validated against
    # torch.nan_to_num on +/-NaN/+inf/-inf/+0/-0/subnormal/+max/-max/random
    # batteries.
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    et = out_ptr.dtype.element_ty
    nan_e = nan.to(et)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        u = x.to(tl.int16, bitcast=True)
        mag = u & 0x7FFF
        is_nan = mag > 0x7F80
        is_inf = mag == 0x7F80
        t = u - is_inf.to(tl.int16)
        y = tl.where(is_nan, nan_e, t.to(et, bitcast=True))
        tl.store(out_ptr + off, y, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def nan_to_num_flat_default_f32_kernel(
    x_ptr,
    out_ptr,
    nan,
    N_total,
    BLOCK: tl.constexpr,
):
    # fp32 default-bounds fast path: int32 twin of the bf16 kernel above
    # (0x7F800000 - 1 = 0x7F7FFFFF = max finite, holds for both signs).  The
    # clamp arm pays two fp32 min/max ops plus the NaN self-compare; on GCU
    # min/max are the pricey ops of that chain, and it exposes over the copy
    # floor (16M in-place: clamp 52.1us vs copy 46.8us, GCU400/L600 do_bench
    # median).  The AND/compare/sub int chain hides instead: 16M 48.1us.
    # aten semantics never clip finite values, so the bit form is exact;
    # bitwise-validated against torch.nan_to_num on the same battery as the
    # bf16 kernel (+/-NaN/+inf/-inf/+0/-0/subnormal/+max/-max/random).
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    et = out_ptr.dtype.element_ty
    nan_e = nan.to(et)
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + arange
        mask = off < N_total
        x = tl.load(x_ptr + off, mask=mask)
        u = x.to(tl.int32, bitcast=True)
        mag = u & 0x7FFFFFFF
        is_nan = mag > 0x7F800000
        is_inf = mag == 0x7F800000
        t = u - is_inf.to(tl.int32)
        y = tl.where(is_nan, nan_e, t.to(et, bitcast=True))
        tl.store(out_ptr + off, y, mask=mask)


@pointwise_dynamic(
    is_tensor=[True, False, False, False], promotion_methods=[(0, "DEFAULT")]
)
@triton.jit
def nan_to_num_func(x, nan, posinf, neginf):
    x_nan = x != x
    x_posinf = x == float("inf")
    x_neginf = x == -float("inf")
    x = tl.where(x_nan, nan, x)
    x = tl.where(x_posinf, posinf, x)
    x = tl.where(x_neginf, neginf, x)
    return x


def _choose_block(N_total, itemsize):
    # do_bench(median) sweep, kernel mode, vs torch.nan_to_num on GCU400/L600:
    #   16M bucket peaks at 65536 (fp16 28.1us vs 28.9us @32768, bf16 34.3 vs
    #   36.5, fp32 ties); the >2^28 bucket splits by dtype -- 2-byte elements
    #   gain from 131072 (bf16 754 vs 811us @65536) while fp32 prefers 65536
    #   (1344 vs 1352us).
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if N_total >= (1 << 28) and itemsize <= 2:
        return 131072
    return 65536


def _choose_block_default(N_total):
    # BLOCK ladder for the bf16 default-bounds bit-domain kernel only.  Same
    # probe protocol as _choose_block, plus the 2026-11-16 re-sweep: the 16M
    # bucket still peaks at 65536 (32.0us vs 32.4 @131072), 32M ties (54.7 vs
    # 54.6), and from >2^25 on the wide tile wins and the gap widens with size
    # (64M 97.9 vs 100.6us, 256M 360.5 vs ~375, 1G ~1410us band) -- the shared
    # ladder above only switches at >=2^28, leaving the 32M-256M range on the
    # narrow tile.  B262144 spills (1831.7us @1G); nw4 loses at every size
    # (16M 37.1 vs 32.0, 1G 1497.8 vs 1414.4).
    if N_total <= 8192:
        return max(256, triton.next_power_of_2(N_total))
    if N_total <= 65536:
        return 8192
    if N_total <= 524288:
        return 32768
    if N_total > (1 << 25):
        return 131072
    return 65536


def _try_flat(A, nan, posinf, neginf, out=None):
    # Returns the result tensor when the flat kernel applied, else None.
    if A.dtype not in _FLAT_DTYPES or not A.is_contiguous() or A.numel() == 0:
        return None
    N_total = A.numel()
    finfo = torch.finfo(A.dtype)
    clampable = posinf >= finfo.max and neginf <= finfo.min
    default_bounds = posinf == finfo.max and neginf == finfo.min
    if A.dtype == torch.bfloat16 and default_bounds:
        # The bf16 bit-domain kernel takes its own BLOCK ladder: from >2^25
        # elements the wide 131072 tile wins and the gap to 65536 widens with
        # size (64M 97.9 vs 100.6us, 256M 360.5 vs ~375), while the 16M bucket
        # keeps its 65536 sweet spot (32.0 vs 32.4us).
        BLOCK = _choose_block_default(N_total)
    else:
        BLOCK = _choose_block(N_total, A.element_size())
    num_blocks = triton.cdiv(N_total, BLOCK)
    if num_blocks * BLOCK > INT32_MAX:
        return None
    if out is None:
        out = torch.empty_like(A)
    grid_size = min(num_blocks, NUM_SIPS * 2)
    if A.dtype == torch.bfloat16 and default_bounds:
        # Default-bounds bf16 takes the bit-domain kernel at every size: the
        # promoted fp32 clamp chain is the measured bottleneck below the
        # split2 threshold (16M bf16 in-place 34.1 -> 31.5us, 256M 412.7 ->
        # 374.0us, do_bench median) and stays one above it (1G in-place
        # 1447us on the split2 clamp kernel vs 1397us for this kernel at
        # BLOCK=131072; the split2 prefetch structure buys nothing
        # measurable for the short int16 chain -- split2/interleaved/plain
        # forms all land within run noise).
        with torch_device_fn.device(A.device):
            nan_to_num_flat_default_kernel[(grid_size,)](
                A, out, nan, N_total, BLOCK=BLOCK, num_warps=2
            )
    elif A.dtype == torch.float32 and default_bounds:
        # Default-bounds fp32 takes its int32 bit-domain twin: the clamp arm
        # pays two fp32 min/max ops plus the NaN self-compare and exposes
        # over the copy floor (16M in-place 52.1 -> 48.1us, copy 46.8us).
        with torch_device_fn.device(A.device):
            nan_to_num_flat_default_f32_kernel[(grid_size,)](
                A, out, nan, N_total, BLOCK=BLOCK, num_warps=2
            )
    elif clampable and A.dtype == torch.bfloat16 and N_total > SPLIT2_MIN_NUMEL:
        # Remaining >2^28 bf16 corner: clampable bounds that are not exactly
        # the default finfo.max/min (e.g. +/-inf), where the bit-domain
        # shortcut does not apply and the promoted clamp chain still needs
        # the split2 prefetch to hide under the stream.
        with torch_device_fn.device(A.device):
            nan_to_num_flat_split2_kernel[(grid_size,)](
                A, out, nan, posinf, neginf, N_total, BLOCK=BLOCK, num_warps=2
            )
    else:
        with torch_device_fn.device(A.device):
            nan_to_num_flat_kernel[(grid_size,)](
                A,
                out,
                nan,
                posinf,
                neginf,
                N_total,
                BLOCK=BLOCK,
                CLAMPABLE=clampable,
                num_warps=2,
            )
    return out


def nan_to_num(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_ENFLAME NAN_TO_NUM")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    result = _try_flat(A, nan, posinf, neginf)
    if result is not None:
        return result
    return nan_to_num_func(A, nan, posinf, neginf)


def nan_to_num_(A, nan=None, posinf=None, neginf=None):
    logger.debug("GEMS_ENFLAME NAN_TO_NUM_")
    if posinf is None:
        posinf = torch.finfo(A.dtype).max
    if neginf is None:
        neginf = torch.finfo(A.dtype).min
    if nan is None:
        nan = 0.0
    result = _try_flat(A, nan, posinf, neginf, out=A)
    if result is not None:
        return result
    nan_to_num_func(A, nan, posinf, neginf, out0=A)
    return A
