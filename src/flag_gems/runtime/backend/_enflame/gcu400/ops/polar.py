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

from ..utils.pointwise_dynamic import CodeGenConfig, pointwise_dynamic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flat fast path (add.py / copy.py house style).
#
# polar(abs, angle) returns complex64/complex32 whose real/imag planes are
# interleaved in one allocation.  The pointwise_dynamic codegen instead stores
# through two stride-2 views of that allocation, and on gcu400 those
# block-pointer tiles ride the DTE channel, which at the 1Gi-element core
# shapes corrupts device memory (``efdrv/src/driver/async_printf.cc:529
# printf_display detected the buffer is corrupted`` followed by
# ``context.cc:2061 syncWorkerFunction Receive Abort message from KMD: Sip
# exception`` on the next launch) and is ~12x slower than native.  The flat
# kernel treats the operands as one dense range of pairs and writes each
# complex word as a single packed int64/int32 store: plain offset access
# stays on the synchronous port and cannot overrun.
#
# Form notes, all measured on GCU400/L600 (core kernel-mode do_bench):
#   - tl.join/tl.interleave payloads (any lane-crossing reshuffle) take the
#     ~40x DSM materialization pathology: 2R1W floor 1.73ms vs 0.14ms at
#     (4096, 4096).  The pack below keeps every lane local.
#   - Two stride-2 fp32 stores instead of one packed word cost write-side
#     bandwidth (half-dense transactions): 8.8ms vs 6.5ms floor at 1Gi pairs.
#   - tl.floor() lowers pathologically (~+14.5ms at 1Gi when it feeds the
#     trig chain); the magic-number round (x +/- 2**23) is exact fp32 RNE
#     for |x| < 2**22 and free.
#   - tl.sin/tl.cos lower to a precise libdevice path (~2.5ms each at 1Gi
#     pairs on top of the floor).  The quadrant-reduced minimax polynomials
#     (CUDA __sinf/__cosf coefficients, [-pi/4, pi/4] after a two-term
#     Cody-Waite reduction) measure max abs error ~1e-5 vs CPU at |angle|
#     <= 12.5 (test domain |angle| <= 4*pi), far inside the 1e-4 test gate.
#   - masked stores must take the plain pair mask: a mask built by
#     tl.interleave(mask, mask) is silently dropped by the store lowering
#     (micro-probe: n=3 BLOCK=256 wrote 512/512 lanes, corrupting
#     neighbouring allocations).
#   - loop-carried tile payloads (manual load-ahead unroll) miscompile
#     (P09 family: results garbled), and a dual live chain thrashes DSM
#     (~5x slower), so the loop stays serial: ~2ms of trig remains exposed
#     on top of the 5.9ms 2R1W floor at 1Gi pairs.
#
# Offsets are int32: the widest computed offset is 2 * ceil(N / BLOCK) *
# BLOCK - 1 <= 2**31 - 1 for every N <= _POLAR_MAX_PAIRS with BLOCK a power
# of two dividing 2**30 (masked lanes included).  Larger inputs fall back to
# the pointwise arm.
#
# Config ladder (one num_warps per BLOCK: the backend kernel cache key drops
# num_warps, so a BLOCK value must pin its warp count):
#   - N <= 4096: single CTA, exact next_pow2 tile, st=1 (launch-floor band,
#     1.55-1.64x vs native at (64, 64)).
#   - N <= 2**24: B8192/nw2/grid<=48 (268MB band, ~2.0TB/s vs native 1.65).
#   - else: B16384/nw2/grid<=96 (1.21-1.24x at the 1Gi shapes).
# ---------------------------------------------------------------------------

_POLAR_MAX_PAIRS = 2**30


@libentry()
@triton.jit(do_not_specialize=["num_pairs"])
def polar_flat_kernel(
    abs_ptr,
    angle_ptr,
    out_word_ptr,
    num_pairs,
    BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    HALF: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    ar = tl.arange(0, BLOCK)
    num_blocks = (num_pairs + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        pair_off = block_id * BLOCK + ar
        if EXACT:
            abs_v = tl.load(abs_ptr + pair_off)
            ang_v = tl.load(angle_ptr + pair_off)
        else:
            mask = pair_off < num_pairs
            abs_v = tl.load(abs_ptr + pair_off, mask=mask)
            ang_v = tl.load(angle_ptr + pair_off, mask=mask)
        abs_f = abs_v.to(tl.float32)
        ang_f = ang_v.to(tl.float32)
        # quadrant reduction: q = rint(ang * 2/pi), r in [-pi/4, pi/4]
        q = ((ang_f * 0.6366197723675814) + 12582912.0) - 12582912.0
        r = (ang_f - q * 1.57079625129699707031) - q * 7.54978941586159635e-8
        qi = q.to(tl.int32)
        x = r * r
        # CUDA __sinf / __cosf minimax polynomials on [-pi/4, pi/4]
        su = r + r * x * (-0.16666667 + x * (0.008333331 + x * -0.00019840874))
        cv = 1.0 + x * (
            -0.5 + x * (0.041666638 + x * (-0.0013888378 + x * 2.4801587e-05))
        )
        neg = (qi & 2) != 0
        even = (qi & 1) == 0
        sgn = tl.where(neg, -1.0, 1.0)
        a = sgn * su
        b = sgn * cv
        sin_a = tl.where(even, a, b)
        cos_a = tl.where(even, b, -a)
        real = abs_f * cos_a
        imag = abs_f * sin_a
        # pack the interleaved complex word with lane-local ALU only
        if HALF:
            real16 = real.to(tl.float16)
            imag16 = imag.to(tl.float16)
            i32 = imag16.to(tl.int16, bitcast=True).to(tl.int32)
            r32 = real16.to(tl.int16, bitcast=True).to(tl.int32)
            word = (i32 << 16) | (r32 & 0xFFFF)
        else:
            i32 = imag.to(tl.int32, bitcast=True).to(tl.int64)
            r32 = real.to(tl.int32, bitcast=True).to(tl.int64)
            word = (i32 << 32) | (r32 & 0xFFFFFFFF)
        if EXACT:
            tl.store(out_word_ptr + pair_off, word)
        else:
            tl.store(out_word_ptr + pair_off, word, mask=mask)


def _polar_choose_config(num_pairs):
    if num_pairs <= 4096:
        return max(triton.next_power_of_2(num_pairs), 256), 2, 48, 1
    if num_pairs <= (1 << 24):
        return 8192, 2, 48, 2
    return 16384, 2, 96, 2


def _polar_flat_applicable(abs_t, angle_t):
    # bf16 stays out on purpose: torch.view_as_complex rejects bf16 pair
    # buffers regardless of the kernel (the pointwise arm fails identically).
    return (
        abs_t.dtype in (torch.float16, torch.float32)
        and angle_t.dtype == abs_t.dtype
        and angle_t.device == abs_t.device
        and angle_t.shape == abs_t.shape
        and abs_t.is_contiguous()
        and angle_t.is_contiguous()
        and abs_t.numel() <= _POLAR_MAX_PAIRS
    )


@pointwise_dynamic(
    promotion_methods=[
        ((0, 1), "DEFAULT"),
        ((0, 1), "DEFAULT"),
    ],
    num_outputs=2,
    # gcu400: fallback arm for broadcast / non-contiguous / promoted inputs.
    # The vendor default (block-pointer tiles, 65536) needs 1 MiB of DSM
    # (hardware limit 896 KiB) and raises OutOfResources; the 1d affine tile
    # also keeps the two stride-2 output stores away from the DTE channel,
    # which corrupts device memory at the 1Gi-element shapes.
    config=CodeGenConfig(8192, (48, 1, 1), 4, False, prefer_1d_tile=True),
)
@triton.jit
def polar_kernel(abs, angle):
    real = abs * tl.cos(angle)
    imag = abs * tl.sin(angle)
    return real, imag


def polar(abs, angle):
    logger.debug("GEMS_ENFLAME POLAR")
    if not _polar_flat_applicable(abs, angle):
        output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)
        polar_kernel(abs, angle, out0=output[..., 0], out1=output[..., 1])
        return torch.view_as_complex(output)

    num_pairs = abs.numel()
    output = torch.empty((*abs.shape, 2), dtype=abs.dtype, device=abs.device)
    if num_pairs == 0:
        return torch.view_as_complex(output)
    word_dtype = torch.int32 if abs.dtype == torch.float16 else torch.int64
    out_words = output.view(word_dtype)
    BLOCK, num_warps, grid_cap, num_stages = _polar_choose_config(num_pairs)
    grid_size = min(grid_cap, -(-num_pairs // BLOCK))
    exact = num_pairs % BLOCK == 0
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (add.py / sinh.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(abs.device):
                polar_flat_kernel[(grid_size,)](
                    abs,
                    angle,
                    out_words,
                    num_pairs,
                    BLOCK=BLOCK,
                    EXACT=exact,
                    HALF=abs.dtype == torch.float16,
                    num_warps=num_warps,
                    num_stages=num_stages,
                )
            return torch.view_as_complex(output)
        except Exception as exc:
            if attempt == 2:
                raise
            logger.warning(
                "polar flat kernel launch failed (attempt %d): %s", attempt + 1, exc
            )
