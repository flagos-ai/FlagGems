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

"""Element-wise greater-than, GCU400-optimized (sibling of ``greater_equal``).

The generic ``gt_func`` pointwise path caps the launch grid at 12 CTAs and
pays an fp32 up-cast compare; the flat grid-stride kernels below replace it
for contiguous same-dtype float inputs.  2026-11-16 tensor rework (probe-
validated, do_bench kernel median w1000/r100, in-session A/B vs ``torch.gt``;
re-landed 2026-11-16 after a sealing conflict dropped it from the branch):

* interleaved grid-stride block loop (``tl.range(pid, num_blocks, num_pids)``)
  with a mask-free constexpr ``EXACT`` arm when ``n % BLOCK == 0``; per-CTA
  contiguous spans (the previous COV/CHUNK loop form) lose up to 5% to DRAM
  page/bank interleaving at 1Gi.
* grid capped at 48 (16 SIPs x 3): larger grids re-measured slower.
* tensor ladder (2 loads + bool store): 16-bit <=2^17 8192/nw4, <=2^23
  16384/nw4, else 65536/nw2 (no soft-pipeline: the ns3 gain was card-state
  dependent; ns1/ns3 measured at parity in the re-land window and ns1 matches
  the ``ge`` twin); fp32 <=2^21 8192/nw4, <=2^27
  32768/nw2, else 16384/nw1 (1Gi 3929 -> 3704us, at the 9B/elem floor);
  tiny <=4096 single-CTA next_pow2/nw1 (3.19us, launch floor 2.43us).
* physical ceiling (probe floors, 2R1W-1B-out patterns): 1Gi fp16 floor
  1933us == native 1932us; 16.7M fp16 floor ~36.9us vs native 40.4;
  1Gi fp32 floor 3704 vs native 3809.  Large shapes are at the wall with
  native, so the mean speedup is bounded by the small/mid bands.
* comparisons stay in the native input dtype (exact for same-dtype floats);
  the scalar overload casts the Python scalar to ``A.dtype`` before
  comparing, matching torch TensorIterator semantics (the public fp32
  up-cast kernel deviates from torch on fp16 round-down scalars like 0.1).

Everything the fast guards reject (broadcast shapes, dtype promotion,
non-contiguous layouts, non-float dtypes, int32-unsafe sizes) keeps the
``gt_func`` / ``gt_func_scalar`` fallbacks unchanged.

Scalar ladder (2026-11-16 second-round probe, kernel do_bench w1000/r100,
in-session A/B vs torch native, L600):
  n < 2**14              -> single-CTA next_pow2 kernel, nw=4 (16-bit) / 8 (f32)
                           [64,64] f32 4.1 -> 3.2 us; the old flat-16384 arm
                           held 128 fp32 lanes/thread (512 B register spill
                           serialization), nw=8 caps at 256 B/lane.
  2**14..2**20           -> flat BLOCK=16384, nw=4, EVEN arm when n%block==0
  2**20..2**22 (f16/bf16) -> gs48 BLOCK=16384, nw=4 when n%16384==0
                           (64-256 CTA cdiv grids pay the wave-quantization
                           tax; 2**20: 6.3 -> 5.4 us) else flat 16384 masked
  2**20..2**22 (f32)     -> flat BLOCK=65536, nw=4 (2**20: 9.6 -> 7.3 us)
  2**22..2**26 (f32)     -> gs48 BLOCK=65536, nw=4 when n%65536==0
                           (2**24: 38.5 -> 36.8 us, 2.04 -> 2.28 TB/s) else
                           flat 131072 masked
  n >= 2**26 (f32)       -> loop grid<=64: COV=131072*cdiv(n, 64*131072),
                           CHUNK=131072, nw=8, st=1 (unchanged: med-best at
                           2**30, 1949 us vs gs 65536/nw1 1959-1976)
  2**22..2**26 (f16/bf16) -> gs48 BLOCK=131072, nw=4 when n%131072==0
                           (2**23: 15.9 -> 15.3, 2**24: 23.9 -> 23.3 us)
                           else flat 262144 masked
  n >= 2**26 (f16/bf16)   -> flat BLOCK=262144, nw=4 with the EVEN arm
                           (paired A/B: the gs arm loses 0.6-1.2% at
                           2**27/2**28 -- at the ~3.0 TB/s wall the larger
                           flat tile wins)
Odd / non-multiple sizes keep the previous flat masked configs, so only
exact-multiple shapes move to the interleaved grid-stride form.
``gt_func`` / ``gt_func_scalar`` (public pointwise kernels) remain the
fallback, so untuned corners keep their previous behaviour.

The scalar f32 loop-64 tier (2026-11-16 probe): the flat 131072 kernel runs
8192 CTAs at n=2**30 and only ~2600 GB/s (inter-CTA DDR contention), while a
<=64-CTA coverage loop streams ~2720 GB/s at parity with torch native
(x1.00 vs x0.95; 32-CTA collapses to 2336 GB/s, so the grid target is 64).
n=2**24 keeps the flat kernel (64-CTA loop loses ~0.4% there); f16/bf16 stay
flat at all sizes (measured at the ~2.95-3.0 TB/s wall, loop gains nothing).
"""

import logging
import time

import torch
import triton
import triton.language as tl

from flag_gems.ops.gt import gt_func
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 16
_GRID_CAP = NUM_SIPS * 3
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MID_N = 2**22
_INT32_SAFE_LIMIT = 0x7FFFFFFF
_TINY_N = 4096
_SCALAR_TINY_N = 2**14
_SCALAR_WIDE_N = 2**20
_SCALAR_F16_GS_HI = 2**26
_SCALAR_GS_CAP = 48
_F32_LOOP_N = 2**26
_F32_LOOP_CHUNK = 131072
_F32_LOOP_GRID = 64


def _launch_with_retry(launch, attempts=4, delay=1.0):
    """Retry flaky cold compiles (P02 toolchain-segv family); re-raise on exhaust."""
    for i in range(attempts):
        try:
            return launch()
        except Exception:
            if i == attempts - 1:
                raise
            time.sleep(delay)


@libentry()
@triton.jit(do_not_specialize=["N_total"])
def gt_flat_kernel(
    A, B, OUT, N_total, BLOCK: tl.constexpr, EXACT: tl.constexpr, NS: tl.constexpr
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = (N_total + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        if EXACT:
            x = tl.load(A + off)
            y = tl.load(B + off)
            r = x > y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty))
        else:
            mask = off < N_total
            x = tl.load(A + off, mask=mask, other=0)
            y = tl.load(B + off, mask=mask, other=0)
            r = x > y
            tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _gt_cmp_scalar_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, EVEN_N: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN_N:
        x = tl.load(A + off)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))
    else:
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gt_cmp_scalar_gs_kernel(A, OUT, N, Y, BLOCK: tl.constexpr, NS: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    arange = tl.arange(0, BLOCK)
    num_blocks = N // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids, num_stages=NS):
        off = block_id * BLOCK + arange
        x = tl.load(A + off)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty))


@triton.jit
def _gt_cmp_scalar_loop_kernel(A, OUT, N, Y, COV: tl.constexpr, CHUNK: tl.constexpr):
    base = tl.program_id(0) * COV
    for j in range(0, COV // CHUNK):
        off = base + j * CHUNK + tl.arange(0, CHUNK)
        m = off < N
        x = tl.load(A + off, mask=m, other=0)
        r = x > Y.to(x.dtype)
        tl.store(OUT + off, r.to(OUT.dtype.element_ty), mask=m)


def _gt_pair_ok(A, B):
    return (
        isinstance(B, torch.Tensor)
        and A.dtype == B.dtype
        and A.dtype in _FLOAT_DTYPES
        and A.shape == B.shape
        and A.is_contiguous()
        and B.is_contiguous()
    )


def _gt_pair_config(n, itemsize):
    if n <= _TINY_N:
        # fp32 needs nw2 here: one warp holding 128 fp32 lanes per operand
        # serializes (4096/nw1 = 6.4us vs 3.77us at nw2); 16-bit is fine at nw1.
        return max(triton.next_power_of_2(n), 256), (1 if itemsize == 2 else 2), 1
    if itemsize == 2:
        if n <= (1 << 17):
            return 8192, 4, 1
        if n <= (1 << 23):
            return 16384, 4, 1
        # ns=1 vs ns=3 paired probe (2026-11-16 re-land window): parity at
        # 2**24/2**27/2**30 (ns3 within noise); ns1 matches the ge twin and
        # never loses the paired probe across windows.
        return 65536, 2, 1
    if n <= (1 << 21):
        return 8192, 4, 1
    if n <= (1 << 27):
        return 32768, 2, 1
    return 16384, 1, 1


def _gt_launch(kernel, grid, args, block, num_warps, **constexprs):
    # The external llvm converter intermittently fails on cold compiles
    # (nondeterministic; converterResult=-2 family, see P02); retrying the
    # launch recompiles and succeeds (add.py / sinh.py precedent).
    for attempt in range(3):
        try:
            with torch_device_fn.device(args[0].device):
                kernel[(grid,)](
                    *args,
                    BLOCK=block,
                    num_warps=num_warps,
                    num_stages=1,
                    **constexprs,
                )
            return
        except Exception:
            if attempt == 2:
                raise
            logger.warning(
                "greater flat kernel launch failed (attempt %d)",
                attempt + 1,
                exc_info=True,
            )


def _gt_pair_run(A, B, out):
    """Launch the tuned comparison kernel; False falls back to pointwise."""
    n = A.numel()
    if n == 0:
        return False
    block, num_warps, num_stages = _gt_pair_config(n, A.element_size())
    if n > _INT32_SAFE_LIMIT - block:
        return False
    grid = min(-(-n // block), _GRID_CAP)
    _gt_launch(
        gt_flat_kernel,
        grid,
        (A, B, out, n),
        block,
        num_warps,
        EXACT=n % block == 0,
        NS=num_stages,
    )
    return True


def _gt_scalar_fast(A, B, out=None):
    """Launch the tuned scalar-compare kernel into ``out`` (allocated if None).

    Returns the output tensor, or None when the fast path does not apply.
    Exact-multiple sizes take the probe-validated interleaved grid-stride
    arms; odd sizes keep the previous flat masked configs.
    """
    if not (
        isinstance(B, (int, float)) and A.dtype in _FLOAT_DTYPES and A.is_contiguous()
    ):
        return None
    n = A.numel()
    if n <= 0:
        return None
    y = float(B)
    is_f32 = A.dtype == torch.float32
    if out is None:
        out = torch.empty_like(A, dtype=torch.bool)

    if n < _SCALAR_TINY_N:
        block = 1 << (n - 1).bit_length() if n > 1 else 1
        if block < 32:
            block = 32
        _launch_with_retry(
            lambda: _gt_cmp_scalar_kernel[(1,)](
                A,
                out,
                n,
                y,
                BLOCK=block,
                EVEN_N=(n % block == 0),
                num_warps=(8 if is_f32 else 4),
            )
        )
        return out

    if n < _SCALAR_WIDE_N:
        block = 16384
    elif not is_f32 and n < _MID_N:
        if n % 16384 == 0:
            _launch_with_retry(
                lambda: _gt_cmp_scalar_gs_kernel[(_SCALAR_GS_CAP,)](
                    A, out, n, y, BLOCK=16384, NS=1, num_warps=4, num_stages=1
                )
            )
            return out
        block = 16384
    elif is_f32 and n < _MID_N:
        block = 65536
    elif is_f32:
        if n >= _F32_LOOP_N:
            cov = _F32_LOOP_CHUNK * -(-n // (_F32_LOOP_GRID * _F32_LOOP_CHUNK))
            if n <= _INT32_SAFE_LIMIT - cov:
                _launch_with_retry(
                    lambda: _gt_cmp_scalar_loop_kernel[(-(-n // cov),)](
                        A,
                        out,
                        n,
                        y,
                        COV=cov,
                        CHUNK=_F32_LOOP_CHUNK,
                        num_warps=8,
                        num_stages=1,
                    )
                )
                return out
            return None
        if n % 65536 == 0 and n <= _INT32_SAFE_LIMIT - 65536:
            _launch_with_retry(
                lambda: _gt_cmp_scalar_gs_kernel[(_SCALAR_GS_CAP,)](
                    A, out, n, y, BLOCK=65536, NS=1, num_warps=4, num_stages=1
                )
            )
            return out
        block = 131072
    else:
        if (
            n < _SCALAR_F16_GS_HI
            and n % 131072 == 0
            and n <= _INT32_SAFE_LIMIT - 131072
        ):
            _launch_with_retry(
                lambda: _gt_cmp_scalar_gs_kernel[(_SCALAR_GS_CAP,)](
                    A, out, n, y, BLOCK=131072, NS=1, num_warps=4, num_stages=1
                )
            )
            return out
        block = 262144

    if n > _INT32_SAFE_LIMIT - block:
        return None
    _launch_with_retry(
        lambda: _gt_cmp_scalar_kernel[(-(-n // block),)](
            A, out, n, y, BLOCK=block, EVEN_N=(n % block == 0), num_warps=4
        )
    )
    return out


def greater(A, B):
    logger.debug("GEMS_ENFLAME GREATER")
    if _gt_pair_ok(A, B):
        out = torch.empty_like(A, dtype=torch.bool)
        if _gt_pair_run(A, B, out):
            return out
    return gt_func(A, B)


def greater_out(A, B, *, out=None):
    logger.debug("GEMS_ENFLAME GREATER_OUT")
    if out is None:
        return greater(A, B)
    if (
        _gt_pair_ok(A, B)
        and out.dtype == torch.bool
        and out.shape == A.shape
        and out.is_contiguous()
    ):
        if _gt_pair_run(A, B, out):
            return out
    gt_func(A, B, out0=out)
    return out
