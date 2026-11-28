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

"""signbit, GCU400-optimized flat 1D kernel (final form of the grid-override v1).

Root cause (probe, 2026-11-13, job j20261113024326_43ee59d2): the previous
candidate only raised ``CodeGenConfig.max_grid_size`` to ``(1024, 1, 1)`` and
kept the generic ``pointwise_dynamic`` block-pointer/t4096 codegen — 512M fp16
1.656ms (0.91TB/s, 0.325x torch) and ``signbit_out`` still ran the plain
12-CTA config at 5.94ms (0.091x). Same family pathology as not_equal /
greater / isposinf.

This file replaces both entry points with hand-written flat kernels
(bitcast + sign test per dtype, int32 offsets, ``do_not_specialize=["N"]``,
EVEN no-mask fast path) and probe-validated tiers (do_bench median
w1000/r100, vs torch native in brackets):

* n <= 2**20   -> one-tile-per-CTA flat, BLOCK=min(np2(n), 65536|32768),
                  num_warps=4 (4096 fp16 3.14us [4.17]; 1M fp16 6.36 [6.44])
* n > 2**20    -> persistent grid-stride, grid=min(cdiv,48) (16.7M and 512M
                  core shapes; interleaved A/B 2026-11-18):
  - es<=2, n >= 48*131072: BLOCK=131072, num_stages=1, nw=4 (n<=2**24) /
    nw=2 above (16.7M fp16 24.96us [24.99 native], bf16 24.92 [24.96];
    512M bf16 546.5 [549.8])
  - es==4, n >= 48*65536:  BLOCK=65536, num_stages=1, nw=4 / nw=2
    (16.7M fp32 39.43us [40.06 native]; 512M fp32 1034 [1035])
  - smaller persistent n keeps the old arms: BLOCK=65536 (es<=2) /
    32768 (es==4), default stages, nw=4|2 (unchanged, zero-regression).
  The wide arms rely on the new-stack staging pool: compiled
  ``gcu.dsm_memory_size`` is a fixed 262144B (<448KB, 48-way resident) for
  every BLOCK in {32768,65536,131072} — the old ``2*BLOCK*es+BLOCK`` budget
  was over-conservative (it capped BLOCK at 65536/32768); ``num_stages=1``
  plus the 2x wider tile buys the mid-tier win.

Vetoed by probe: int64 wide-channel load + packed int32 bool store (shift/
and/or chain costs more than the bandwidth gain: 1040us vs 550us @512M fp16),
one-tile-per-CTA beyond ~2**20 (713us @512M; 16.7M flat B=32768 g=512 44.6us
vs persistent 39.2-25.0us), grid cap 96/192 (worse than 48 at every size),
BLOCK<65536 in the persistent tier (16.7M fp16 B=32768/nw4 29.9us, B=16384
30.4us, B=8192 36.3us), grid 24 (786us @512M fp16), nw=1 (672us @512M fp16),
nw=8 mid (25.59 vs 25.54 current), mid-tier num_stages 2 on B=65536
(25.27 vs 25.54 = WITHIN-NOISE), B=131072 with default stages (DSM formula
violation; superseded by the ns=1 wide arm).

Non-contiguous / non-fast-dtype / n >= 2**31-BLOCK inputs fall back to the
production ``signbit_func`` pointwise kernel (previous behaviour, zero
regression); the host only allocates the bool output.
"""

import logging

import torch
import triton
import triton.language as tl

from .signbit import signbit_func

logger = logging.getLogger(__name__)

_FAST_DTYPES = {
    dt
    for dt in (
        getattr(torch, name, None)
        for name in (
            "uint8",
            "int8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "float16",
            "bfloat16",
            "float32",
        )
    )
    if dt is not None
}

_MAX_FLAT_N = 2**20
_MID_N = 2**24
_NUM_SPC = 48
# Keeps (n + BLOCK - 1) inside int32 for every BLOCK in use (max 131072).
_I32_SAFE_MAX = 2**31 - 2**17
# Wide persistent arms (2026-11-18 probe, new stack): the compiler stages the
# elementwise tile through a fixed 256KB pool (gcu.dsm_memory_size = 262144
# for every BLOCK), so num_stages=1 plus a 2x wider BLOCK is affordable and
# wins at the mid tier. Only engaged when all _NUM_SPC slots get whole blocks;
# smaller persistent n keeps the previously validated arms.
_WIDE_ES2_BLOCK = 131072
_WIDE_ES2_MIN_N = _NUM_SPC * _WIDE_ES2_BLOCK
_WIDE_ES4_BLOCK = 65536
_WIDE_ES4_MIN_N = _NUM_SPC * _WIDE_ES4_BLOCK


@triton.jit
def _signbit_body(x):
    if tl.constexpr(x.dtype.is_fp32()):
        return x.to(tl.int32, bitcast=True) < 0
    elif tl.constexpr(x.dtype.is_fp16()):
        return x.to(tl.int16, bitcast=True) < 0
    elif tl.constexpr(x.dtype.is_bf16()):
        return x.to(tl.int16, bitcast=True) < 0
    elif tl.constexpr(x.dtype.is_fp64()):
        return x.to(tl.int64, bitcast=True) < 0
    else:
        return x < 0


@triton.jit(do_not_specialize=["N"])
def _signbit_flat_kernel(X, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        x = tl.load(X + off)
        tl.store(Out + off, _signbit_body(x))
    else:
        mask = off < N
        x = tl.load(X + off, mask=mask)
        tl.store(Out + off, _signbit_body(x), mask=mask)


@triton.jit(do_not_specialize=["N"])
def _signbit_pers_kernel(X, Out, N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)
    num_blocks = (N + BLOCK - 1) // BLOCK
    for block_id in tl.range(pid, num_blocks, num_pids):
        off = block_id * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            x = tl.load(X + off)
            tl.store(Out + off, _signbit_body(x))
        else:
            mask = off < N
            x = tl.load(X + off, mask=mask)
            tl.store(Out + off, _signbit_body(x), mask=mask)


def _big_block(itemsize):
    return 65536 if itemsize <= 2 else 32768


def _launch(A, out):
    n = A.numel()
    if n == 0:
        return True
    es = A.element_size()
    num_stages = None
    if n <= _MAX_FLAT_N:
        block = min(max(16, triton.next_power_of_2(n)), _big_block(es))
        grid = (triton.cdiv(n, block),)
        num_warps = 4
        kernel = _signbit_flat_kernel
    else:
        if es <= 2:
            if n >= _WIDE_ES2_MIN_N:
                block, num_stages = _WIDE_ES2_BLOCK, 1
            else:
                block = 65536
        else:
            if n >= _WIDE_ES4_MIN_N:
                block, num_stages = _WIDE_ES4_BLOCK, 1
            else:
                block = 32768
        grid = (min(triton.cdiv(n, block), _NUM_SPC),)
        num_warps = 4 if n <= _MID_N else 2
        kernel = _signbit_pers_kernel
    kwargs = {} if num_stages is None else {"num_stages": num_stages}
    kernel[grid](
        A, out, n, BLOCK=block, EVEN=(n % block == 0), num_warps=num_warps, **kwargs
    )
    return True


def _fast_ok(A):
    return A.dtype in _FAST_DTYPES and A.is_contiguous() and A.numel() <= _I32_SAFE_MAX


def signbit(A):
    logger.debug("GEMS_ENFLAME SIGNBIT")
    if _fast_ok(A):
        out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
        _launch(A, out)
        return out
    return signbit_func(A)


def signbit_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME SIGNBIT_OUT")
    if out is None:
        return signbit(A)
    if (
        _fast_ok(A)
        and out.dtype is torch.bool
        and out.shape == A.shape
        and out.is_contiguous()
    ):
        if A.numel() == 0:
            return out
        _launch(A, out)
        return out
    signbit_func(A, out0=out)
    return out
