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
                  num_warps=4 (4096 fp16 3.14us [4.16]; 1M fp16 6.36 [6.44])
* n <= 2**24   -> persistent grid-stride, grid=min(cdiv,48), num_warps=4
                  (16.7M fp16 25.0us [24.4])
* n > 2**24    -> same, num_warps=2 (512M fp16 550us [548] = 2.93TB/s;
                  fp32 BLOCK=32768 988us [987]; bf16 547us [546])
* BLOCK by element size: 65536 (es<=2) / 32768 (es==4), DSM peak
  2*BLOCK*es + BLOCK <= 320KB < 448KB/program (not_equal-validated budget).

Vetoed by probe: int64 wide-channel load + packed int32 bool store (shift/
and/or chain costs more than the bandwidth gain: 1040us vs 550us @512M fp16),
one-tile-per-CTA beyond ~2**20 (713us @512M), grid cap 96/192 (worse than 48),
num_stages 2/4 (no effect).

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
# Keeps (n + BLOCK - 1) inside int32 for every BLOCK in use (max 65536).
_I32_SAFE_MAX = 2**31 - 2**16


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
    block_cap = _big_block(A.element_size())
    if n <= _MAX_FLAT_N:
        block = min(max(16, triton.next_power_of_2(n)), block_cap)
        grid = (triton.cdiv(n, block),)
        num_warps = 4
        kernel = _signbit_flat_kernel
    else:
        block = block_cap
        grid = (min(triton.cdiv(n, block), _NUM_SPC),)
        num_warps = 4 if n <= _MID_N else 2
        kernel = _signbit_pers_kernel
    kernel[grid](A, out, n, BLOCK=block, EVEN=(n % block == 0), num_warps=num_warps)
    return True


def _fast_ok(A):
    return A.dtype in _FAST_DTYPES and A.is_contiguous() and A.numel() <= _I32_SAFE_MAX


def signbit(A):
    logger.debug("GEMS_ENFLAME_GCU400 SIGNBIT")
    if _fast_ok(A):
        out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
        _launch(A, out)
        return out
    return signbit_func(A)


def signbit_out(A, *, out=None):
    logger.debug("GEMS_ENFLAME_GCU400 SIGNBIT_OUT")
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
