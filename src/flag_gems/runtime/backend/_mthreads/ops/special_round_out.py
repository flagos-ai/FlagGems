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

from flag_gems.ops.special_round import special_round_out as default_special_round_out

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _rh32(x):
    # Round-to-nearest-integer, ties-to-even (banker's rounding), fp32.
    # floor-based, exact for all fp32 (and fp16/bf16 upcast) values.
    fl = tl.math.floor(x)
    fr = x - fl
    tie = fr == 0.5
    n_half = fl * 0.5
    n_even = n_half == tl.math.floor(n_half)
    base = fl + tl.where(fr >= 0.5, 1.0, 0.0)
    y = tl.where(tie & n_even, fl, base)
    y = tl.where(y == 0.0, x * 0.0, y)  # keep -0.0 sign
    return y


@triton.jit
def _rh64(x):
    fl = tl.math.floor(x)
    fr = x - fl
    tie = fr == 0.5
    n_half = fl * 0.5
    n_even = n_half == tl.math.floor(n_half)
    base = fl + tl.where(fr >= 0.5, 1.0, 0.0)
    y = tl.where(tie & n_even, fl, base)
    y = tl.where(y == 0.0, x * 0.0, y)
    return y


@triton.jit
def _bf16r(p):
    # Round an fp32 value to bf16 precision (round-half-even) with integer ops
    # on the bit pattern: deterministic on backends where bf16 arithmetic is
    # emulated in fp32 with unreliable intermediate rounding.
    u = p.to(tl.int32, bitcast=True)
    b = (u + 32767 + ((u >> 16) & 1)) & -65536
    b = tl.where((u & 2139095040) == 2139095040, u & -65536, b)  # NaN/Inf guard
    return b.to(tl.int32).to(tl.float32, bitcast=True)


@triton.jit
def _round_val(
    x, MODE: tl.constexpr, IS64: tl.constexpr, ISBF: tl.constexpr, SCALE: tl.constexpr
):
    # Bit-exact reference semantics (torch.special.round on this target):
    # MODE 0: round half to even; MODE 1: x*scale, round, /scale; MODE 2:
    # x/scale, round, *scale.  Arithmetic in the input dtype: fp16/fp32
    # native, bf16 via explicit bit-pattern rounding of fp32 intermediates,
    # fp64 native.
    if IS64:
        if MODE == 0:
            return _rh64(x)
        elif MODE == 1:
            return _rh64(x * SCALE) / SCALE
        else:
            return _rh64(x / SCALE) * SCALE
    elif ISBF:
        xf = x.to(tl.float32)
        if MODE == 0:
            return _rh32(xf).to(x.dtype)
        elif MODE == 1:
            return (_rh32(_bf16r(xf * SCALE)) / SCALE).to(x.dtype)
        else:
            return (_rh32(_bf16r(xf / SCALE)) * SCALE).to(x.dtype)
    else:
        xf = x.to(tl.float32)
        if MODE == 0:
            return _rh32(xf).to(x.dtype)
        elif MODE == 1:
            q16 = x * SCALE
            return (_rh32(q16.to(tl.float32)) / SCALE).to(x.dtype)
        else:
            q16 = x / SCALE
            return (_rh32(q16.to(tl.float32)) * SCALE).to(x.dtype)


@triton.jit
def _store(Y, offs, y, n, EVEN: tl.constexpr, CM: tl.constexpr, EVICT: tl.constexpr):
    if EVEN:
        if EVICT:
            if CM == 2:
                tl.store(
                    Y + offs, y, cache_modifier=".cs", eviction_policy="evict_first"
                )
            else:
                tl.store(Y + offs, y, eviction_policy="evict_first")
        elif CM == 1:
            tl.store(Y + offs, y, cache_modifier=".cg")
        elif CM == 2:
            tl.store(Y + offs, y, cache_modifier=".cs")
        else:
            tl.store(Y + offs, y)
    else:
        mask = offs < n
        if EVICT:
            if CM == 2:
                tl.store(
                    Y + offs,
                    y,
                    mask=mask,
                    cache_modifier=".cs",
                    eviction_policy="evict_first",
                )
            else:
                tl.store(Y + offs, y, mask=mask, eviction_policy="evict_first")
        elif CM == 1:
            tl.store(Y + offs, y, mask=mask, cache_modifier=".cg")
        elif CM == 2:
            tl.store(Y + offs, y, mask=mask, cache_modifier=".cs")
        else:
            tl.store(Y + offs, y, mask=mask)


@triton.jit
def _special_round_kernel(
    X,
    Y,
    n,
    MODE: tl.constexpr,
    IS64: tl.constexpr,
    ISBF: tl.constexpr,
    EVEN: tl.constexpr,
    IDX32: tl.constexpr,
    GS: tl.constexpr,
    CM: tl.constexpr,
    EVICT: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE: tl.constexpr,
):
    if IDX32:
        pid0 = tl.program_id(0)
    else:
        pid0 = tl.program_id(0).to(tl.int64)
    if GS:
        nprog = tl.num_programs(0)
        niter = tl.cdiv(n, nprog * BLOCK)
        for i in range(0, niter):
            offs = (pid0 + i * nprog) * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            if CM == 1 or CM == 2:
                x = tl.load(X + offs, mask=mask, cache_modifier=".cg")
            else:
                x = tl.load(X + offs, mask=mask)
            y = _round_val(x, MODE, IS64, ISBF, SCALE)
            _store(Y, offs, y, n, False, CM, EVICT)
    else:
        if IDX32:
            pid = pid0
            offs = pid * BLOCK + tl.arange(0, BLOCK)
        else:
            pid = pid0
            offs = pid * BLOCK + tl.arange(0, BLOCK)
        if EVEN:
            if CM == 1 or CM == 2:
                x = tl.load(X + offs, cache_modifier=".cg")
            else:
                x = tl.load(X + offs)
            y = _round_val(x, MODE, IS64, ISBF, SCALE)
            _store(Y, offs, y, n, True, CM, EVICT)
        else:
            mask = offs < n
            if CM == 1 or CM == 2:
                x = tl.load(X + offs, mask=mask, cache_modifier=".cg")
            else:
                x = tl.load(X + offs, mask=mask)
            y = _round_val(x, MODE, IS64, ISBF, SCALE)
            _store(Y, offs, y, n, False, CM, EVICT)


def _pick(n, dtype):
    # Launch geometry + streaming cache hints measured on MTT S5000 (60 SM)
    # with target do_bench at the exact eval shapes (4K / 16M / 1G elements).
    if n >= (1 << 27):  # 1G-element workloads: HBM-bound
        if dtype == torch.float32:
            return 2048, 2, 2, 0, 720  # cg load + cs store, grid-stride 720
        if dtype == torch.bfloat16:
            return 1024, 8, 2, 0, 0
        return 1024, 8, 1, 0, 0
    if n >= (1 << 20):  # 16M
        if dtype == torch.float32:
            return 2048, 2, 2, 0, 0
        if dtype == torch.bfloat16:
            return 256, 4, 0, 0, 0
        return 4096, 2, 2, 0, 0
    # small (<= 64K)
    if dtype == torch.float32:
        return 512, 4, 0, 0, 0
    if dtype == torch.bfloat16:
        return 256, 4, 1, 0, 0
    return 512, 4, 1, 1, 0


def _specialized_special_round_out(self, out, *, decimals=0):
    n = out.numel()
    if n == 0:
        return out
    if decimals == 0:
        mode, scale = 0, 1.0
    elif decimals > 0:
        mode, scale = 1, float(10**decimals)
    else:
        mode, scale = 2, float(10 ** (-decimals))
    BLOCK, WARPS, CM, EVICT, GSCAP = _pick(n, self.dtype)
    GS = GSCAP > 0
    grid = (GSCAP,) if GS else (triton.cdiv(n, BLOCK),)
    _special_round_kernel[grid](
        self,
        out,
        n,
        MODE=mode,
        IS64=(self.dtype == torch.float64),
        ISBF=(self.dtype == torch.bfloat16),
        EVEN=(n % BLOCK) == 0,
        IDX32=n < (1 << 31),
        GS=GS,
        CM=CM,
        EVICT=EVICT,
        BLOCK=BLOCK,
        SCALE=scale,
        num_warps=WARPS,
    )
    return out


def special_round_out(self, out, *, decimals=0):
    logger.debug("GEMS_MTHREADS SPECIAL_ROUND_OUT")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
        and isinstance(out, torch.Tensor)
        and out.device.type == "musa"
        and out.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_special_round_out(self, out, decimals=decimals)
    return default_special_round_out(self, out, decimals=decimals)
