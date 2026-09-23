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

from flag_gems.ops.special_round import special_round as default_special_round

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@triton.jit
def _bf16_rne_f32(x):
    # fp32 -> bf16 (RNE on the 7-bit mantissa) returned directly as an fp32 value
    # with the low 16 mantissa bits zeroed. Replicates torch_musa's cast behavior;
    # the native fp32->bf16->fp32 round trip is folded by the compiler, so the
    # reference's observable bf16 intermediate rounding must be done manually.
    bits = x.to(tl.int32, bitcast=True)
    t = bits + 0x7FFF + ((bits >> 16) & 1)
    return (t & -65536).to(tl.float32, bitcast=True)


@triton.jit
def _round_dec_f32(A, Out, n_elements, factor, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(A + offs, mask=mask)
    y = x * factor
    y = tl.extra.musa.libdevice.rint(y)
    y = y / factor
    tl.store(Out + offs, y, mask=mask)


@triton.jit
def _round_dec_f32_d0(A, Out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(A + offs, mask=mask)
    y = tl.extra.musa.libdevice.rint(x)
    tl.store(Out + offs, y, mask=mask)


@triton.jit
def _round_dec_f64(A, Out, n_elements, factor, NEG: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(A + offs, mask=mask)
    if NEG:
        # reference scales by division for negative decimals: round(x / 10^|d|) * 10^|d|
        y = x / factor
        y = tl.extra.musa.libdevice.rint(y)
        y = y * factor
    else:
        y = x * factor
        y = tl.extra.musa.libdevice.rint(y)
        y = y / factor
    tl.store(Out + offs, y, mask=mask)


@triton.jit
def _round_dec_f16(A, Out, n_elements, factor, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(A + offs, mask=mask)
    y = x.to(tl.float32) * factor
    y = y.to(x.dtype)
    y = y.to(tl.float32)
    y = tl.extra.musa.libdevice.rint(y)
    y = y / factor
    tl.store(Out + offs, y.to(x.dtype), mask=mask)


@triton.jit
def _round_dec_lowp_d0(A, Out, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    # streaming access hints: the tensor is read and written exactly once
    x = tl.load(A + offs, mask=mask, eviction_policy="evict_first")
    y = tl.extra.musa.libdevice.rint(x.to(tl.float32))
    tl.store(
        Out + offs, y, mask=mask, cache_modifier=".cs"
    )  # store auto-cast to fp16/bf16 is RNE


@triton.jit
def _round_dec_bf16(A, Out, n_elements, factor, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(A + offs, mask=mask)
    y = x.to(tl.float32) * factor
    y = _bf16_rne_f32(y)
    y = tl.extra.musa.libdevice.rint(y)
    y = y / factor
    tl.store(Out + offs, y, mask=mask)  # store auto-cast to bf16 is RNE7


_BLOCK = 1024
_SMALL = 1 << 18  # tier boundary: tiny shapes (launch-overhead bound)
_LARGE = 1 << 26  # tier boundary: bandwidth-saturated shapes


def _specialized_special_round(A, *, decimals=0):
    n = A.numel()
    out = torch.empty_like(A)
    if n == 0:
        return out
    dt = A.dtype
    f = 10.0**decimals
    grid = (triton.cdiv(n, _BLOCK),)
    if dt == torch.float32:
        if n >= _LARGE:
            # most blocks, most warps: best measured config at 67M-1B elements
            if decimals == 0:
                _round_dec_f32_d0[(triton.cdiv(n, 512),)](
                    A, out, n, BLOCK=512, num_warps=8
                )
            else:
                _round_dec_f32[(triton.cdiv(n, 512),)](
                    A, out, n, float(f), BLOCK=512, num_warps=8
                )
        elif n < _SMALL:
            # tiny shapes: small blocks keep launch/occupancy efficient
            if decimals == 0:
                _round_dec_f32_d0[(triton.cdiv(n, 256),)](
                    A, out, n, BLOCK=256, num_warps=4
                )
            else:
                _round_dec_f32[(triton.cdiv(n, 256),)](
                    A, out, n, float(f), BLOCK=256, num_warps=4
                )
        else:
            if decimals == 0:
                _round_dec_f32_d0[grid](A, out, n, BLOCK=_BLOCK, num_warps=4)
            else:
                _round_dec_f32[grid](A, out, n, float(f), BLOCK=_BLOCK, num_warps=4)
    elif dt == torch.float64:
        if decimals >= 0:
            _round_dec_f64[grid](A, out, n, f, NEG=False, BLOCK=_BLOCK, num_warps=4)
        else:
            _round_dec_f64[grid](
                A, out, n, 10.0 ** (-decimals), NEG=True, BLOCK=_BLOCK, num_warps=4
            )
    elif dt in (torch.float16, torch.bfloat16):
        if decimals == 0:
            if n < _SMALL:
                _round_dec_lowp_d0[(triton.cdiv(n, 512),)](
                    A, out, n, BLOCK=512, num_warps=4
                )
            else:
                _round_dec_lowp_d0[grid](A, out, n, BLOCK=_BLOCK, num_warps=4)
        elif dt == torch.float16:
            if decimals >= 5:
                f = float(torch.tensor(f, dtype=torch.float16).item())
            _round_dec_f16[grid](A, out, n, float(f), BLOCK=_BLOCK, num_warps=4)
        else:
            if decimals >= 4:
                f = float(torch.tensor(f, dtype=torch.bfloat16).item())
            _round_dec_bf16[grid](A, out, n, float(f), BLOCK=_BLOCK, num_warps=4)
    else:
        raise TypeError(f"special_round: unsupported dtype {dt}")
    return out


def special_round(A, *, decimals=0):
    logger.debug("GEMS_MTHREADS SPECIAL_ROUND")
    if (
        isinstance(A, torch.Tensor)
        and A.device.type == "musa"
        and A.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_special_round(A, decimals=decimals)
    return default_special_round(A, decimals=decimals)
