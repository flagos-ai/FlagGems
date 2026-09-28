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

from flag_gems.ops.lcm import lcm_ as default_lcm_

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

_SUPPORTED_DTYPES = {torch.int16, torch.int32}


@triton.jit
def _lcm_step(a, b):
    # one Euclid step, predicated per lane; div-by-zero guarded
    bs = tl.where(b != 0, b, 1)
    r = a % bs
    na = tl.where(b != 0, b, a)
    nb = tl.where(b != 0, r, b)
    return na, nb


@triton.jit
def _lcm_small(
    A,
    B,
    numel,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    UNROLL: tl.constexpr,
    WIDTH: tl.constexpr,
    SIGNED: tl.constexpr,
    MASKED: tl.constexpr,
):
    # WIDTH: 8/16/32 -> gcd computed in uint32; 8/16-bit results truncated on store
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < numel
        x = tl.load(A + offs, mask=mask, other=0)
        y = tl.load(B + offs, mask=mask, other=0)
    else:
        x = tl.load(A + offs)
        y = tl.load(B + offs)
    if WIDTH == 8:
        x = x.to(tl.int32)
        y = y.to(tl.int32)
    elif WIDTH == 16:
        x = x.to(tl.int32)
        y = y.to(tl.int32)
    ux = x.to(tl.uint32)
    uy = y.to(tl.uint32)
    if SIGNED:
        a0 = tl.where(x < 0, 0 - ux, ux)
        b0 = tl.where(y < 0, 0 - uy, uy)
    else:
        a0 = ux
        b0 = uy
    a = a0
    b = b0
    i = 0
    while (tl.max(b) != 0) & (i < CAP):
        for _ in tl.static_range(UNROLL):
            a, b = _lcm_step(a, b)
        i += UNROLL
    g = tl.where(a == 0, 1, a)
    pos = (a0 // g) * b0
    if WIDTH == 8:
        if SIGNED:
            r8 = pos.to(tl.int8)
        else:
            r8 = pos.to(tl.uint8)
        if MASKED:
            tl.store(A + offs, r8, mask=mask)
        else:
            tl.store(A + offs, r8)
    elif WIDTH == 16:
        if SIGNED:
            r16 = pos.to(tl.int16)
        else:
            r16 = pos.to(tl.uint16)
        if MASKED:
            tl.store(A + offs, r16, mask=mask)
        else:
            tl.store(A + offs, r16)
    else:
        if SIGNED:
            p32 = pos.to(tl.int32)
            res = tl.where(p32 < 0, 0 - p32, p32)
        else:
            res = pos
        if MASKED:
            tl.store(A + offs, res, mask=mask)
        else:
            tl.store(A + offs, res)


@triton.jit
def _lcm_u64(
    A,
    B,
    numel,
    BLOCK: tl.constexpr,
    CAP: tl.constexpr,
    UNROLL: tl.constexpr,
    SIGNED: tl.constexpr,
    MASKED: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if MASKED:
        mask = offs < numel
        x = tl.load(A + offs, mask=mask, other=0)
        y = tl.load(B + offs, mask=mask, other=0)
    else:
        x = tl.load(A + offs)
        y = tl.load(B + offs)
    ux = x.to(tl.uint64)
    uy = y.to(tl.uint64)
    if SIGNED:
        a0 = tl.where(x < 0, 0 - ux, ux)
        b0 = tl.where(y < 0, 0 - uy, uy)
    else:
        a0 = ux
        b0 = uy
    a = a0
    b = b0
    i = 0
    while (tl.max(b) != 0) & (i < CAP):
        for _ in tl.static_range(UNROLL):
            a, b = _lcm_step(a, b)
        i += UNROLL
    g = tl.where(a == 0, 1, a)
    pos = (a0 // g) * b0
    if SIGNED:
        p64 = pos.to(tl.int64)
        res = tl.where(p64 < 0, 0 - p64, p64)
    else:
        res = pos
    if MASKED:
        tl.store(A + offs, res, mask=mask)
    else:
        tl.store(A + offs, res)


def _specialized_lcm_(self, other):
    numel = self.numel()
    if numel == 0:
        return self
    dtype = self.dtype
    # Tiny workloads: BLOCK=128 (1 elem/lane, smaller reduction tree, more blocks)
    # wins on S5000 by ~25% vs BLOCK=256; large workloads stay BLOCK=256 (sweep-verified).
    if numel <= 65536:
        BLOCK = 128
    else:
        BLOCK = 256
    masked = (numel % BLOCK) != 0
    grid = (triton.cdiv(numel, BLOCK),)
    if dtype == torch.int8:
        _lcm_small[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=24,
            UNROLL=4,
            WIDTH=8,
            SIGNED=True,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.uint8:
        _lcm_small[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=24,
            UNROLL=4,
            WIDTH=8,
            SIGNED=False,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.int16:
        _lcm_small[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=24,
            UNROLL=4,
            WIDTH=16,
            SIGNED=True,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.uint16:
        _lcm_small[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=24,
            UNROLL=4,
            WIDTH=16,
            SIGNED=False,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.int32:
        _lcm_small[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=48,
            UNROLL=4,
            WIDTH=32,
            SIGNED=True,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.uint32:
        _lcm_small[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=48,
            UNROLL=4,
            WIDTH=32,
            SIGNED=False,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.int64:
        _lcm_u64[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=96,
            UNROLL=1,
            SIGNED=True,
            MASKED=masked,
            num_warps=4,
        )
    elif dtype == torch.uint64:
        _lcm_u64[grid](
            self,
            other,
            numel,
            BLOCK=BLOCK,
            CAP=96,
            UNROLL=1,
            SIGNED=False,
            MASKED=masked,
            num_warps=4,
        )
    else:
        raise NotImplementedError(f"lcm_ unsupported dtype {dtype}")
    return self


def lcm_(self, other):
    logger.debug("GEMS_MTHREADS LCM_")
    if (
        isinstance(self, torch.Tensor)
        and self.device.type == "musa"
        and self.dtype in _SUPPORTED_DTYPES
        and isinstance(other, torch.Tensor)
        and other.device.type == "musa"
        and other.dtype in _SUPPORTED_DTYPES
    ):
        return _specialized_lcm_(self, other)
    return default_lcm_(self, other)
