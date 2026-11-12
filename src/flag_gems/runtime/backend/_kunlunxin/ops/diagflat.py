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

# Kunlunxin(XPU) override for diagflat (aten::diagflat).
#
# The generic implementation (src/flag_gems/ops/diagflat.py) routes small
# outputs (m*m <= 2**21) through a 2-D tiled fused kernel (_diagflat_tile_kernel).
# On XPU a 2-D tile is a known trap (KERNEL_OPT_EXPERIENCE 2.4-73): the m<=1024
# cells measure ~0.005x.  The 1-D zero-fill used for larger m also under-fills
# the memory bandwidth with BLOCK=4096.
#
# Fix (forward, real dtypes): a large-block unmasked zero fill over the
# same-itemsize integer view (BLOCK=65536) plus the existing 1-D strided
# diagonal scatter.  No 2-D tile, no fused fill+diag barrier/division path.
# Backward and complex forward are delegated to the generic implementation
# (still the flag_gems Triton kernels, not an aten fallback).

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.diagflat import (
    _COMPLEX_REAL,
    _DiagFlat,
    _fill_view,
    run as _generic_run,
)

logger = logging.getLogger(__name__)

_BLOCK_FILL = 65536
_BLOCK_SCATTER = 512


@triton.jit
def _diagflat_zero_kernel(
    out_ptr, count, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    zero = tl.zeros([BLOCK], dtype=out_ptr.dtype.element_ty)
    if NEED_MASK:
        tl.store(out_ptr + offs, zero, mask=offs < count)
    else:
        tl.store(out_ptr + offs, zero)


@triton.jit
def _diagflat_scatter_kernel(
    src_ptr, dst_ptr, count, sstride, dstride, base, BLOCK: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < count
    val = tl.load(src_ptr + offs * sstride, mask=mask)
    tl.store(dst_ptr + offs * dstride + base, val, mask=mask)


def _forward_fast(self, k):
    n = self.numel()
    m = n + (k if k >= 0 else -k)
    out = torch.empty((m, m), dtype=self.dtype, device=self.device)
    if out.numel() == 0:
        return out

    buf = out.view(_fill_view(out.element_size()))
    count = buf.numel()
    _diagflat_zero_kernel[(triton.cdiv(count, _BLOCK_FILL),)](
        buf,
        count,
        BLOCK=_BLOCK_FILL,
        NEED_MASK=(count % _BLOCK_FILL != 0),
        num_warps=32,
    )

    if n > 0:
        flat = self.reshape(-1)
        base = k if k >= 0 else (-k) * m
        _diagflat_scatter_kernel[(triton.cdiv(n, _BLOCK_SCATTER),)](
            flat,
            out,
            n,
            flat.stride(0),
            m + 1,
            base,
            BLOCK=_BLOCK_SCATTER,
            num_warps=4,
        )
    return out


def diagflat(self, offset=0):
    logger.debug("GEMS_KUNLUNXIN DIAGFLAT")
    if not isinstance(self, torch.Tensor):
        raise TypeError("diagflat(): argument 'self' must be a Tensor")
    if not isinstance(offset, int):
        raise TypeError("diagflat(): argument 'offset' must be an integer")
    k = int(offset)
    if self.requires_grad:
        return _DiagFlat.apply(self, k)
    if self.dtype.is_complex:
        return _generic_run(self, k)
    return _forward_fast(self, k)
