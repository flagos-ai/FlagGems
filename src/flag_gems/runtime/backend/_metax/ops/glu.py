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

"""MetaX GLU: contiguous last-axis tiles with a general strided fallback."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from flag_gems.ops.glu import glu_kernel
from flag_gems.runtime import torch_device_fn

_CONFIG_2BYTE = (16, 2, 1, 1)
_CONFIG_WIDE = (8, 4, 3, 0)
_BLOCK_FLAT = 1024
_NUM_WARPS_FLAT = 4


@triton.jit
def _glu_lastdim_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    D: tl.constexpr,
    D2: tl.constexpr,
    BLOCK_M: tl.constexpr,
    RECIP_FREE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, D)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_M), BLOCK_M)
    cols = tl.max_contiguous(tl.multiple_of(cols, D), D)
    in_offs = rows[:, None] * D2 + cols[None, :]
    if NEED_MASK:
        row_mask = (rows < n_rows)[:, None]
        a = tl.load(x_ptr + in_offs, mask=row_mask, other=0.0)
        b = tl.load(x_ptr + in_offs + D, mask=row_mask, other=0.0)
    else:
        a = tl.load(x_ptr + in_offs)
        b = tl.load(x_ptr + in_offs + D)
    a32 = a.to(tl.float32)
    b32 = b.to(tl.float32)
    if RECIP_FREE:
        den = 1.0 + tl.exp2(b32 * -1.4426950408889634)
        res = a32 * tl.exp2(-libdevice.log2(den))
    else:
        res = a32 / (1.0 + tl.exp2(b32 * -1.4426950408889634))
    out_offs = rows[:, None] * D + cols[None, :]
    if NEED_MASK:
        tl.store(
            out_ptr + out_offs,
            res.to(out_ptr.dtype.element_ty),
            mask=(rows < n_rows)[:, None],
        )
    else:
        tl.store(out_ptr + out_offs, res.to(out_ptr.dtype.element_ty))


def glu(self, dim=-1):
    ndim = self.ndim
    if ndim == 0:
        raise RuntimeError("glu: input must have at least one dimension")
    if not -ndim <= dim < ndim:
        raise IndexError("glu: dimension out of range")
    dim %= ndim
    width = self.shape[dim]
    if width % 2:
        raise RuntimeError("glu: split dimension must be even")
    half = width // 2
    shape = list(self.shape)
    shape[dim] = half
    if self.numel() == 0:
        return torch.empty(shape, dtype=self.dtype, device=self.device)
    if (
        dim != ndim - 1
        or not self.is_contiguous()
        or half > 4096
        or half & (half - 1)
        or self.numel() >= 2**31
        or self.dtype not in (torch.float16, torch.bfloat16, torch.float32)
    ):
        a, b = torch.chunk(self, 2, dim=dim)
        return glu_kernel(a, b)
    out = torch.empty(shape, dtype=self.dtype, device=self.device)
    rows = out.numel() // half
    bm, warps, stages, recip_free = (
        _CONFIG_2BYTE if self.element_size() == 2 else _CONFIG_WIDE
    )
    with torch_device_fn.device(self.device):
        _glu_lastdim_kernel[(triton.cdiv(rows, bm),)](
            self,
            out,
            rows,
            D=half,
            D2=width,
            BLOCK_M=bm,
            RECIP_FREE=recip_free,
            NEED_MASK=rows % bm != 0,
            num_warps=warps,
            num_stages=stages,
        )
    return out
