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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _red_inner_static(
    X, OUT, inn, NIB, R: tl.constexpr, BLOCK_I: tl.constexpr, MODE: tl.constexpr
):
    pid = tl.program_id(0)
    o = pid // NIB
    ib = pid % NIB
    off = ib * BLOCK_I + tl.arange(0, BLOCK_I)
    m = off < inn
    base = o * (R * inn) + off
    if MODE == 0:
        acc = tl.zeros([BLOCK_I], dtype=tl.float32)
        for r in tl.static_range(R):
            acc += tl.load(X + base + r * inn, mask=m, other=0.0).to(tl.float32)
        tl.store(OUT + o * inn + off, acc.to(OUT.dtype.element_ty), mask=m)
    elif MODE == 1:
        acc = tl.zeros([BLOCK_I], dtype=tl.int64)
        for r in tl.static_range(R):
            acc += tl.load(X + base + r * inn, mask=m, other=0).to(tl.int64)
        tl.store(OUT + o * inn + off, acc.to(OUT.dtype.element_ty), mask=m)
    else:
        acc = tl.zeros([BLOCK_I], dtype=tl.int32)
        for r in tl.static_range(R):
            acc += tl.load(X + base + r * inn, mask=m, other=0).to(tl.int32)
        tl.store(OUT + o * inn + off, acc.to(OUT.dtype.element_ty), mask=m)


def sum_dim(inp, dim=None, keepdim=False, *, dtype=None):
    axis = dim[0] if isinstance(dim, (list, tuple)) and len(dim) == 1 else dim
    if (
        inp.device.type != "npu"
        or not inp.is_contiguous()
        or inp.layout != torch.strided
        or inp.ndim != 3
        or type(axis) is not int
        or axis not in (-2, 1)
        or not (0 < inp.shape[1] <= 16 and inp.numel() < 2**31)
        or inp.dtype not in (torch.bfloat16, torch.float32)
        or dtype not in (None, inp.dtype, torch.float32)
        or inp.is_neg()
    ):
        from flag_gems.runtime.backend._ascend.ops.sum import sum_dim as ascend_sum_dim

        return ascend_sum_dim(inp, dim, keepdim, dtype=dtype)
    outer, reduce, inner = inp.shape
    out = torch.empty(
        (outer, 1, inner) if keepdim else (outer, inner),
        device=inp.device,
        dtype=dtype or inp.dtype,
    )
    if not out.numel():
        return out
    block = min(1024, triton.next_power_of_2(inner))
    tiles = triton.cdiv(inner, block)
    with torch_device_fn.device(inp.device):
        _red_inner_static[(outer * tiles,)](inp, out, inner, tiles, reduce, block, 0)
    return out
