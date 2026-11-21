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

import math

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _slice_rows(
    in_ptr,
    out_ptr,
    outer,
    L,
    src_base,
    src_stride,
    BR: tl.constexpr,
    BC: tl.constexpr,
    MASK_R: tl.constexpr,
    MASK_C: tl.constexpr,
    SRC_WIDE: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)

    src = src_base + rows[:, None] * src_stride + cols[None, :]
    dst = rows[:, None] * L + cols[None, :]

    if SRC_WIDE:
        # Rows are narrower than one cache line: read the whole row contiguously
        # (BC == src_stride, guaranteed in bounds) and let the store mask the
        # tail beyond L. Loading the same HBM lines, but as full-line runs
        # instead of sub-line strided requests.
        if MASK_R:
            m = rows[:, None] < outer
            v = tl.load(in_ptr + src, mask=m, other=0)
            tl.store(
                out_ptr + dst, v, mask=(rows[:, None] < outer) & (cols[None, :] < L)
            )
        else:
            v = tl.load(in_ptr + src)
            tl.store(out_ptr + dst, v, mask=cols[None, :] < L)
    elif MASK_R:
        if MASK_C:
            m = (rows[:, None] < outer) & (cols[None, :] < L)
            v = tl.load(in_ptr + src, mask=m, other=0)
            tl.store(out_ptr + dst, v, mask=m)
        else:
            m = rows[:, None] < outer
            v = tl.load(in_ptr + src, mask=m, other=0)
            tl.store(out_ptr + dst, v, mask=m)
    else:
        if MASK_C:
            m = cols[None, :] < L
            v = tl.load(in_ptr + src, mask=m, other=0)
            tl.store(out_ptr + dst, v, mask=m)
        else:
            v = tl.load(in_ptr + src)
            tl.store(out_ptr + dst, v)


def slice_copy(inp, dim=0, start=None, end=None, step=1):
    if (
        inp.device.type != "npu"
        or not 1 <= inp.ndim <= 4
        or not -inp.ndim <= dim < inp.ndim
        or step != 1
        or not inp.is_contiguous()
        or inp.numel() >= 2**31
        or inp.dtype not in (torch.bfloat16, torch.float32, torch.int32, torch.int64)
        or inp.is_neg()
    ):
        from flag_gems.ops.slice_copy import slice_copy as generic_slice_copy

        return generic_slice_copy(inp, dim, start, end, step)
    axis = dim % inp.ndim
    begin, stop, _ = slice(start, end).indices(inp.shape[axis])
    length = max(stop - begin, 0)
    shape = list(inp.shape)
    shape[axis] = length
    out = torch.empty(shape, device=inp.device, dtype=inp.dtype)
    if not out.numel():
        return out
    outer = math.prod(inp.shape[:axis])
    inner = math.prod(inp.shape[axis + 1 :])
    span = length * inner
    stride = inp.shape[axis] * inner
    bc = min(2048, triton.next_power_of_2(span))
    br = min(triton.next_power_of_2(outer), max(1, 8192 // bc))
    with torch_device_fn.device(inp.device):
        _slice_rows[(triton.cdiv(outer, br), triton.cdiv(span, bc))](
            inp,
            out,
            outer,
            span,
            begin * inner,
            stride,
            br,
            bc,
            outer % br != 0,
            span % bc != 0,
            False,
        )
    return out
