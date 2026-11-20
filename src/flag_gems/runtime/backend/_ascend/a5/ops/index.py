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


@triton.jit(debug=True)
def _gather_rows(
    X,
    I,
    Y,
    ROWS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    if WIDTH == 1:
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        live = offsets < N
        idx = tl.load(I + offsets, mask=live, other=0)
        valid = (idx >= -ROWS) & (idx < ROWS)
        if tl.max((live & ~valid).to(tl.int32), 0):
            tl.device_assert(False, "row index out of bounds")
        row = tl.where(idx < 0, idx + ROWS, idx)
        value = tl.load(X + row * ROW_STRIDE, mask=live & valid, other=0)
        tl.store(Y + offsets, value, mask=live)
    else:
        dest_row = tl.program_id(0)
        idx = tl.load(I + dest_row)
        valid = (idx >= -ROWS) & (idx < ROWS)
        if not valid:
            tl.device_assert(False, "row index out of bounds")
        row = tl.where(idx < 0, idx + ROWS, idx)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(X + row * ROW_STRIDE + col, mask=valid & (col < WIDTH), other=0)
        tl.store(Y + dest_row * WIDTH + col, value, mask=col < WIDTH)


@triton.jit(debug=True)
def _gather_short_rows(
    X,
    I,
    Y,
    ROWS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BC)
    idx = tl.load(I + r, mask=r < N, other=0)
    valid = (idx >= -ROWS) & (idx < ROWS)
    if tl.max(((~valid) & (r < N)).to(tl.int32), 0):
        tl.device_assert(False, "row index out of bounds")
    src = tl.where(idx < 0, idx + ROWS, idx)
    live = (r < N)[:, None] & (c < WIDTH)[None, :]
    val = tl.load(
        X + src[:, None] * ROW_STRIDE + c[None, :], mask=live & valid[:, None], other=0
    )
    tl.store(Y + r[:, None] * WIDTH + c[None, :], val, mask=live)


def index(inp, indices):
    if (
        inp.device.type != "npu"
        or not isinstance(indices, (list, tuple))
        or len(indices) != 1
        or not isinstance(indices[0], torch.Tensor)
        or indices[0].device != inp.device
        or indices[0].dtype not in (torch.int64, torch.int32)
        or indices[0].ndim != 1
        or not indices[0].is_contiguous()
        or inp.ndim not in (1, 2)
        or inp.stride(-1) != 1
        or (
            inp.ndim == 2
            and (inp.stride(0) < inp.shape[1] or inp.stride(0) * inp.shape[0] >= 2**31)
        )
        or not inp.numel()
        or inp.numel() >= 2**31
        or inp.is_neg()
        or inp.dtype not in (torch.bfloat16, torch.float32, torch.int32, torch.int64)
        or (inp.ndim == 2 and inp.is_contiguous() and indices[0].numel() > 1)
        or indices[0].numel() * (inp.numel() // inp.shape[0]) >= 2**31
    ):
        from flag_gems.runtime.backend._ascend.ops.index import index as ascend_index

        return ascend_index(inp, indices)
    idx = indices[0]
    width = inp.numel() // inp.shape[0]
    out = torch.empty((idx.numel(), *inp.shape[1:]), dtype=inp.dtype, device=inp.device)
    if 1 < width <= 128 and idx.numel() >= 16:
        with torch_device_fn.device(inp.device):
            _gather_short_rows[(triton.cdiv(idx.numel(), 16),)](
                inp,
                idx,
                out,
                inp.shape[0],
                inp.stride(0),
                width,
                idx.numel(),
                16,
                triton.next_power_of_2(width),
            )
        return out
    if out.numel():
        block = min(triton.next_power_of_2(out.numel() if width == 1 else width), 2048)
        grid = (
            (triton.cdiv(out.numel(), block), 1)
            if width == 1
            else (idx.numel(), triton.cdiv(width, block))
        )
        with torch_device_fn.device(inp.device):
            _gather_rows[grid](
                inp, idx, out, inp.shape[0], inp.stride(0), width, out.numel(), block
            )
    return out
