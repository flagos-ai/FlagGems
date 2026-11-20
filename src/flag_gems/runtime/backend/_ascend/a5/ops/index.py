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
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    live = off < N
    ix = tl.load(I + off // WIDTH, mask=live, other=0)
    ok = (ix >= -ROWS) & (ix < ROWS)
    tl.device_assert((~live) | ok, "row index out of bounds")
    row = tl.where(ix < 0, ix + ROWS, ix)
    safe_row = tl.where(ok, row, 0).to(tl.int32)
    values = tl.load(X + safe_row * WIDTH + off % WIDTH, mask=live & ok, other=0)
    tl.store(Y + off, values, mask=live)


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
        or not inp.is_contiguous()
        or inp.ndim not in (1, 2)
        or not inp.numel()
        or inp.numel() >= 2**31
        or inp.is_neg()
        or inp.dtype not in (torch.bfloat16, torch.float32, torch.int32, torch.int64)
        or indices[0].numel() * (inp.numel() // inp.shape[0]) >= 2**31
    ):
        from flag_gems.runtime.backend._ascend.ops.index import index as ascend_index

        return ascend_index(inp, indices)
    idx = indices[0]
    width = inp.numel() // inp.shape[0]
    out = torch.empty((idx.numel(), *inp.shape[1:]), dtype=inp.dtype, device=inp.device)
    if out.numel():
        block = min(triton.next_power_of_2(out.numel()), 2048)
        with torch_device_fn.device(inp.device):
            _gather_rows[(triton.cdiv(out.numel(), block),)](
                inp, idx, out, inp.shape[0], width, out.numel(), block
            )
    return out
