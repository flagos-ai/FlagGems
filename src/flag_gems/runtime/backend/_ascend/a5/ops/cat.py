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
def _cat_flat3(
    dst_ptr,
    in0,
    in1,
    in2,
    N: tl.constexpr,
    CUMS: tl.constexpr,
    OFFS: tl.constexpr,
    RLN: tl.constexpr,
    MSKS: tl.constexpr,
    PERM: tl.constexpr,
    NTOT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    p0 = tl.program_id(0)
    if PERM > 1:
        pid = (p0 * PERM) % NTOT
    else:
        pid = p0
    srcs = (in0, in1, in2)
    for j in tl.static_range(N):
        if (pid >= CUMS[j]) & (pid < CUMS[j + 1]):
            rpos = (pid - CUMS[j]) * BLOCK + tl.arange(0, BLOCK)
            if MSKS[j]:
                m = rpos < RLN[j]
                v = tl.load(srcs[j] + rpos, mask=m, other=0)
                tl.store(
                    dst_ptr + OFFS[j] + rpos, v.to(dst_ptr.dtype.element_ty), mask=m
                )
            else:
                v = tl.load(srcs[j] + rpos)
                tl.store(dst_ptr + OFFS[j] + rpos, v.to(dst_ptr.dtype.element_ty))


def cat(A, dim=0):
    if (
        not isinstance(A, (list, tuple))
        or len(A) not in (2, 3)
        or dim not in (-1, 0)
        or any(
            t.device.type != "npu"
            or t.device != A[0].device
            or t.dtype != A[0].dtype
            or t.ndim != 1
            or not t.is_contiguous()
            or t.is_neg()
            for t in A
        )
        or A[0].dtype not in (torch.int32, torch.bfloat16, torch.float32, torch.int64)
        or sum(t.numel() for t in A) >= 2**31
    ):
        from flag_gems.runtime.backend._ascend.ops.cat import cat as ascend_cat

        return ascend_cat(A, dim)
    sizes = tuple(t.numel() for t in A)
    total = sum(sizes)
    out = torch.empty((total,), device=A[0].device, dtype=A[0].dtype)
    if not total:
        return out
    block = min(4096, triton.next_power_of_2(max(sizes)))
    offsets = [0]
    tiles = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
        tiles.append(tiles[-1] + triton.cdiv(size, block))
    with torch_device_fn.device(A[0].device):
        _cat_flat3[(tiles[-1],)](
            out,
            A[0],
            A[1],
            A[-1],
            len(A),
            tuple(tiles),
            tuple(offsets[:-1]),
            sizes,
            tuple(size % block != 0 for size in sizes),
            1,
            tiles[-1],
            block,
        )
    return out
