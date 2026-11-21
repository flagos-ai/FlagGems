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


@triton.jit
def _cat_two_rows(
    A,
    B,
    OUT,
    ROWS: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    BR: tl.constexpr,
    BL: tl.constexpr,
    BK: tl.constexpr,
):
    row = tl.program_id(0) * BR + tl.arange(0, BR)
    left = tl.arange(0, BL)
    right = tl.arange(0, BK)
    live = row[:, None] < ROWS
    lmask = live & (left[None, :] < LEFT)
    rmask = live & (right[None, :] < RIGHT)
    lhs = tl.load(A + row[:, None] * LEFT + left[None, :], mask=lmask, other=0)
    rhs = tl.load(B + row[:, None] * RIGHT + right[None, :], mask=rmask, other=0)
    dst = OUT + row[:, None] * (LEFT + RIGHT)
    tl.store(dst + left[None, :], lhs, mask=lmask)
    tl.store(dst + LEFT + right[None, :], rhs, mask=rmask)


def cat(A, dim=0):
    if isinstance(A, (list, tuple)) and len(A) == 2:
        left, right = A
        if (
            left.device.type == "npu"
            and left.device == right.device
            and left.ndim in (2, 3)
            and right.ndim == left.ndim
            and dim in (-1, left.ndim - 1)
            and left.shape[:-1] == right.shape[:-1]
            and 0 < left.shape[-1] <= 2048
            and 0 < right.shape[-1] <= 2048
            and left.is_contiguous()
            and right.is_contiguous()
            and left.dtype == right.dtype
            and left.dtype in (torch.float32, torch.bfloat16, torch.int32)
            and not left.is_neg()
            and not right.is_neg()
            and left.numel() + right.numel() < 2**31
        ):
            a, b = left.shape[-1], right.shape[-1]
            rows = left.numel() // a
            out = torch.empty(
                (*left.shape[:-1], a + b), device=left.device, dtype=left.dtype
            )
            if rows:
                br = min(8, triton.next_power_of_2(rows))
                with torch_device_fn.device(left.device):
                    _cat_two_rows[(triton.cdiv(rows, br),)](
                        left,
                        right,
                        out,
                        rows,
                        a,
                        b,
                        br,
                        triton.next_power_of_2(a),
                        triton.next_power_of_2(b),
                    )
            return out

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
