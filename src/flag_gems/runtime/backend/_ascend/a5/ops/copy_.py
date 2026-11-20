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
def _copy_dense(D, S, n: tl.constexpr, BLOCK: tl.constexpr, SPLAT: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n
    if SPLAT:
        val = tl.load(S)
        val = tl.broadcast_to(val, (BLOCK,))
    else:
        val = tl.load(S + offsets, mask=mask, other=0)
    tl.store(D + offsets, val, mask=mask)


def copy_(dst, src, non_blocking=False):
    if (
        dst.device.type != "npu"
        or src.device != dst.device
        or src.shape != dst.shape
        or not dst.is_contiguous()
        or not (src.is_contiguous() or all(s == 0 for s in src.stride()))
        or (
            src.dtype != dst.dtype
            and (src.dtype, dst.dtype) != (torch.bfloat16, torch.float32)
        )
        or src.dtype
        not in (torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool)
        or dst.numel() >= 2**31
        or torch._C._is_alias_of(src, dst)
        or src.is_neg()
        or dst.is_neg()
        or src.requires_grad
        or dst.requires_grad
    ):
        from flag_gems.runtime.backend._ascend.ops.copy import copy_ as ascend_copy

        return ascend_copy(dst, src, non_blocking=non_blocking)
    n = dst.numel()
    if n:
        # Use byte lanes for booleans; splat copies preserve signed zeros.
        source = src.view(torch.uint8) if src.dtype == torch.bool else src
        dest = dst.view(torch.uint8) if dst.dtype == torch.bool else dst
        block = min(triton.next_power_of_2(n), 8192)
        with torch_device_fn.device(dst.device):
            _copy_dense[(triton.cdiv(n, block),)](
                dest, source, n, block, not src.is_contiguous()
            )
    return dst
