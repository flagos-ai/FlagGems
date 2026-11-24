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
def _cast_kernel(
    in_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
    IN_BOOL: tl.constexpr,
    OUT_BOOL: tl.constexpr,
):
    """Multi-program / partial-tile form: masked load and store."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(in_ptr + offs, mask=mask)
    if IN_BOOL:
        # bool storage is 0/1, so a direct unsigned conversion is exact and
        # skips the extra compare + select per vector chunk.
        if OUT_BOOL:
            y = (x != 0).to(tl.uint8)
        else:
            y = x.to(out_ptr.dtype.element_ty)
    elif OUT_BOOL:
        y = (x != 0).to(tl.uint8)
    else:
        y = x.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _cast_pipe(
    in_ptr,
    out_ptr,
    n,
    SUB: tl.constexpr,
    NSUB: tl.constexpr,
    IN_BOOL: tl.constexpr,
    OUT_BOOL: tl.constexpr,
):
    """Multi-program path, per-program tile walked as NSUB sub-tiles.

    Same elements per program and same grid as the flat form; the only change
    is that the tile is expressed as a 2-step pipelined loop so the compiler can
    begin the second GM<->UB transfer while the first sub-tile converts.
    """
    pid = tl.program_id(0)
    base = pid * (SUB * NSUB)
    for k in tl.range(0, NSUB, num_stages=2):
        offs = base + k * SUB + tl.arange(0, SUB)
        mask = offs < n
        x = tl.load(in_ptr + offs, mask=mask)
        if IN_BOOL:
            if OUT_BOOL:
                y = (x != 0).to(tl.uint8)
            else:
                y = x.to(out_ptr.dtype.element_ty)
        elif OUT_BOOL:
            y = (x != 0).to(tl.uint8)
        else:
            y = x.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + offs, y, mask=mask)


def to_copy(
    x,
    *,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=None,
    non_blocking=False,
    memory_format=None,
):
    # Leave layout, inter-device copies and type combinations outside this path
    # to the standard implementation, including its validation.
    supported = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.int32,
        torch.int64,
        torch.bool,
    )
    if (
        x.layout != torch.strided
        or x.device.type != "npu"
        or not x.is_contiguous()
        or x.dtype not in supported
        or dtype not in supported
        or not (
            dtype == x.dtype
            or (x.dtype, dtype)
            in (
                (torch.bfloat16, torch.float32),
                (torch.float32, torch.bfloat16),
                (torch.int32, torch.int64),
                (torch.bool, torch.int32),
            )
        )
        or layout not in (None, torch.strided)
        or (device is not None and torch.device(device) != x.device)
        or pin_memory is not None
        or non_blocking
        or memory_format not in (None, torch.preserve_format, torch.contiguous_format)
        or x.numel() >= 2**31
        or x.is_neg()
        or x.is_conj()
    ):
        from flag_gems.ops.to import to_copy as generic_to_copy

        return generic_to_copy(
            x,
            dtype=dtype,
            layout=layout,
            device=device,
            pin_memory=pin_memory,
            non_blocking=non_blocking,
            memory_format=memory_format,
        )
    out = torch.empty_like(x, dtype=dtype, memory_format=torch.contiguous_format)
    n = x.numel()
    if not n:
        return out
    src = x.view(torch.uint8) if x.dtype == torch.bool else x
    dst = out.view(torch.uint8) if dtype == torch.bool else out
    with torch_device_fn.device(x.device):
        if n <= 8192:
            block = triton.next_power_of_2(n)
            _cast_kernel[(1,)](
                src, dst, n, block, x.dtype == torch.bool, dtype == torch.bool
            )
        else:
            _cast_pipe[(triton.cdiv(n, 8192),)](
                src, dst, n, 4096, 2, x.dtype == torch.bool, dtype == torch.bool
            )
    return out
