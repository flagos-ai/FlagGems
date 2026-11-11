# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense casts that preserve physical element order."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.to import to_copy as default_to_copy
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from .to_kernels import cast_fp32_bf16_packed, cast_fp32_bf16_quad

_PACKED_MIN_ELEMENTS = 8 * 1024 * 1024


@triton.jit
def convert_dense(value, Y):
    if Y.dtype.element_ty == tl.int1:
        value = value != 0
    return value.to(Y.dtype.element_ty)


@libentry()
@triton.jit
def cast_dense(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if N % BLOCK == 0:
        tl.store(Y + offsets, convert_dense(tl.load(X + offsets), Y))
    else:
        # Keep full blocks vectorized when the last block is partial.
        if tl.program_id(0) < N // BLOCK:
            tl.store(Y + offsets, convert_dense(tl.load(X + offsets), Y))
        else:
            value = tl.load(X + offsets, offsets < N, other=0)
            tl.store(Y + offsets, convert_dense(value, Y), offsets < N)


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
    target_device = x.device if device is None else torch.device(device)
    dense_pair = (x.dtype, dtype) in (
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.bool),
        (torch.bfloat16, torch.float32),
        (torch.int64, torch.float32),
        (torch.bool, torch.int32),
    )
    dense_format = memory_format in (None, torch.preserve_format) or (
        memory_format == torch.contiguous_format and x.is_contiguous()
    )
    if (
        dense_pair
        and x.layout == torch.strided
        and layout in (None, torch.strided)
        and x.device.type in ("cuda", "flagos")
        and target_device == x.device
        and pin_memory is None
        and dense_format
        and not x.is_neg()
        and not x.is_conj()
        and 0 < x.numel() <= 2147483647
        and (x.is_contiguous() or torch.ops.aten.is_non_overlapping_and_dense(x))
    ):
        # Dense permutations have no storage holes. Casting their physical order
        # into an output with identical strides preserves logical coordinates.
        with torch_device_fn.device(x.device):
            out = torch.empty_strided(x.shape, x.stride(), dtype=dtype, device=x.device)
            n = x.numel()
            kernel = cast_dense
            block, warps = (256, 1) if n <= 4096 else (1024, 1)
            if x.dtype == torch.bfloat16 and dtype == torch.float32:
                if n >= 8 * 1024 * 1024:
                    block, warps = 4096, 4
                elif n > 262144:
                    block, warps = (512, 1) if n < 524288 else (2048, 2)
            elif (
                x.dtype == torch.float32
                and dtype == torch.bfloat16
                and (n >= _PACKED_MIN_ELEMENTS or 32768 <= n <= 262144)
                and n % 2 == 0
                and x.storage_offset() % 2 == 0
            ):
                # Each naturally aligned 64-bit load contains two FP32 values;
                # the rounded BF16 pair is written with one 32-bit store.
                kernel = cast_fp32_bf16_packed
                block, warps = (1024, 1) if n < _PACKED_MIN_ELEMENTS else (2048, 2)
                if (
                    n >= _PACKED_MIN_ELEMENTS
                    and n % 4 == 0
                    and x.storage_offset() % 4 == 0
                ):
                    # Match input/output vector widths with four BF16 values
                    # per 64-bit store, retaining the same integer RNE rule.
                    kernel = cast_fp32_bf16_quad
                    block, warps = 4096, 4
            kernel[(triton.cdiv(n, block),)](
                x,
                out,
                n,
                BLOCK=block,
                num_warps=warps,
                num_stages=1,
                allow_flush_denorm=False,
            )
        return out
    return default_to_copy(
        x,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        non_blocking=non_blocking,
        memory_format=memory_format,
    )
