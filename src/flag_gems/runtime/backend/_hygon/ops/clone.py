# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Single device kernel for dense and strided clone layouts."""

import torch
import triton
import triton.language as tl

from flag_gems.ops.clone import clone as _generic_clone
from flag_gems.runtime import torch_device_fn


@triton.jit
def _copy(
    X,
    Y,
    SIZE: tl.constexpr,
    SHAPE: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    FLAT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if FLAT:
        x, y = i, i
    else:
        cur = i
        x = tl.full((BLOCK,), 0, tl.int64)
        y = tl.full((BLOCK,), 0, tl.int64)
        for d in tl.static_range(len(SHAPE) - 1, -1, -1):
            coord = cur % SHAPE[d]
            x += coord * XS[d]
            y += coord * YS[d]
            cur = cur // SHAPE[d]
    value = tl.load(X + x, i < SIZE, other=0)
    tl.store(Y + y, value, i < SIZE)


class _Clone(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, memory_format):
        out = torch.empty_like(inp, memory_format=memory_format)
        if inp.numel():
            flat = tuple(inp.stride()) == tuple(
                out.stride()
            ) and torch.ops.aten.is_non_overlapping_and_dense(inp)
            small_transpose = (
                inp.ndim >= 2
                and inp.shape[-2] * inp.shape[-1] <= 32
                and inp.stride(-2) == 1
                and inp.stride(-1) == inp.shape[-2]
            )
            block = (
                256
                if inp.numel() < 4096
                else (512 if small_transpose and not flat else 1024)
            )
            with torch_device_fn.device(inp.device):
                _copy[(triton.cdiv(inp.numel(), block),)](
                    inp,
                    out,
                    inp.numel(),
                    tuple(inp.shape),
                    tuple(inp.stride()),
                    tuple(out.stride()),
                    flat,
                    block,
                    num_warps=4,
                )
        return out

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def clone(inp, memory_format=None):
    if inp.is_complex() or inp.is_conj() or inp.is_neg():
        return _generic_clone(inp, memory_format=memory_format)
    return _Clone.apply(inp, memory_format or torch.preserve_format)
