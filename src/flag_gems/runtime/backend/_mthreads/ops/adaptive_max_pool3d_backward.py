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

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.adaptive_max_pool3d_backward import (
    adaptive_max_pool3d_backward as default_adaptive_max_pool3d_backward,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(
    f"flag_gems.runtime.backend._mthreads.ops.{__name__.split('.')[-1]}"
)

_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}


@libentry()
@triton.jit
def _zero_fill_kernel(out_ptr, n_in, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_in
    tl.store(out_ptr + offs, tl.zeros((BLOCK,), dtype=tl.float32), mask=mask)


@libentry()
@triton.jit
def _scatter_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    n_out,
    plane_in,
    plane_out,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_out
    g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
    idx = tl.load(idx_ptr + offs, mask=mask, other=0)
    target = (offs // plane_out) * plane_in + idx
    # accumulate: overlapping adaptive windows can map two outputs to one input
    tl.atomic_add(out_ptr + target, g, mask=mask)


@libentry()
@triton.jit
def _gather_kernel(
    grad_ptr,
    idx_ptr,
    out_ptr,
    D_IN: tl.constexpr,
    PLANE_IN: tl.constexpr,
    PLANE_OUT: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK: tl.constexpr,
    FULL: tl.constexpr,
):
    # Divisible case (all axes integral): adaptive windows tile the input
    # disjointly, so each input position belongs to exactly one window o(x)
    # and is a scatter target iff idx[o(x)] == x. 3D grid (plane, z, hw-block):
    # od is a scalar (z//KD) and per-lane work is only the (y,w) decomposition,
    # so the integer math chain is short. Single launch, no zero pass, no
    # atomics, stores fully coalesced.
    plane = tl.program_id(0)
    z = tl.program_id(1)
    pid = tl.program_id(2)
    od = z // KD
    hw = pid * BLOCK + tl.arange(0, BLOCK)
    if FULL:
        y = hw // W_IN
        w = hw - y * W_IN
        oh = y // KH
        ow = w // KW
        o_local = od * (H_OUT * W_OUT) + oh * W_OUT + ow
        o_flat = plane * PLANE_OUT + o_local
        idx0 = tl.load(idx_ptr + o_flat)
        g0 = tl.load(grad_ptr + o_flat)
        local = z * (H_IN * W_IN) + hw
        hit = idx0 == local
        val = tl.where(hit, g0, tl.zeros((BLOCK,), dtype=g0.dtype))
        tl.store(out_ptr + plane * PLANE_IN + local, val)
    else:
        mask = hw < H_IN * W_IN
        y = hw // W_IN
        w = hw - y * W_IN
        oh = y // KH
        ow = w // KW
        o_local = od * (H_OUT * W_OUT) + oh * W_OUT + ow
        o_flat = plane * PLANE_OUT + o_local
        idx0 = tl.load(idx_ptr + o_flat, mask=mask, other=0)
        g0 = tl.load(grad_ptr + o_flat, mask=mask, other=0.0)
        local = z * (H_IN * W_IN) + hw
        hit = idx0 == local
        val = tl.where(hit, g0, tl.zeros((BLOCK,), dtype=g0.dtype))
        tl.store(out_ptr + plane * PLANE_IN + local, val, mask=mask)


def _use_triton_kernel(grad_output, self_input, indices) -> bool:
    if (
        not isinstance(grad_output, torch.Tensor)
        or not isinstance(self_input, torch.Tensor)
        or not isinstance(indices, torch.Tensor)
    ):
        return False
    if grad_output.device.type != "musa" or grad_output.dtype not in _SUPPORTED_DTYPES:
        return False
    if (
        not grad_output.is_contiguous()
        or not self_input.is_contiguous()
        or not indices.is_contiguous()
    ):
        return False
    if grad_output.numel() == 0 or self_input.numel() == 0:
        return False
    return True


def adaptive_max_pool3d_backward(grad_output, self_input, indices):
    logger.debug("GEMS_MTHREADS ADAPTIVE_MAX_POOL3D_BACKWARD")
    if not _use_triton_kernel(grad_output, self_input, indices):
        return default_adaptive_max_pool3d_backward(grad_output, self_input, indices)
    out = torch.empty_like(self_input)
    n_in = self_input.numel()
    n_out = grad_output.numel()
    ds, hs, ws = (
        self_input.shape[-3],
        self_input.shape[-2],
        self_input.shape[-1],
    )
    do_, ho_, wo_ = (
        grad_output.shape[-3],
        grad_output.shape[-2],
        grad_output.shape[-1],
    )
    plane_in = ds * hs * ws
    plane_out = do_ * ho_ * wo_
    divisible = (ds % do_ == 0) and (hs % ho_ == 0) and (ws % wo_ == 0)
    with torch_device_fn.device(grad_output.device):
        if divisible:
            hw = hs * ws
            n_planes = grad_output.numel() // plane_out
            if hw >= 512:
                BLOCK, W = 128, 4
            elif hw >= 128:
                BLOCK, W = (256, 8) if hw % 256 == 0 else (128, 4)
            else:
                BLOCK, W = 64, 2
            full = hw % BLOCK == 0
            _gather_kernel[
                (n_planes, ds, hw // BLOCK if full else triton.cdiv(hw, BLOCK))
            ](
                grad_output,
                indices,
                out,
                D_IN=ds,
                PLANE_IN=plane_in,
                PLANE_OUT=plane_out,
                H_IN=hs,
                W_IN=ws,
                H_OUT=ho_,
                W_OUT=wo_,
                KD=ds // do_,
                KH=hs // ho_,
                KW=ws // wo_,
                BLOCK=BLOCK,
                FULL=full,
                num_warps=W,
            )
        else:
            BLOCK = 1024
            _zero_fill_kernel[(triton.cdiv(n_in, BLOCK),)](
                out, n_in, BLOCK=BLOCK, num_warps=4
            )
            _scatter_kernel[(triton.cdiv(n_out, BLOCK),)](
                grad_output,
                indices,
                out,
                n_out,
                plane_in,
                plane_out,
                BLOCK=BLOCK,
                num_warps=4,
            )
    return out


__all__ = ["adaptive_max_pool3d_backward"]
