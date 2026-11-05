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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

NUM_SIPS = 24

# Threshold: below this many elements the simpler flat kernel wins
# (less overhead, no boundary-check); above it the 2-D block_ptr kernel
# wins (contiguous writes + better read locality).
FLAT_THRESHOLD = 8192


@triton.jit
def pixel_unshuffle_kernel_flat(
    in_ptr,
    out_ptr,
    n_elements,
    sN_in,
    sC_in,
    sH_in,
    sN_out: tl.constexpr,
    sC_out: tl.constexpr,
    sH_out: tl.constexpr,
    R: tl.constexpr,
    R2: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Flat element-wise kernel: contiguous output write, address-mapped input read.

    Strides are constexpr so integer divisions are compiled to multiply-shift.
    """
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)

    for block_id in range(pid, num_blocks, num_pid):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Decode output linear index → (n, c_out, h_out, w_out) — constexpr divs
        n = offsets // sN_out
        rem1 = offsets - n * sN_out
        c_out = rem1 // sC_out
        rem2 = rem1 - c_out * sC_out
        h_out = rem2 // sH_out
        w_out = rem2 - h_out * sH_out

        # Decode c_out → (c_in, dh, dw) — constexpr divs
        c_in = c_out // R2
        remc = c_out - c_in * R2
        dh = remc // R
        dw = remc - dh * R

        in_index = (
            n * sN_in + c_in * sC_in + (h_out * R + dh) * sH_in + (w_out * R + dw)
        )

        x = tl.load(in_ptr + in_index, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)


@libentry()
@triton.jit(do_not_specialize=["N", "C_out", "H_out", "W_out", "C_in", "H_in", "W_in"])
def pixel_unshuffle_kernel_block_ptr(
    in_ptr,
    out_ptr,
    N,
    C_out,
    H_out,
    W_out,
    C_in,
    H_in,
    W_in,
    R: tl.constexpr,
    R2: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """2-D tiled kernel using block_ptr for both strided reads and coalesced writes.

    Best for large tensors: 2-D strided reads have better spatial locality than
    the flat kernel's scattered reads, and the coalesced writes are critical for
    GCU300 write throughput.
    """
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)

    num_h_blocks = tl.cdiv(H_out, BLOCK_H)
    num_w_blocks = tl.cdiv(W_out, BLOCK_W)
    total_blocks = N * C_out * num_h_blocks * num_w_blocks

    for block_id in range(pid, total_blocks, num_pid):
        w_block = block_id % num_w_blocks
        rem1 = block_id // num_w_blocks
        h_block = rem1 % num_h_blocks
        rem2 = rem1 // num_h_blocks
        c_out = rem2 % C_out
        n = rem2 // C_out

        # Decode c_out → (c_in, dh, dw) — ONCE per tile (constexpr)
        c_in = c_out // R2
        remc = c_out - c_in * R2
        dh = remc // R
        dw = remc - dh * R

        h_out_start = h_block * BLOCK_H
        w_out_start = w_block * BLOCK_W

        # Input block_ptr: strided access with strides (R*W_in, R)
        in_base = (
            in_ptr + n * (C_in * H_in * W_in) + c_in * (H_in * W_in) + dh * W_in + dw
        )
        in_block = tl.make_block_ptr(
            in_base,
            shape=(H_out, W_out),
            strides=(R * W_in, R),
            offsets=(h_out_start, w_out_start),
            block_shape=(BLOCK_H, BLOCK_W),
            order=(1, 0),
        )

        # Output block_ptr: contiguous access with strides (W_out, 1)
        out_base = out_ptr + n * (C_out * H_out * W_out) + c_out * (H_out * W_out)
        out_block = tl.make_block_ptr(
            out_base,
            shape=(H_out, W_out),
            strides=(W_out, 1),
            offsets=(h_out_start, w_out_start),
            block_shape=(BLOCK_H, BLOCK_W),
            order=(1, 0),
        )

        x = tl.load(in_block, boundary_check=(0, 1))
        tl.store(out_block, x, boundary_check=(0, 1))


def _launch_pixel_unshuffle_kernel(
    inp: torch.Tensor, downscale_factor: int, out: torch.Tensor
):
    N, C_in, H_in, W_in = inp.shape
    r = int(downscale_factor)
    R2 = r * r
    C_out = C_in * R2
    H_out = H_in // r
    W_out = W_in // r

    n_elements = N * C_out * H_out * W_out

    # Strides for contiguous NCHW
    sN_in = C_in * H_in * W_in
    sC_in = H_in * W_in
    sH_in = W_in
    sN_out = C_out * H_out * W_out
    sC_out = H_out * W_out
    sH_out = W_out

    if n_elements <= FLAT_THRESHOLD:
        # --- Flat kernel for small tensors ---
        if n_elements <= 1024:
            BLOCK_SIZE = 1024
        elif n_elements <= 4096:
            BLOCK_SIZE = 4096
        else:
            BLOCK_SIZE = 8192

        num_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
        grid = (min(NUM_SIPS, num_blocks),)

        with torch_device_fn.device(inp.device):
            pixel_unshuffle_kernel_flat[grid](
                inp,
                out,
                n_elements,
                sN_in,
                sC_in,
                sH_in,
                sN_out,
                sC_out,
                sH_out,
                r,
                R2,
                BLOCK_SIZE,
                num_warps=4,
                num_stages=2,
            )
    else:
        # --- Block_ptr kernel for large tensors ---
        BLOCK_H = min(triton.next_power_of_2(H_out), 64)
        BLOCK_W = min(triton.next_power_of_2(W_out), 256)

        num_h_blocks = triton.cdiv(H_out, BLOCK_H)
        num_w_blocks = triton.cdiv(W_out, BLOCK_W)
        total_blocks = N * C_out * num_h_blocks * num_w_blocks
        grid = (min(NUM_SIPS, total_blocks),)

        # num_warps=1: for memory-bound copy, fewer warps → less memory-port
        # contention → higher per-warp effective bandwidth.
        # num_stages=3: marginal pipelining benefit for the loop body.
        num_warps = 1
        num_stages = 3

        with torch_device_fn.device(inp.device):
            pixel_unshuffle_kernel_block_ptr[grid](
                inp,
                out,
                N,
                C_out,
                H_out,
                W_out,
                C_in,
                H_in,
                W_in,
                r,
                R2,
                BLOCK_H,
                BLOCK_W,
                num_warps=num_warps,
                num_stages=num_stages,
            )


def pixel_unshuffle(input, downscale_factor, *, layout=None):
    logger.debug("GEMS_ENFLAME PIXEL_UNSHUFFLE")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = x.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (
        W % r == 0
    ), "H and W must be divisible by downscale_factor"

    out_shape = (N, C * r * r, H // r, W // r)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    if x.numel() == 0:
        return out
    _launch_pixel_unshuffle_kernel(x, r, out)
    return out


def pixel_unshuffle_out(input, downscale_factor, out):
    logger.debug("GEMS_ENFLAME PIXEL_UNSHUFFLE_OUT")
    x = input
    if not x.is_contiguous():
        x = x.contiguous()
    assert x.ndim == 4, "Input must be a 4D tensor (N, C, H, W)"
    N, C, H, W = x.shape
    r = int(downscale_factor)
    assert r > 0, "downscale_factor must be > 0"
    assert (H % r == 0) and (
        W % r == 0
    ), "H and W must be divisible by downscale_factor"
    expected_shape = (N, C * r * r, H // r, W // r)
    assert tuple(out.shape) == expected_shape, f"out must have shape {expected_shape}"
    assert out.dtype == x.dtype, "out dtype must match input dtype"
    assert out.device == x.device, "out device must match input device"
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")

    if x.numel() == 0:
        return out
    _launch_pixel_unshuffle_kernel(x, r, out)
    return out
