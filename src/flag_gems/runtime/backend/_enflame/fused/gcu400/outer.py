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

# GCU400 adaptation of ``outer``.
#
# The generic implementation broadcasts ``(M, 1) x (1, N)`` through the
# pointwise ``mul`` op. On GCU400 that lands in ``mul_broadcast_2d_kernel``,
# which computes per-element ``offsets // n_cols`` / modulo indices and issues
# two computed-index gather loads per element; at (4096, 4096) fp16 it takes
# 811us (topsprof, 97% of op time) versus a ~15us store-bandwidth floor.
#
# The rewrite is a row-streaming kernel: every program owns row groups of 8
# rows (grid-stride), loads the 8 row scalars once per group, and streams
# BLOCK_N-wide chunks of the shared y vector (scalar x vector multiply only,
# no 2D broadcast value). Job j20261110124523 probe/sweep on L600 48SP:
#   - grid cap 48 is the sweet spot (24 underfills, 96/128 pay the CTA
#     dispatch tax; e.g. (1024, 1024) fp16: grid48 4.8us vs grid128 10.4us);
#   - BLOCK_N = next_pow2(N) capped at 8192 (BN=4096 at (1024, 65536) is
#     17% slower than 8192);
#   - num_warps = 2 for BLOCK_N <= 4096, 4 for 8192 (fp16/bf16 8192:
#     47.7/46.9us at nw4 vs 52.4/51.5us at nw2);
#   - the kernel lands 0-8% above the dummy-store floor on all core shapes.
# DSM budget: one BLOCK_N live vector <= 32KB (fp32, BN=8192) per program,
# far below the 917504B gcu400 limit; 8KB/warp keeps the masked-1D first
# compile on the safe side of the slice_pad_start regression.
# Tail handling: the last row group is anchored back to ``M - 8`` so scalar
# loads/stores need no row masks (rows are pure functions of the inputs, the
# overlap rewrite is idempotent); M < 8 dispatches one row per program.

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.mv import mv
from flag_gems.runtime import device as runtime_device

from ...gcu400.ops.mul import mul

logger = logging.getLogger(__name__)

_DEVICE_NAME = runtime_device.name

_GRID_CAP = 48
_BLOCK_N_CAP = 8192
_WIDE_ELEMS = 1 << 31


@triton.jit
def outer_fwd_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    M,
    N,
    GRID,
    BLOCK_N: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
    EVEN_COLS: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if ROWS_PER_PROG == 8:
        num_groups = tl.cdiv(M, 8)
        for g in range(pid, num_groups, GRID):
            r0 = g * 8
            if r0 > M - 8:
                r0 = M - 8
            x0 = tl.load(x_ptr + r0 + 0)
            x1 = tl.load(x_ptr + r0 + 1)
            x2 = tl.load(x_ptr + r0 + 2)
            x3 = tl.load(x_ptr + r0 + 3)
            x4 = tl.load(x_ptr + r0 + 4)
            x5 = tl.load(x_ptr + r0 + 5)
            x6 = tl.load(x_ptr + r0 + 6)
            x7 = tl.load(x_ptr + r0 + 7)
            if WIDE:
                base = r0.to(tl.int64) * N
            else:
                base = r0 * N
            for j0 in range(0, N, BLOCK_N):
                cols = j0 + tl.arange(0, BLOCK_N)
                if EVEN_COLS:
                    yv = tl.load(y_ptr + cols)
                    tl.store(out_ptr + base + cols, x0 * yv)
                    tl.store(out_ptr + base + N + cols, x1 * yv)
                    tl.store(out_ptr + base + 2 * N + cols, x2 * yv)
                    tl.store(out_ptr + base + 3 * N + cols, x3 * yv)
                    tl.store(out_ptr + base + 4 * N + cols, x4 * yv)
                    tl.store(out_ptr + base + 5 * N + cols, x5 * yv)
                    tl.store(out_ptr + base + 6 * N + cols, x6 * yv)
                    tl.store(out_ptr + base + 7 * N + cols, x7 * yv)
                else:
                    cm = cols < N
                    yv = tl.load(y_ptr + cols, mask=cm, other=0.0)
                    tl.store(out_ptr + base + cols, x0 * yv, mask=cm)
                    tl.store(out_ptr + base + N + cols, x1 * yv, mask=cm)
                    tl.store(out_ptr + base + 2 * N + cols, x2 * yv, mask=cm)
                    tl.store(out_ptr + base + 3 * N + cols, x3 * yv, mask=cm)
                    tl.store(out_ptr + base + 4 * N + cols, x4 * yv, mask=cm)
                    tl.store(out_ptr + base + 5 * N + cols, x5 * yv, mask=cm)
                    tl.store(out_ptr + base + 6 * N + cols, x6 * yv, mask=cm)
                    tl.store(out_ptr + base + 7 * N + cols, x7 * yv, mask=cm)
    else:
        for r in range(pid, M, GRID):
            xv = tl.load(x_ptr + r)
            if WIDE:
                base = r.to(tl.int64) * N
            else:
                base = r * N
            for j0 in range(0, N, BLOCK_N):
                cols = j0 + tl.arange(0, BLOCK_N)
                if EVEN_COLS:
                    yv = tl.load(y_ptr + cols)
                    tl.store(out_ptr + base + cols, xv * yv)
                else:
                    cm = cols < N
                    yv = tl.load(y_ptr + cols, mask=cm, other=0.0)
                    tl.store(out_ptr + base + cols, xv * yv, mask=cm)


def _fast_path_ok(inp, weight):
    if inp.ndim != 1 or weight.ndim != 1:
        return False
    if inp.dtype is not weight.dtype:
        return False
    dtype = inp.dtype
    if dtype.is_complex or dtype is torch.bool:
        return False
    return inp.device.type == _DEVICE_NAME


def _outer_launch(inp, weight):
    if not inp.is_contiguous():
        inp = inp.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    M = inp.numel()
    N = weight.numel()
    out = torch.empty((M, N), device=inp.device, dtype=inp.dtype)
    if M == 0 or N == 0:
        return out
    BLOCK_N = min(_BLOCK_N_CAP, triton.next_power_of_2(N))
    if M >= 8:
        grid = min(_GRID_CAP, triton.cdiv(M, 8))
        rows_per_prog = 8
    else:
        grid = M
        rows_per_prog = 1
    num_warps = 2 if BLOCK_N <= 4096 else 4
    outer_fwd_kernel[(grid,)](
        inp,
        weight,
        out,
        M,
        N,
        grid,
        BLOCK_N=BLOCK_N,
        ROWS_PER_PROG=rows_per_prog,
        EVEN_COLS=(N % BLOCK_N == 0),
        WIDE=(M * N >= _WIDE_ELEMS),
        num_warps=num_warps,
    )
    return out


def _outer_fallback(inp, weight):
    inp1 = inp[:, None]
    weight1 = weight[None, :]
    inp1 = inp1.contiguous()
    weight1 = weight1.contiguous()
    return mul(inp1, weight1)


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("GEMS_ENFLAME OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        ctx.save_for_backward(inp, weight)
        if _fast_path_ok(inp, weight):
            return _outer_launch(inp, weight)
        return _outer_fallback(inp, weight)

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_ENFLAME OUTER_VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_grad = mv(out_grad, weight)
        weight_grad = mv(out_grad.t(), inp)

        return inp_grad, weight_grad


def outer(inp, weight):
    # Inference / no-grad calls skip the autograd.Function.apply context
    # build (same convention as the sibling gcu400 fused ops).
    if (
        not torch.is_grad_enabled() or not (inp.requires_grad or weight.requires_grad)
    ) and _fast_path_ok(inp, weight):
        return _outer_launch(inp, weight)
    return Outer.apply(inp, weight)
