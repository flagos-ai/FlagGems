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

#      gather loads + scalar FMA (~77x slower dot loops).
#   2. Masked loads with negative addresses are not safe (silent wrong
#      values). Positive out-of-bounds tails under mask are fine.
# Negative tap windows are therefore handled without per-lane clamps:
#   - grad_weight re-parameterizes the reduction so both operands are read at
#     non-negative rows (in_off = max(dk, 0), go_off = max(-dk, 0)).
#   - grad_input reads grad_output through a front-padded copy whenever
#     front = kw-1-pad > 0 (tiny staging kernel); otherwise directly.

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _conv_tbc_grad_weight_kernel(
    IN,
    GO,
    GW,
    rows,
    in_rows,
    batch,
    in_c,
    out_c,
    pad,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    k = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_o = tl.program_id(2)
    # Tap window rows are r + dk with dk = (k - pad) * batch; re-parameterize
    # the reduction so both reads start at a non-negative row.
    dk = (k - pad) * batch
    in_off = tl.maximum(dk, 0)
    go_off = tl.maximum(-dk, 0)
    n_valid = tl.minimum(in_rows - in_off, rows - go_off)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = offs_c < in_c
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    o_ok = offs_o < out_c

    acc = tl.zeros((BLOCK_C, BLOCK_O), dtype=tl.float32)
    for r0 in range(0, n_valid, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        a = tl.load(
            IN + (offs_r + in_off)[:, None] * in_c + offs_c[None, :],
            mask=((offs_r + in_off) < in_rows)[:, None] & c_ok[None, :],
            other=0.0,
        )
        g = tl.load(
            GO + (offs_r + go_off)[:, None] * out_c + offs_o[None, :],
            mask=((offs_r + go_off) < rows)[:, None] & o_ok[None, :],
            other=0.0,
        )
        acc += tl.dot(tl.trans(a), g, allow_tf32=False)

    gw_off = (k * in_c + offs_c[:, None]) * out_c + offs_o[None, :]
    tl.store(
        GW + gw_off, acc.to(GW.dtype.element_ty), mask=c_ok[:, None] & o_ok[None, :]
    )


@triton.jit
def _conv_tbc_grad_input_kernel(
    GOP,
    W,
    GI,
    gop_rows,
    rows_i,
    batch,
    in_c,
    out_c,
    kw,
    jfront,
    BLOCK_R: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    r_ok = offs_r < rows_i
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_ok = offs_c < in_c

    acc = tl.zeros((BLOCK_R, BLOCK_C), dtype=tl.float32)
    for j in range(0, kw):
        # gop row r + (j - jfront) * batch is always non-negative: gop is the
        # front-padded grad_output when front > 0 (jfront = 0), or grad_output
        # itself when front <= 0 (jfront = front <= 0).
        g_row = offs_r + (j - jfront) * batch
        g_ok = r_ok & (g_row < gop_rows)
        for o0 in range(0, out_c, BLOCK_O):
            offs_o = o0 + tl.arange(0, BLOCK_O)
            o_ok = offs_o < out_c
            g = tl.load(
                GOP + g_row[:, None] * out_c + offs_o[None, :],
                mask=g_ok[:, None] & o_ok[None, :],
                other=0.0,
            )
            w = tl.load(
                W
                + (kw - 1 - j) * in_c * out_c
                + offs_c[:, None] * out_c
                + offs_o[None, :],
                mask=c_ok[:, None] & o_ok[None, :],
                other=0.0,
            )
            acc += tl.dot(g, tl.trans(w), allow_tf32=False)

    gi_off = offs_r[:, None] * in_c + offs_c[None, :]
    tl.store(
        GI + gi_off, acc.to(GI.dtype.element_ty), mask=r_ok[:, None] & c_ok[None, :]
    )


@triton.jit
def _conv_tbc_frontpad_kernel(
    GO,
    GOP,
    total,
    boundary,
    BLOCK: tl.constexpr,
):
    """Prepend zero rows to grad_output: gop = [zeros, grad_output] flat."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    in_copy = offs >= boundary
    src = tl.maximum(offs - boundary, 0)
    v = tl.load(GO + src, mask=in_copy & (offs < total), other=0.0)
    tl.store(GOP + offs, v, mask=offs < total)


@triton.jit
def _conv_tbc_bias_kernel(
    GO,
    BIAS,
    rows,
    out_c,
    BLOCK_R: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    pid_o = tl.program_id(0)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    o_ok = offs_o < out_c
    ones = tl.full((16, BLOCK_R), 1.0, dtype=GO.dtype.element_ty)
    acc = tl.zeros((16, BLOCK_O), dtype=tl.float32)
    for r0 in range(0, rows, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        r_ok = offs_r < rows
        g = tl.load(
            GO + offs_r[:, None] * out_c + offs_o[None, :],
            mask=r_ok[:, None] & o_ok[None, :],
            other=0.0,
        )
        acc += tl.dot(ones, g, allow_tf32=False)
    row0 = tl.arange(0, 16)[:, None] == 0
    ptrs = BIAS + offs_o[None, :] + row0 * 0
    tl.store(ptrs, acc.to(BIAS.dtype.element_ty), mask=row0 & o_ok[None, :])


@triton.jit
def _conv_tbc_bias_partial_kernel(
    GO,
    PART,
    rows,
    out_c,
    slab,
    BLOCK_R: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_o = tl.program_id(1)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    o_ok = offs_o < out_c
    r_lo = pid_s * slab
    r_hi = tl.minimum(r_lo + slab, rows)
    ones = tl.full((16, BLOCK_R), 1.0, dtype=GO.dtype.element_ty)
    acc = tl.zeros((16, BLOCK_O), dtype=tl.float32)
    for r0 in range(r_lo, r_hi, BLOCK_R):
        offs_r = r0 + tl.arange(0, BLOCK_R)
        g = tl.load(
            GO + offs_r[:, None] * out_c + offs_o[None, :],
            mask=(offs_r < r_hi)[:, None] & o_ok[None, :],
            other=0.0,
        )
        acc += tl.dot(ones, g, allow_tf32=False)
    row0 = tl.arange(0, 16)[:, None] == 0
    ptrs = PART + pid_s * out_c + offs_o[None, :] + row0 * 0
    tl.store(ptrs, acc, mask=row0 & o_ok[None, :])


@triton.jit
def _conv_tbc_bias_reduce_kernel(
    PART,
    BIAS,
    out_c,
    n_split,
    BLOCK_O: tl.constexpr,
):
    pid_o = tl.program_id(0)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    o_ok = offs_o < out_c
    o_cl = tl.minimum(offs_o, out_c - 1)
    acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
    for s in range(0, n_split):
        acc += tl.load(PART + s * out_c + o_cl, mask=o_ok, other=0.0)
    tl.store(BIAS + offs_o, acc.to(BIAS.dtype.element_ty), mask=o_ok)


def _grad_bias(grad_output, out, rows, out_c):
    out_c = int(out_c)
    block_o = 32
    o_blocks = triton.cdiv(out_c, block_o)
    if rows <= 1024:
        _conv_tbc_bias_kernel[(o_blocks,)](
            grad_output,
            out,
            rows,
            out_c,
            BLOCK_R=512,
            BLOCK_O=block_o,
            num_warps=4,
            num_stages=1,
        )
    else:
        n_split = max(
            1, min(32, (32 + o_blocks - 1) // o_blocks, triton.cdiv(rows, 128))
        )
        slab = triton.cdiv(rows, n_split)
        n_split = triton.cdiv(rows, slab)
        part = torch.empty(
            (n_split, out_c), device=grad_output.device, dtype=torch.float32
        )
        _conv_tbc_bias_partial_kernel[(n_split, o_blocks)](
            grad_output,
            part,
            rows,
            out_c,
            slab,
            BLOCK_R=128,
            BLOCK_O=block_o,
            num_warps=4,
            num_stages=1,
        )
        _conv_tbc_bias_reduce_kernel[(o_blocks,)](
            part,
            out,
            out_c,
            n_split,
            BLOCK_O=block_o,
            num_warps=1,
            num_stages=1,
        )


def conv_tbc_backward(grad_output, input, weight, bias, pad):
    logger.debug("GEMS CONV_TBC_BACKWARD")

    ilen, batch, in_c = input.shape
    kw, _, out_c = weight.shape
    olen = grad_output.shape[0]
    rows = olen * batch
    rows_i = ilen * batch
    front = kw - 1 - pad

    grad_output = grad_output.contiguous()
    input = input.contiguous()
    weight = weight.contiguous()

    grad_weight = torch.empty(
        (kw, in_c, out_c), device=grad_output.device, dtype=weight.dtype
    )
    grad_input = torch.empty(
        (ilen, batch, in_c), device=grad_output.device, dtype=input.dtype
    )
    grad_bias = torch.empty((out_c,), device=grad_output.device, dtype=weight.dtype)

    if rows > 0:
        # ---- grad_weight: tap-fused GEMM, no im2col staging ----
        _conv_tbc_grad_weight_kernel[
            (kw, triton.cdiv(in_c, 64), triton.cdiv(out_c, 64))
        ](
            input,
            grad_output,
            grad_weight,
            rows,
            rows_i,
            batch,
            in_c,
            out_c,
            pad,
            BLOCK_R=512,
            BLOCK_C=64,
            BLOCK_O=64,
            num_warps=4,
            num_stages=1,
        )

        # ---- grad_bias: split column reduction ----
        _grad_bias(grad_output, grad_bias, rows, out_c)
    else:
        grad_weight.zero_()
        grad_bias.zero_()

    if rows_i > 0:
        # ---- grad_input: tap-fused GEMM with flipped-weight tap reads ----
        if front > 0:
            pad_rows = front * batch
            gop = torch.empty(
                (pad_rows + rows, out_c),
                device=grad_output.device,
                dtype=grad_output.dtype,
            )
            total = (pad_rows + rows) * out_c
            _conv_tbc_frontpad_kernel[(triton.cdiv(total, 4096),)](
                grad_output,
                gop,
                total,
                pad_rows * out_c,
                BLOCK=4096,
                num_warps=4,
                num_stages=1,
            )
            gop_ptr, jfront, gop_rows = gop, 0, pad_rows + rows
        else:
            gop_ptr, jfront, gop_rows = grad_output, front, rows
        _conv_tbc_grad_input_kernel[(triton.cdiv(rows_i, 256), triton.cdiv(in_c, 64))](
            gop_ptr,
            weight,
            grad_input,
            gop_rows,
            rows_i,
            batch,
            in_c,
            out_c,
            kw,
            jfront,
            BLOCK_R=256,
            BLOCK_C=64,
            BLOCK_O=128,
            num_warps=4,
            num_stages=1,
        )
    else:
        grad_input.zero_()

    return grad_input, grad_weight, grad_bias
