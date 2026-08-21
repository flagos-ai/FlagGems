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

from flag_gems.utils import libentry

from ..ops import mul, mv_cluster

logger = logging.getLogger(__name__)

# ============================================================================
# outer(x, y)[i, j] = x[i] * y[j] on Kunlunxin XPU.
#
# XPU 2026-08 environment findings (all measured on XPU 6):
#   1. The historical broadcast-mul path (x[:, None] / y[None, :]) emits
#      0-stride discrete gathers; per-element div/mod in a flat kernel also
#      costs ~2-3x over the achievable copy bandwidth.
#   2. Register-level 2D broadcast (xv[:, None] * yv[None, :], tl.broadcast_to)
#      is mis-compiled for larger shapes on this backend (wrong values), so a
#      broadcast-free formulation is mandatory.
#   3. tl.dot with operands read from DRAM is CORRECT (fp32/fp16 bit-exact
#      with input_precision="ieee"); it is the fastest formulation found.
#   outer == (x*(1/16)) @ ones? no — expand: out[i,j] = sum_k a[i,k]*b[k,j]
#   with a[i,k] = x[i]/16 and b[k,j] = y[j] (K=16 identical terms): the dot
#   sum reproduces x[i]*y[j] exactly (division by 16 is an exact exponent
#   shift for any float dtype).
#
# Fast path (the only path in the perf matrix — all benchmark shapes are
# divisible by (64, 128)): materialize a=(M,16) / b=(16,N) with the tiny
# torch ops below, then one tl.dot tile kernel. Output uses torch.empty_strided
# to skip the registered-aten `empty` zero-fill tax.
#
# Fallback path (non-divisible dims, e.g. 5333x497 / 1x32 in the accuracy
# suite): flat 1D kernel over M*N with clamped loads (proven stable pattern).
# Complex inputs keep the old broadcast-mul path (mul handles complex).
# ============================================================================

_KN = 16
_DOT_BM = 256
_DOT_BN = 512


@libentry()
@triton.jit
def outer_dot_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    N,
    KN: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, KN)
    a = tl.load(a_ptr + offs_m[:, None] * KN + offs_k[None, :])  # (BM, KN)
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])  # (KN, BN)
    res = tl.dot(a, b, input_precision="ieee")  # (BM, BN)
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        res.to(out_ptr.dtype.element_ty),
    )


@libentry()
@triton.jit
def outer_flat_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    total,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        mask = offs < total
        safe = tl.where(mask, offs, 0)
        i = safe // N
        j = safe - i * N
        xv = tl.load(x_ptr + i)
        yv = tl.load(y_ptr + j)
        tl.store(out_ptr + offs, tl.where(mask, xv * yv, 0.0), mask=mask)
    else:
        i = offs // N
        j = offs - i * N
        xv = tl.load(x_ptr + i)
        yv = tl.load(y_ptr + j)
        tl.store(out_ptr + offs, xv * yv)


def _outer_dot(inp, weight, bm, bn):
    m = inp.shape[0]
    n = weight.shape[0]
    kn = _KN
    a = inp.unsqueeze(-1).repeat(1, kn) * (1.0 / kn)  # (M, KN) exact copies
    b = weight.unsqueeze(0).repeat(kn, 1)  # (KN, N)
    out = torch.empty_strided((m, n), (n, 1), dtype=inp.dtype, device=inp.device)
    grid = (m // bm, n // bn)
    outer_dot_kernel[grid](a, b, out, n, KN=kn, BM=bm, BN=bn)
    return out


def _outer_flat(inp, weight):
    m = inp.shape[0]
    n = weight.shape[0]
    total = m * n
    out = torch.empty_strided((m, n), (n, 1), dtype=inp.dtype, device=inp.device)
    if total == 0:
        return out
    block = 4096 if total >= 4096 else triton.next_power_of_2(total)
    grid = (triton.cdiv(total, block),)
    outer_flat_kernel[grid](
        inp, weight, out, total, N=n, BLOCK=block, NEED_MASK=total % block != 0
    )
    return out


class Outer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight):
        logger.debug("GEMS_KUNLUNXIN OUTER")
        assert inp.ndim == 1 and weight.ndim == 1, "Invalid input"
        inp = inp.contiguous()
        weight = weight.contiguous()
        if inp.is_complex() or weight.is_complex():
            inp1 = inp[:, None].contiguous()
            weight1 = weight[None, :].contiguous()
            out = mul(inp1, weight1)
        elif (
            inp.shape[0] > 0
            and weight.shape[0] > 0
            and inp.shape[0] % 256 == 0
            and weight.shape[0] % 512 == 0
        ):
            out = _outer_dot(inp, weight, 256, 512)
        elif (
            inp.shape[0] > 0
            and weight.shape[0] > 0
            and inp.shape[0] % 64 == 0
            and weight.shape[0] % 128 == 0
        ):
            out = _outer_dot(inp, weight, 64, 128)
        else:
            out = _outer_flat(inp, weight)
        ctx.save_for_backward(inp, weight)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        logger.debug("GEMS_KUNLUNXIN OUTER_VJP")
        assert out_grad.ndim == 2, "invalide out_grad shape"

        inp, weight = ctx.saved_tensors

        inp_grad = mv_cluster(out_grad, weight)
        weight_grad = mv_cluster(out_grad.t().contiguous(), inp)

        return inp_grad, weight_grad


def outer(inp, weight):
    return Outer.apply(inp, weight)