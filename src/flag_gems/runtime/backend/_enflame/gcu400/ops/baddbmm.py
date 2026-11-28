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

from .bmm import (
    _16BIT_TILE_LADDER,
    _DSM_HARD_LIMIT_BYTES,
    _SAFE_FALLBACK_CONFIG,
    _clamp_tile,
    _select_gemm_config,
    bmm,
)
from .mul import mul

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Every shape runs the Triton GEMM below: the operator must stay a Triton
# implementation on this backend, so there is no vendor-kernel fallback.  The
# tile selection is shared with bmm.py (single source of truth for the GEMM
# core), and large problems rely on that table's software pipeline.
# ---------------------------------------------------------------------------

# baddbmm's bias epilogue costs one extra accumulator-sized local buffer on top
# of the fp32 accumulator, so this kernel needs roughly twice the local budget
# the bmm.py tile table was sized for.  Measured on (16,4096,4096,4096) fp16
# with the shared wide tile (512,256,64): the compiler reports
# "required total:1146880" (= 2 * 512*256*4 + one 96KiB staged A/B tile)
# against the 917504 B pool, i.e. exactly the doubled accumulator plus one
# stage.  Every launch then fell back to the 64x64 retry tile, and at
# (B=16, M=N=4096) that retry needs 16 * 64 * 64 = 65536 programs -- one over
# the hardware grid.x limit -- so the operator failed instead of just running
# slower.
_ACC_BYTES = 4


def _tile_fits(block_m, block_n, block_k):
    """Does a tile fit the DSM pool next to baddbmm's two accumulator-sized
    local buffers (the fp32 accumulator and the bias tile)?"""
    return (
        2 * block_m * block_n * _ACC_BYTES + (block_m * block_k + block_k * block_n) * 2
        <= _DSM_HARD_LIMIT_BYTES
    )


def _baddbmm_stages(block_m, block_n, block_k, max_stages=4):
    """Largest pipeline depth whose staged A/B tiles still fit next to those two
    accumulator-sized buffers."""
    acc = block_m * block_n * _ACC_BYTES
    for stages in range(max_stages, 0, -1):
        staged = (block_m * block_k + block_k * block_n) * 2 * stages
        if 2 * acc + staged <= _DSM_HARD_LIMIT_BYTES:
            return stages
    return 1


def _select_baddbmm_config(Batch, M, N, K, dtype):
    # baddbmm shares the batched GEMM core with bmm and only adds a stride-based
    # bias epilogue, so the tile table lives in bmm.py (single source of truth).
    # That table also clamps every tile to the problem dimensions and fits the
    # software pipeline into the DSM budget, which is what used to make the
    # N == 1 accuracy cases fail to compile.
    config = _select_gemm_config(Batch, M, N, K, dtype)
    block_m, block_n, block_k, num_warps, num_stages, even = config
    if dtype not in (torch.float16, torch.bfloat16) or _tile_fits(
        block_m, block_n, block_k
    ):
        return config
    # The wide 16-bit tile does not fit this kernel's local budget: walk the
    # shared ladder down to the first tile that does.  The bias tile shrinks
    # with the output tile, so (256,256,*) fits where (512,256,*) does not.
    for cand_m, cand_n, cand_k, cand_warps in _16BIT_TILE_LADDER:
        cand_m = _clamp_tile(M, cand_m)
        cand_n = _clamp_tile(N, cand_n)
        cand_k = _clamp_tile(K, cand_k, floor=16)
        if not _tile_fits(cand_m, cand_n, cand_k):
            continue
        num_stages = _baddbmm_stages(cand_m, cand_n, cand_k)
        even = M % cand_m == 0 and N % cand_n == 0 and K % cand_k == 0
        return cand_m, cand_n, cand_k, cand_warps, num_stages, even
    return config


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    C,
    bias,
    alpha,
    beta,
    Batch,
    M,
    N,
    K,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_bias_b: tl.constexpr,
    stride_bias_m: tl.constexpr,
    stride_bias_n: tl.constexpr,
    dot_out_dtype: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = grid_m * grid_n

    pid_b = pid // tiles_per_batch
    pid_mn = pid % tiles_per_batch

    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = (
        A
        + pid_b * stride_ab
        + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    )
    b_ptrs = (
        B
        + pid_b * stride_bb
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=dot_out_dtype)
    # EVEN is a compile-time constant, so the compiler statically trims it to a single loop,
    # avoiding a runtime branch interrupting the tl.dot pipeline (a runtime branch degrades dot to scalar compute).
    if EVEN:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b, out_dtype=dot_out_dtype, allow_tf32=False)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = (
        C
        + pid_b * stride_cb
        + offs_cm[:, None] * stride_cm
        + offs_cn[None, :] * stride_cn
    )
    # bias participates in the epilogue with its original stride (0-stride is broadcast), avoiding broadcast_to+
    # contiguous full materialization of a whole (B,M,N) copy.
    bias_ptrs = (
        bias
        + pid_b * stride_bias_b
        + offs_cm[:, None] * stride_bias_m
        + offs_cn[None, :] * stride_bias_n
    )

    if EVEN:
        bi = tl.load(bias_ptrs)
        out = acc * alpha + bi * beta
        tl.store(c_ptrs, out.to(C.dtype.element_ty))
    else:
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        bi = tl.load(bias_ptrs, mask=c_mask, other=0.0)
        out = acc * alpha + bi * beta
        tl.store(c_ptrs, out.to(C.dtype.element_ty), mask=c_mask)


def _launch_baddbmm(A, B, bias, out, Batch, M, N, K, alpha, beta):
    # Centralize config dispatch and kernel launch, shared by baddbmm / baddbmm_out.
    block_m, block_n, block_k, num_warps, num_stages, even = _select_baddbmm_config(
        Batch, M, N, K, A.dtype
    )
    if bias.dim() == 3 and bias.shape == (Batch, M, N):
        bbias = bias
    else:
        # bias keeps its own strides (0 strides for broadcast), so no (B, M, N)
        # materialization is needed.
        bbias = bias.broadcast_to((Batch, M, N))
    args = (
        A,
        B,
        out,
        bbias,
        alpha,
        beta,
        Batch,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        bbias.stride(0),
        bbias.stride(1),
        bbias.stride(2),
    )
    dot_dtype = tl.float64 if A.dtype == torch.float64 else tl.float32

    def _issue(bm, bn, bk, warps, stages, ev):
        grid = (Batch * triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        baddbmm_kernel[grid](
            *args,
            dot_out_dtype=dot_dtype,
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=8,
            EVEN=ev,
            num_warps=warps,
            num_stages=stages,
        )

    with torch_device_fn.device(A.device):
        try:
            _issue(block_m, block_n, block_k, num_warps, num_stages, even)
        except triton.runtime.errors.OutOfResources:
            # Tiles exceed the hardware budget for this shape: retry once with
            # the conservative tile instead of failing the operator.  The safe
            # tile must still keep the program count inside the grid.x limit:
            # 64x64 already needs 65536 programs at (B=16, M=N=4096), one over
            # the limit, so grow the retry tile until the grid is legal.
            block_m, block_n, block_k, num_warps, num_stages = _SAFE_FALLBACK_CONFIG
            block_m = _clamp_tile(M, block_m)
            block_n = _clamp_tile(N, block_n)
            block_k = _clamp_tile(K, block_k, floor=16)
            while (
                Batch * triton.cdiv(M, block_m) * triton.cdiv(N, block_n) > 65535
                and block_m < 512
                and block_n < 512
            ):
                block_m *= 2
                block_n *= 2
            even = M % block_m == 0 and N % block_n == 0 and K % block_k == 0
            _issue(block_m, block_n, block_k, num_warps, num_stages, even)


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_ENFLAME BADDBMM_FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        batch, M, K = A.shape
        _, _, N = B.shape
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
        _launch_baddbmm(A, B, bias, out, batch, M, N, K, alpha, beta)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_ENFLAME BADDBMM_BACKWARD")
        A, B, bias = ctx.saved_tensors

        grad_A = None
        grad_B = None
        grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_bias = compute_bias_grad(grad_output, ctx.beta, bias)
        if ctx.needs_input_grad[1]:
            grad_A = compute_A_grad(grad_output, B, ctx.alpha)
        if ctx.needs_input_grad[2]:
            grad_B = compute_B_grad(A, grad_output, ctx.alpha)

        return grad_bias, grad_A, grad_B, None, None


def compute_bias_grad(d_output, beta, bias):
    # The reference baddbmm backward multiplies and reduces in the upcast
    # (float32) reference dtype, so a low-precision beta * sum() loses up to a
    # whole fp16 ULP on entries where the eight broadcast terms cancel and can
    # exceed the 1e-4 * reduce_dim tolerance (measured maxdiff 9.8e-4 vs
    # atol 8e-4 at batch=2, M=4). Reduce in float32 and cast back, mirroring
    # compute_A_grad / compute_B_grad below.
    work_dtype = (
        torch.float32
        if d_output.dtype in (torch.float16, torch.bfloat16)
        else d_output.dtype
    )
    grad_bias = mul(d_output.to(work_dtype), beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape).to(bias.dtype)


def compute_A_grad(d_output, B, alpha):
    B_T = B.transpose(1, 2)
    if B.dtype == torch.float16:
        Bcopy = B_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul1 = bmm(dcopye, Bcopy)
        grad_A = mul(mul1, alpha)
        grad_A = grad_A.to(torch.float16)
    else:
        mul1 = bmm(d_output, B_T)
        grad_A = mul(mul1, alpha)
    return grad_A


def compute_B_grad(A, d_output, alpha):
    A_T = A.transpose(1, 2)
    if A.dtype == torch.float16:
        Acopy = A_T.to(torch.float32)
        dcopye = d_output.to(torch.float32)
        mul2 = bmm(Acopy, dcopye)
        grad_B = mul(mul2, alpha)
        grad_B = grad_B.to(torch.float16)
    else:
        mul2 = bmm(A_T, d_output)
        grad_B = mul(mul2, alpha)
    return grad_B


def baddbmm_out(bias, A, B, *, beta=1.0, alpha=1.0, out):
    logger.debug("GEMS_ENFLAME BADDBMM_OUT")
    batch, M, K = A.shape
    _, _, N = B.shape
    assert (
        out.shape == (batch, M, N) and out.dtype == A.dtype
    ), "Incompatible output shape or dtype for baddbmm.out"
    _launch_baddbmm(A, B, bias, out, batch, M, N, K, alpha, beta)
    return out


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    logger.debug("GEMS_ENFLAME BADDBMM")
    return BaddbmmFunction.apply(bias, A, B, beta, alpha)
