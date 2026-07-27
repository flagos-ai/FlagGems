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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

if runtime.device.vendor_name == "iluvatar":
    from flag_gems.runtime.backend._iluvatar.ops.bmm import bmm
else:
    from .bmm import bmm

from .mul import mul

logger = logging.getLogger(__name__)

@libentry()
@libtuner(
    configs=runtime.get_tuned_config("baddbmm"),
    key=["M", "N", "K"],
    strategy=["align32", "align32", "align32"],
    warmup=5,
    rep=10,
)
@triton.heuristics(runtime.get_heuristic_config("baddbmm"))
@triton.jit(do_not_specialize=["alpha", "beta"])
def baddbmm_kernel(
    A,
    B,
    O,
    bias,
    alpha,
    beta,
    M,
    N,
    K,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    TILE_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
    DIVISIBLE_N: tl.constexpr,
    DIVISIBLE_K: tl.constexpr,
    bias_batch_stride: tl.constexpr,
    bias_M_stride: tl.constexpr,
    bias_N_stride: tl.constexpr,
):
    # batch offsets
    pid_b = ext.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N
    bias += pid_b * bias_batch_stride

    pidx = ext.program_id(0)
    pidy = ext.program_id(1)

    if GROUP_M == 1:
        pid_m, pid_n = pidx, pidy
    else:
        gridx = ext.num_programs(0)
        gridy = ext.num_programs(1)
        pid = pidx + pidy * gridx
        num_CTA_per_group = gridy * GROUP_M
        group_id = pid // num_CTA_per_group
        inner_group_id = pid % num_CTA_per_group
        GROUP_SIZE = tl.where(
            (group_id * GROUP_M + GROUP_M) > gridx, gridx % GROUP_M, GROUP_M
        )
        pid_m = group_id * GROUP_M + inner_group_id % GROUP_SIZE
        pid_n = inner_group_id // GROUP_SIZE

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    if not DIVISIBLE_M:
        mask_m = offs_m < M
    if not DIVISIBLE_N:
        mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    accumulator = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        if DIVISIBLE_K:
            if DIVISIBLE_M:
                mask_a = None
            else:
                mask_a = mask_m[:, None]
            if DIVISIBLE_N:
                mask_b = None
            else:
                mask_b = mask_n[None, :]
        else:
            mask_k = offs_k < K
            if DIVISIBLE_M:
                mask_a = mask_k[None, :]
            else:
                mask_a = mask_m[:, None] & mask_k[None, :]
            if DIVISIBLE_N:
                mask_b = mask_k[:, None]
            else:
                mask_b = mask_k[:, None] & mask_n[None, :]
        a = tl.load(a_ptrs, mask=mask_a)
        b = tl.load(b_ptrs, mask=mask_b)
        accumulator += tl.dot(a, b, allow_tf32=False)
        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

    bias_ptrs = bias + offs_m[:, None] * bias_M_stride + offs_n[None, :] * bias_N_stride

    if DIVISIBLE_M and DIVISIBLE_N:
        mask_c = None
    else:
        mask_c = True
        if not DIVISIBLE_M:
            mask_c &= offs_m[:, None] < M
        if not DIVISIBLE_N:
            mask_c &= offs_n[None, :] < N

    bi = tl.load(bias_ptrs, mask=mask_c)
    out = accumulator * alpha + bi * beta
    o = out.to(bi.dtype)
    tl.store(o_ptrs, o, mask=mask_c)


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_epilogue_kernel(
    matmul,
    output,
    bias,
    alpha,
    beta,
    total_elements,
    M,
    N,
    bias_batch_stride: tl.constexpr,
    bias_M_stride: tl.constexpr,
    bias_N_stride: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    safe_offsets = tl.minimum(offsets, total_elements - 1)
    row = safe_offsets // N
    column = safe_offsets % N
    batch = row // M
    matrix_row = row % M
    bias_offsets = (
        batch * bias_batch_stride
        + matrix_row * bias_M_stride
        + column * bias_N_stride
    )
    matmul_value = tl.load(matmul + safe_offsets).to(tl.float32)
    bias_value = tl.load(bias + bias_offsets).to(tl.float32)
    result = matmul_value * alpha + bias_value * beta
    tl.store(output + safe_offsets, result, mask=mask)


@libentry()
@triton.jit
def _baddbmm_pad_chunk_kernel(
    source,
    destination,
    M,
    N,
    K,
    start,
    width,
    column_start,
    chunk_N,
    PADDED_DIM: tl.constexpr,
    IS_LHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    outer = ext.program_id(0)
    chunk = ext.program_id(1)
    lanes = chunk * BLOCK + tl.arange(0, BLOCK)
    if IS_LHS:
        batch = outer // PADDED_DIM
        row = outer % PADDED_DIM
        mask = lanes < 256
        valid = (row < M) & (lanes < width)
        safe_row = tl.minimum(row, M - 1)
        safe_reduction = tl.minimum(lanes, width - 1)
        source_offsets = batch * M * K + safe_row * K + start + safe_reduction
        destination_offsets = outer * 256 + lanes
    else:
        batch = outer // 256
        reduction = outer % 256
        mask = lanes < PADDED_DIM
        valid = (reduction < width) & (lanes < chunk_N)
        safe_reduction = tl.minimum(reduction, width - 1)
        safe_column = column_start + tl.minimum(lanes, chunk_N - 1)
        source_offsets = batch * K * N + (start + safe_reduction) * N + safe_column
        destination_offsets = outer * PADDED_DIM + lanes
    value = tl.load(source + source_offsets)
    tl.store(destination + destination_offsets, tl.where(valid, value, 0.0), mask=mask)


@libentry()
@triton.jit(do_not_specialize=["scale"])
def _baddbmm_accumulate_kernel(
    partial,
    output,
    scale,
    total_elements,
    M,
    output_N,
    chunk_N,
    column_start,
    PARTIAL_BATCH_STRIDE: tl.constexpr,
    PARTIAL_M_STRIDE: tl.constexpr,
    FIRST: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = ext.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < total_elements
    safe_offsets = tl.minimum(offsets, total_elements - 1)
    row = safe_offsets // chunk_N
    batch = row // M
    matrix_row = row % M
    column = safe_offsets % chunk_N
    partial_offsets = (
        batch * PARTIAL_BATCH_STRIDE
        + matrix_row * PARTIAL_M_STRIDE
        + column
    )
    output_offsets = (
        batch * M * output_N
        + matrix_row * output_N
        + column_start
        + column
    )
    value = tl.load(partial + partial_offsets).to(tl.float32) * scale
    if not FIRST:
        value += tl.load(output + output_offsets)
    tl.store(output + output_offsets, value, mask=mask)


def _direct_baddbmm(A, B, bias, alpha, beta, out=None):
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    if out is None:
        out = torch.empty((batch, M, N), dtype=A.dtype, device=A.device)
    bbias = torch.broadcast_to(bias, (batch, M, N)).contiguous()

    grid = lambda meta: (
        triton.cdiv(M, meta["TILE_M"]),
        triton.cdiv(N, meta["TILE_N"]),
        batch,
    )
    with torch_device_fn.device(A.device):
        baddbmm_kernel[grid](
            A,
            B,
            out,
            bbias,
            alpha,
            beta,
            M,
            N,
            K,
            bias_batch_stride=bbias.stride(0),
            bias_M_stride=bbias.stride(1),
            bias_N_stride=bbias.stride(2),
        )
    return out


def _chunked_bmm(lhs, rhs, scale):
    lhs = lhs.contiguous()
    rhs = rhs.contiguous()
    batch, M, K = lhs.shape
    _, _, N = rhs.shape
    if K <= 1024:
        bias = torch.zeros((batch, M, N), dtype=lhs.dtype, device=lhs.device)
        return _direct_baddbmm(lhs, rhs, bias, scale, 0.0)

    output = torch.empty((batch, M, N), dtype=torch.float32, device=lhs.device)
    padded_M = triton.cdiv(M, 128) * 128
    padded_N = 1024
    for start in range(0, K, 256):
        end = min(start + 256, K)
        lhs_chunk = torch.empty(
            (batch, padded_M, 256), dtype=lhs.dtype, device=lhs.device
        )
        with torch_device_fn.device(lhs.device):
            _baddbmm_pad_chunk_kernel[(batch * padded_M, 1)](
                lhs,
                lhs_chunk,
                M,
                N,
                K,
                start,
                end - start,
                0,
                1,
                PADDED_DIM=padded_M,
                IS_LHS=True,
                BLOCK=256,
                isCloseVectorization=True,
                buffer_size_limit=2048,
            )
        for column_start in range(0, N, padded_N):
            chunk_N = min(padded_N, N - column_start)
            rhs_chunk = torch.empty(
                (batch, 256, padded_N), dtype=rhs.dtype, device=rhs.device
            )
            with torch_device_fn.device(lhs.device):
                _baddbmm_pad_chunk_kernel[(batch * 256, 1)](
                    rhs,
                    rhs_chunk,
                    M,
                    N,
                    K,
                    start,
                    end - start,
                    column_start,
                    chunk_N,
                    PADDED_DIM=padded_N,
                    IS_LHS=False,
                    BLOCK=1024,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
            partial = bmm(lhs_chunk, rhs_chunk)
            total_elements = batch * M * chunk_N
            grid = (triton.cdiv(total_elements, 1024),)
            with torch_device_fn.device(lhs.device):
                _baddbmm_accumulate_kernel[grid](
                    partial,
                    output,
                    scale,
                    total_elements,
                    M,
                    N,
                    chunk_N,
                    column_start,
                    PARTIAL_BATCH_STRIDE=partial.stride(0),
                    PARTIAL_M_STRIDE=partial.stride(1),
                    FIRST=start == 0,
                    BLOCK=1024,
                    isCloseVectorization=True,
                    buffer_size_limit=2048,
                )
    return output


class BaddbmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bias, A, B, beta, alpha):
        logger.debug("GEMS_KUNLUNXIN BADDBMM_FORWARD")

        ctx.save_for_backward(A, B, bias)
        ctx.alpha = alpha
        ctx.beta = beta

        return _direct_baddbmm(A, B, bias, alpha, beta)

    @staticmethod
    def backward(ctx, grad_output):
        logger.debug("GEMS_KUNLUNXIN BADDBMM_BACKWARD")
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
    grad_bias = mul(d_output, beta)
    if grad_bias.shape != bias.shape:
        # Sum over broadcasted dimensions
        while grad_bias.dim() > bias.dim():
            grad_bias = grad_bias.sum(dim=0)
        for i in range(bias.dim()):
            if bias.shape[i] == 1 and grad_bias.shape[i] > 1:
                grad_bias = grad_bias.sum(dim=i, keepdim=True)
    return grad_bias.view(bias.shape)


def compute_A_grad(d_output, B, alpha):
    output_dtype = B.dtype
    B_T = B.transpose(1, 2)
    if output_dtype in (torch.float16, torch.bfloat16):
        B_T = B_T.to(torch.float32)
        d_output = d_output.to(torch.float32)
    grad_A = _chunked_bmm(d_output, B_T, alpha)
    return grad_A.to(output_dtype)


def compute_B_grad(A, d_output, alpha):
    output_dtype = A.dtype
    A_T = A.transpose(1, 2)
    if output_dtype in (torch.float16, torch.bfloat16):
        A_T = A_T.to(torch.float32)
        d_output = d_output.to(torch.float32)
    grad_B = _chunked_bmm(A_T, d_output, alpha)
    return grad_B.to(output_dtype)


def baddbmm(bias, A, B, beta=1.0, alpha=1.0):
    return BaddbmmFunction.apply(
        bias.contiguous(),
        A.contiguous(),
        B.contiguous(),
        beta,
        alpha,
    )


def baddbmm_out(bias, A, B, beta=1.0, alpha=1.0, *, out):
    result = baddbmm(bias, A, B, beta=beta, alpha=alpha)
    if tuple(out.shape) != tuple(result.shape):
        out.resize_(result.shape)
    out.copy_(result)
    return out
