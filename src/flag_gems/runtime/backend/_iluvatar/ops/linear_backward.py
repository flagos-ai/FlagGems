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


@triton.jit
def _grad_input_kernel(
    grad_output,
    weight,
    grad_input,
    M,
    N,
    K,
    stride_gm,
    stride_gn,
    stride_wn,
    stride_wk,
    stride_gim,
    stride_gik,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    EVEN: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    g_ptrs = grad_output + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    w_ptrs = weight + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    if EVEN:
        for n0 in tl.range(0, N, BLOCK_N, num_stages=NUM_STAGES):
            g = tl.load(g_ptrs + n0 * stride_gn)
            w = tl.load(w_ptrs + n0 * stride_wn)
            acc = tl.dot(g, w, acc, input_precision=INPUT_PRECISION)
        out_ptrs = (
            grad_input + offs_m[:, None] * stride_gim + offs_k[None, :] * stride_gik
        )
        tl.store(out_ptrs, acc.to(grad_input.dtype.element_ty))
    else:
        m_mask = offs_m < M
        k_mask = offs_k < K
        for n0 in range(0, N, BLOCK_N):
            offs_n2 = n0 + offs_n
            n_mask2 = offs_n2 < N
            g = tl.load(
                g_ptrs + n0 * stride_gn,
                mask=m_mask[:, None] & n_mask2[None, :],
                other=0.0,
            )
            w = tl.load(
                w_ptrs + n0 * stride_wn,
                mask=n_mask2[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(g, w, acc, input_precision=INPUT_PRECISION)
        out_ptrs = (
            grad_input + offs_m[:, None] * stride_gim + offs_k[None, :] * stride_gik
        )
        tl.store(
            out_ptrs,
            acc.to(grad_input.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )


@triton.jit
def _grad_weight_kernel(
    grad_output,
    input,
    grad_weight,
    M,
    N,
    K,
    stride_gm,
    stride_gn,
    stride_xm,
    stride_xk,
    stride_gwn,
    stride_gwk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    EVEN: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, BLOCK_M)

    g_ptrs = grad_output + offs_m[None, :] * stride_gm + offs_n[:, None] * stride_gn
    x_ptrs = input + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    if EVEN:
        for m0 in tl.range(0, M, BLOCK_M, num_stages=NUM_STAGES):
            g = tl.load(g_ptrs + m0 * stride_gm)
            x = tl.load(x_ptrs + m0 * stride_xm)
            acc = tl.dot(g, x, acc, input_precision=INPUT_PRECISION)
        out_ptrs = (
            grad_weight + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        tl.store(out_ptrs, acc.to(grad_weight.dtype.element_ty))
    else:
        n_mask = offs_n < N
        k_mask = offs_k < K
        for m0 in range(0, M, BLOCK_M):
            offs_m2 = m0 + offs_m
            m_mask2 = offs_m2 < M
            g = tl.load(
                g_ptrs + m0 * stride_gm,
                mask=m_mask2[None, :] & n_mask[:, None],
                other=0.0,
            )
            x = tl.load(
                x_ptrs + m0 * stride_xm,
                mask=m_mask2[:, None] & k_mask[None, :],
                other=0.0,
            )
            acc = tl.dot(g, x, acc, input_precision=INPUT_PRECISION)
        out_ptrs = (
            grad_weight + offs_n[:, None] * stride_gwn + offs_k[None, :] * stride_gwk
        )
        tl.store(
            out_ptrs,
            acc.to(grad_weight.dtype.element_ty),
            mask=n_mask[:, None] & k_mask[None, :],
        )


@triton.jit
def _grad_bias_kernel(
    grad_output,
    grad_bias,
    M,
    N,
    stride_gm,
    stride_gn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)

    g_ptrs = grad_output + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if EVEN:
        for m0 in range(0, M, BLOCK_M):
            g = tl.load(g_ptrs + m0 * stride_gm)
            acc += tl.sum(g.to(tl.float32), axis=0)
        tl.store(grad_bias + offs_n, acc.to(grad_bias.dtype.element_ty))
    else:
        n_mask = offs_n < N
        for m0 in range(0, M, BLOCK_M):
            offs_m2 = m0 + offs_m
            m_mask2 = offs_m2 < M
            g = tl.load(
                g_ptrs + m0 * stride_gm,
                mask=m_mask2[:, None] & n_mask[None, :],
                other=0.0,
            )
            acc += tl.sum(g.to(tl.float32), axis=0)
        tl.store(grad_bias + offs_n, acc.to(grad_bias.dtype.element_ty), mask=n_mask)


def _mask_at(output_mask, idx):
    v = output_mask[idx]
    if hasattr(v, "item"):
        return bool(v.item())
    return bool(v)


def run(input, grad_output, weight, output_mask):
    need_gi = _mask_at(output_mask, 0)
    need_gw = _mask_at(output_mask, 1)
    need_gb = _mask_at(output_mask, 2)

    M, K = input.shape
    _, N = grad_output.shape

    gi = (
        torch.empty((M, K), dtype=input.dtype, device=input.device) if need_gi else None
    )
    gw = (
        torch.empty((N, K), dtype=weight.dtype, device=weight.device)
        if need_gw
        else None
    )
    gb = (
        torch.empty((N,), dtype=grad_output.dtype, device=grad_output.device)
        if need_gb
        else None
    )

    prec = "ieee"

    # Shape/dtype-specialized tiling measured on BI-V150 (16 SMs, 4x4 f32
    # matrix-mad MMA, warp_size=64): 16-warp blocks spread the fp32 acc over
    # 1024 threads and win on the deep-K big GEMMs; fp16 prefers wide-N tiles
    # with 16 warps; the small fp32 gi wants many small BK=32 blocks.
    if M * N * K > 2**30:
        gi_cfg = (128, 64, 128, 16, 3)
        gw_cfg = (64, 128, 128, 16, 2)
    else:
        if input.dtype == torch.float16:
            gi_cfg = (64, 256, 64, 16, 1)
            gw_cfg = (64, 256, 128, 16, 1)
        else:
            gi_cfg = (32, 128, 32, 8, 1)
            gw_cfg = (64, 128, 128, 8, 1)
    gb_cfg = (64, 64, 4)

    if need_gi:
        BM, BN, BK, nw, ns = gi_cfg
        grid = (triton.cdiv(M, BM), triton.cdiv(K, BK))
        _grad_input_kernel[grid](
            grad_output,
            weight,
            gi,
            M,
            N,
            K,
            grad_output.stride(0),
            grad_output.stride(1),
            weight.stride(0),
            weight.stride(1),
            gi.stride(0),
            gi.stride(1),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            INPUT_PRECISION=prec,
            EVEN=(M % BM == 0) and (N % BN == 0) and (K % BK == 0),
            NUM_STAGES=ns,
            num_warps=nw,
        )
    if need_gw:
        BM, BN, BK, nw, ns = gw_cfg
        grid = (triton.cdiv(N, BN), triton.cdiv(K, BK))
        _grad_weight_kernel[grid](
            grad_output,
            input,
            gw,
            M,
            N,
            K,
            grad_output.stride(0),
            grad_output.stride(1),
            input.stride(0),
            input.stride(1),
            gw.stride(0),
            gw.stride(1),
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            INPUT_PRECISION=prec,
            EVEN=(M % BM == 0) and (N % BN == 0) and (K % BK == 0),
            NUM_STAGES=ns,
            num_warps=nw,
        )
    if need_gb:
        BM, BN, nw = gb_cfg
        grid = (triton.cdiv(N, BN),)
        _grad_bias_kernel[grid](
            grad_output,
            gb,
            M,
            N,
            grad_output.stride(0),
            grad_output.stride(1),
            BLOCK_M=BM,
            BLOCK_N=BN,
            EVEN=(M % BM == 0) and (N % BN == 0),
            num_warps=nw,
        )

    return gi, gw, gb
