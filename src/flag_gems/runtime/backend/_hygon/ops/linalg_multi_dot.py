# Copyright (c) 2024 FlagGems. All rights reserved.
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

from flag_gems.ops.linalg_multi_dot import _matrix_chain_order, _validate_and_prepare

logger = logging.getLogger(__name__)
_KERNELS = {}
_PLANS = {}


@triton.jit
def _mm_kernel(
    A,
    B,
    C,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    F32_DOT: tl.constexpr,
    F64: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    if F64:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.sum(a[:, :, None] * b[None, :, :], axis=1)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            a = tl.load(
                A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            b = tl.load(
                B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            if F32_DOT:
                acc = tl.dot(
                    a.to(tl.float32), b.to(tl.float32), acc, input_precision="ieee"
                )
            else:
                acc = tl.dot(a, b, acc)
    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(C.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


@triton.jit
def _fused_chain_kernel(
    A0,
    A1,
    A2,
    A3,
    OUT,
    s_a0r,
    s_a0c,
    s_a1r,
    s_a1c,
    s_a2r,
    s_a2c,
    s_a3r,
    s_a3c,
    s_or,
    s_oc,
    M,
    N,
    K0,
    K1,
    C,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LEFT_PROD: tl.constexpr,
    RIGHT_PROD: tl.constexpr,
    IEEE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for c0 in range(0, C, BLOCK_K):
        offs_c = c0 + tl.arange(0, BLOCK_K)
        mask_c = offs_c < C
        if LEFT_PROD:
            left = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
            for k0 in range(0, K0, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K0
                a = tl.load(
                    A0 + offs_m[:, None] * s_a0r + offs_k[None, :] * s_a0c,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                b = tl.load(
                    A1 + offs_k[:, None] * s_a1r + offs_c[None, :] * s_a1c,
                    mask=mask_k[:, None] & mask_c[None, :],
                    other=0.0,
                )
                if IEEE:
                    left = tl.dot(a, b, left, input_precision="ieee")
                else:
                    left = tl.dot(a, b, left)
        else:
            left = tl.load(
                A0 + offs_m[:, None] * s_a0r + offs_c[None, :] * s_a0c,
                mask=mask_m[:, None] & mask_c[None, :],
                other=0.0,
            ).to(tl.float32)
        if RIGHT_PROD:
            right = tl.zeros((BLOCK_K, BLOCK_N), dtype=tl.float32)
            for k1 in range(0, K1, BLOCK_K):
                offs_k = k1 + tl.arange(0, BLOCK_K)
                mask_k = offs_k < K1
                a = tl.load(
                    A2 + offs_c[:, None] * s_a2r + offs_k[None, :] * s_a2c,
                    mask=mask_c[:, None] & mask_k[None, :],
                    other=0.0,
                )
                b = tl.load(
                    A3 + offs_k[:, None] * s_a3r + offs_n[None, :] * s_a3c,
                    mask=mask_k[:, None] & mask_n[None, :],
                    other=0.0,
                )
                if IEEE:
                    right = tl.dot(a, b, right, input_precision="ieee")
                else:
                    right = tl.dot(a, b, right)
        else:
            right = tl.load(
                A2 + offs_c[:, None] * s_a2r + offs_n[None, :] * s_a2c,
                mask=mask_c[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
        left = left.to(A0.dtype.element_ty)
        right = right.to(A2.dtype.element_ty)
        if IEEE:
            acc = tl.dot(left, right, acc, input_precision="ieee")
        else:
            acc = tl.dot(left, right, acc)
    tl.store(
        OUT + offs_m[:, None] * s_or + offs_n[None, :] * s_oc,
        acc.to(OUT.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _fused_blocks(dims, s, left_prod, right_prod):
    """Balance repeated partial products across output tiles."""
    M = dims[0]
    N = dims[-1]
    C = dims[s]
    k0 = dims[1] if left_prod else 0
    k1 = dims[s + 1] if right_prod else 0
    left_all = M * k0 * C if left_prod else 0
    right_all = C * k1 * N if right_prod else 0
    d = M * C * N
    best = None
    best_bm = 16
    best_bn = 16
    for bm in (16, 32, 64, 128):
        if bm > 16 and bm > M:
            continue
        nm = (M + bm - 1) // bm
        for bn in (16, 32, 64, 128):
            if bn > 16 and bn > N:
                continue
            nn = (N + bn - 1) // bn
            if nm * nn > 64:
                continue
            score = left_all / nm + right_all / nn + d / (nm * nn)
            if best is None or score < best:
                best = score
                best_bm = bm
                best_bn = bn
    mx = C if C > k0 else k0
    if k1 > mx:
        mx = k1
    bk = 16 if mx <= 16 else 32 if mx <= 32 else 64
    return (best_bm, best_bn, bk)


def _fused(arrays, split, out=None):
    dims = tuple([arrays[0].shape[0]] + [t.shape[1] for t in arrays])
    m, n, c = (dims[0], dims[-1], dims[split])
    left_prod = split == 2
    right_prod = len(arrays) - split == 2
    a0 = arrays[0]
    a1 = arrays[1] if left_prod else a0
    a2 = arrays[split]
    a3 = arrays[split + 1] if right_prod else a2
    if out is None:
        out = torch.empty((m, n), dtype=a0.dtype, device=a0.device)
    if not m or not n:
        return out
    bm, bn, bk = _fused_blocks(dims, split, left_prod, right_prod)
    args = (
        a0,
        a1,
        a2,
        a3,
        out,
        *a0.stride(),
        *a1.stride(),
        *a2.stride(),
        *a3.stride(),
        *out.stride(),
        m,
        n,
        dims[1] if left_prod else 0,
        dims[split + 1] if right_prod else 0,
        c,
        bm,
        bn,
        bk,
        left_prod,
        right_prod,
        a0.dtype == torch.float32,
    )
    key = (
        "fused",
        a0.get_device(),
        a0.dtype,
        dims,
        split,
        tuple((a.stride() for a in (a0, a1, a2, a3, out))),
        tuple((a.data_ptr() % 16 == 0 for a in (a0, a1, a2, a3, out))),
    )
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), 1, 1)
    compiled = _KERNELS.get(key)
    if compiled is None:
        compiled = _fused_chain_kernel[grid](*args, num_warps=4)
        if compiled is not None:
            if len(_KERNELS) >= 256:
                _KERNELS.pop(next(iter(_KERNELS)))
            _KERNELS[key] = compiled
    else:
        compiled[grid](*args, stream=torch.cuda.current_stream().cuda_stream)
    return out


def _mm(left, right, out=None):
    m, k = left.shape
    n = right.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=left.dtype, device=left.device)
    if not m or not n:
        return out
    bm = 16 if m <= 16 else 32
    bn = 16 if n <= 16 else 32
    bk = 16 if k <= 16 else 32
    if left.dtype == torch.float64:
        bm = bn = bk = 16
    args = (
        left,
        right,
        out,
        m,
        n,
        k,
        *left.stride(),
        *right.stride(),
        *out.stride(),
        bm,
        bn,
        bk,
        left.dtype == torch.float32,
        left.dtype == torch.float64,
    )
    key = (
        left.get_device(),
        left.dtype,
        m,
        n,
        k,
        left.stride(),
        right.stride(),
        out.stride(),
        bm,
        bn,
        bk,
        tuple((t.data_ptr() % 16 == 0 for t in (left, right, out))),
    )
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), 1, 1)
    compiled = _KERNELS.get(key)
    if compiled is None:
        compiled = _mm_kernel[grid](*args, num_warps=4)
        if compiled is not None:
            if len(_KERNELS) >= 256:
                _KERNELS.pop(next(iter(_KERNELS)))
            _KERNELS[key] = compiled
    else:
        compiled[grid](*args, stream=torch.cuda.current_stream().cuda_stream)
    return out


def _chain(arrays, out=None):
    count = len(arrays)
    if count == 2:
        return _mm(*arrays, out=out)
    dims = tuple([arrays[0].shape[0]] + [t.shape[1] for t in arrays])
    splits = _PLANS.get(dims)
    if splits is None:
        splits = _matrix_chain_order(arrays)
        if count == 3:
            m, k, c, n = dims
            splits[0][2] = 1 if m * c * (k + n) <= k * n * (m + c) else 0
        if len(_PLANS) >= 128:
            _PLANS.pop(next(iter(_PLANS)))
        _PLANS[dims] = splits

    def evaluate(start, end, destination=None):
        if start == end:
            return arrays[start]
        split = splits[start][end]
        if arrays[0].dtype != torch.float64 and end - start in (2, 3):
            left_count = split - start + 1
            right_count = end - split
            if left_count <= 2 and right_count <= 2:
                return _fused(arrays[start : end + 1], left_count, destination)
        return _mm(evaluate(start, split), evaluate(split + 1, end), out=destination)

    return evaluate(0, count - 1, out)


def linalg_multi_dot(tensors):
    logger.debug("GEMS_HYGON LINALG_MULTI_DOT")
    arrays, shape = _validate_and_prepare(tensors)
    return _chain(arrays).view(shape)


def linalg_multi_dot_out(tensors, *, out):
    logger.debug("GEMS_HYGON LINALG_MULTI_DOT_OUT")
    arrays, shape = _validate_and_prepare(tensors)
    if out.dtype != arrays[0].dtype or out.device != arrays[0].device:
        raise RuntimeError("multi_dot(): incompatible output dtype or device")
    if tuple(out.shape) != shape:
        out.resize_(shape)
    _chain(arrays, out.view(arrays[0].shape[0], arrays[-1].shape[1]))
    return out
