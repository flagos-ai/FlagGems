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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

_CATALOG_MNK = {
    (1, 64, 4096),
    (2, 16, 4096),
    (32, 512, 4096),
    (64, 4, 16384),
    (16384, 1, 4096),
    (512, 256, 2048),
    (128, 128, 64),
}
_CATALOG_BATCHES = {2, 4, 8}


def _supported_b_layout(tensor):
    if tensor.is_contiguous():
        return True
    # View produced by contiguous [B, N, K].transpose(1, 2).
    batch, k, n = tensor.shape
    return tensor.stride() == (n * k, 1, k)


def is_eligible(bias, a, b, alpha, beta, out=None):
    if (
        float(alpha) != 1.0
        or float(beta) != 1.0
        or a.ndim != 3
        or b.ndim != 3
        or bias.ndim != 3
        or a.shape[0] not in _CATALOG_BATCHES
        or (a.shape[1], b.shape[2], a.shape[2]) not in _CATALOG_MNK
        or a.shape[0] != b.shape[0]
        or a.shape[2] != b.shape[1]
        or tuple(bias.shape) != (a.shape[0], a.shape[1], b.shape[2])
        or a.dtype not in (torch.float16, torch.bfloat16)
        or bias.dtype != a.dtype
        or b.dtype != a.dtype
        or (a.shape[1] == 2 and b.is_contiguous())
        or not bias.is_contiguous()
        or not a.is_contiguous()
        or not _supported_b_layout(b)
        or bias.device != a.device
        or b.device != a.device
        or a.device.type != "musa"
    ):
        return False
    return out is None or (
        tuple(out.shape) == tuple(bias.shape)
        and out.dtype == a.dtype
        and out.device == a.device
        and out.is_contiguous()
        and all(not torch._C._overlaps(out, tensor) for tensor in (bias, a, b))
    )


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_m12_nt_striped_kernel(
    bias,
    a,
    b,
    out,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fuse one/two GEMVs over the physical contiguous ``B[B,N,K]`` rows."""
    pid = tl.program_id(0)
    pid_b = pid // N
    pid_n = pid % N
    batch_a = a + pid_b * M * K
    weight = b + pid_b * N * K + pid_n * K

    accumulator0 = tl.zeros((), dtype=tl.float32)
    accumulator1 = tl.zeros((), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offsets_k = k_start + tl.arange(0, BLOCK_K)
        weight_values = tl.load(weight + offsets_k).to(tl.float32)
        vector0 = tl.load(batch_a + offsets_k).to(tl.float32)
        accumulator0 += tl.sum(weight_values * vector0, axis=0)
        if M == 2:
            vector1 = tl.load(batch_a + K + offsets_k).to(tl.float32)
            accumulator1 += tl.sum(weight_values * vector1, axis=0)

    bias0 = tl.load(bias + (pid_b * M) * N + pid_n).to(tl.float32)
    tl.store(out + (pid_b * M) * N + pid_n, alpha * accumulator0 + beta * bias0)
    if M == 2:
        bias1 = tl.load(bias + (pid_b * M + 1) * N + pid_n).to(tl.float32)
        tl.store(
            out + (pid_b * M + 1) * N + pid_n,
            alpha * accumulator1 + beta * bias1,
        )


@libentry()
@triton.jit(do_not_specialize=["alpha", "beta"])
def _baddbmm_strided_b_kernel(
    bias,
    a,
    b,
    out,
    alpha,
    beta,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_ib: tl.constexpr,
    stride_im: tl.constexpr,
    stride_in: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    tiles_per_batch = num_m * num_n
    pid_b = pid // tiles_per_batch
    tile = pid % tiles_per_batch
    pid_m = tile // num_n
    pid_n = tile % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = (
        a
        + pid_b * stride_ab
        + offs_m[:, None] * stride_am
        + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b
        + pid_b * stride_bb
        + offs_k[:, None] * stride_bk
        + offs_n[None, :] * stride_bn
    )
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = k_start * BLOCK_K + offs_k < K
        a_tile = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        accumulator += tl.dot(a_tile, b_tile)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias_ptrs = (
        bias
        + pid_b * stride_ib
        + offs_m[:, None] * stride_im
        + offs_n[None, :] * stride_in
    )
    bias_tile = tl.load(bias_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    out_ptrs = out + (pid_b * M + offs_m[:, None]) * N + offs_n[None, :]
    tl.store(
        out_ptrs,
        alpha * accumulator + beta * bias_tile,
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _config(m, n, k):
    if m <= 2:
        return (16, 64 if n >= 64 else 16, 64, 4)
    if n <= 4:
        return (64, 16, 128, 4)
    if m <= 32:
        return (32, 128, 64, 8)
    if m == 65:
        return (64, 64, 32, 4)
    if m == 128 and k == 64:
        return (32, 32, 32, 4)
    if m == 512:
        return (64, 128, 64, 8)
    return (128, 16, 64, 4)


def launch(bias, a, b, out=None, *, alpha=1.0, beta=1.0):
    batch, m, k = a.shape
    n = b.shape[2]
    if out is None:
        out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    if m <= 2 and not b.is_contiguous():
        block_k, num_warps = (2048, 4) if m == 1 else (1024, 2)
        with torch_device_fn.device(a.device):
            _baddbmm_m12_nt_striped_kernel[(batch * n,)](
                bias,
                a,
                b,
                out,
                alpha,
                beta,
                M=m,
                N=n,
                K=k,
                BLOCK_K=block_k,
                num_warps=num_warps,
                num_stages=1,
            )
        return out
    block_m, block_n, block_k, num_warps = _config(m, n, k)
    grid = (batch * triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
    with torch_device_fn.device(a.device):
        _baddbmm_strided_b_kernel[grid](
            bias,
            a,
            b,
            out,
            alpha,
            beta,
            m,
            n,
            k,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=1,
        )
    return out
