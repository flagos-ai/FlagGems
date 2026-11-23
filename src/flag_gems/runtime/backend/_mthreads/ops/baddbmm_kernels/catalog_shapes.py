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

from ..gemm_common import small_m_nt_gemv_kernel


def _supported_b_layout(tensor):
    if tensor.is_contiguous():
        return True
    # View produced by contiguous [B, N, K].transpose(1, 2).
    batch, k, n = tensor.shape
    return tensor.stride() == (n * k, 1, k)


def _direct_config(batch, m, n, k, transposed_b):
    """Return the direct algorithm and tile for a bounded shape family."""

    if not (2 <= batch <= 8 and batch & (batch - 1) == 0) or m <= 0 or n <= 0 or k <= 0:
        return None

    # One/two-row NT operands are GEMVs over physical [B, N, K] rows.  Keep
    # the M=1 contiguous case on the direct tiled implementation.
    if m <= 2 and n <= 64 and 1024 <= k <= 16384:
        if transposed_b:
            block_k, num_warps = (2048, 4) if m == 1 else (1024, 2)
            return ("nt_gemv", 1, 1, block_k, num_warps)
        if m == 1:
            return ("direct_tiled", 16, 64, 64, 4)
        return None

    # These regimes correspond to distinct tiling strategies, rather than to
    # individual benchmark tuples.  Bounds prevent this specialized path from
    # intercepting unmeasured large problems.
    if 3 <= m <= 16384 and n <= 4 and 1024 <= k <= 16384:
        return ("direct_tiled", 64, 16, 128, 4)
    if 3 <= m <= 32 and 128 <= n <= 512 and 1024 <= k <= 4096:
        return ("direct_tiled", 32, 128, 64, 8)
    if 64 <= m <= 256 and 64 <= n <= 256 and k <= 64:
        return ("direct_tiled", 32, 32, 32, 4)
    if 64 <= m <= 512 and 128 <= n <= 512 and 1024 <= k <= 4096:
        return ("direct_tiled", 64, 128, 64, 8)
    if 1024 <= m <= 16384 and n <= 16 and 1024 <= k <= 4096:
        return ("direct_tiled", 128, 16, 64, 4)
    return None


def is_eligible(bias, a, b, alpha, beta, out=None):
    transposed_b = b.ndim == 3 and not b.is_contiguous() and _supported_b_layout(b)
    if (
        float(alpha) != 1.0
        or float(beta) != 1.0
        or a.ndim != 3
        or b.ndim != 3
        or bias.ndim != 3
        or a.shape[0] != b.shape[0]
        or a.shape[2] != b.shape[1]
        or _direct_config(a.shape[0], a.shape[1], b.shape[2], a.shape[2], transposed_b)
        is None
        or tuple(bias.shape) != (a.shape[0], a.shape[1], b.shape[2])
        or a.dtype not in (torch.float16, torch.bfloat16)
        or bias.dtype != a.dtype
        or b.dtype != a.dtype
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


def launch(bias, a, b, out=None, *, alpha=1.0, beta=1.0):
    batch, m, k = a.shape
    n = b.shape[2]
    if out is None:
        out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
    config = _direct_config(batch, m, n, k, not b.is_contiguous())
    if config is None:
        raise ValueError("unsupported direct baddbmm shape")
    algorithm, block_m, block_n, block_k, num_warps = config
    if algorithm == "nt_gemv":
        with torch_device_fn.device(a.device):
            small_m_nt_gemv_kernel[(n, batch)](
                a,
                b,
                bias,
                out,
                alpha,
                beta,
                a.stride(0),
                a.stride(1),
                a.stride(2),
                b.stride(0),
                b.stride(1),
                b.stride(2),
                bias.stride(0),
                bias.stride(1),
                bias.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                M=m,
                N=n,
                K=k,
                BLOCK_N=1,
                BLOCK_K=block_k,
                FUSE_BIAS=True,
                BETA_ZERO=float(beta) == 0.0,
                num_warps=num_warps,
                num_stages=1,
            )
        return out
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
