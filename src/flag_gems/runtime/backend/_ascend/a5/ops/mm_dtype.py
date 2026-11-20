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

_BLOCK_N = 8
_BLOCK_K = 1024
_NUM_STAGES = 1
_NUM_WARPS = 1
_SKEW = False


# --------------------------------------------------------------------------- #
# M == 1: fully specialized (compile-time extents and strides)
# --------------------------------------------------------------------------- #
@triton.jit
def _gemv_un(
    X,  # pointer to self row, unit stride along k
    W,  # pointer to mat2, rows SW_N apart, unit stride along k
    OUT,  # pointer to out0 row, unit stride
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ITERS: tl.constexpr,
    SW_N: tl.constexpr,
    SKEW: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    wrow = W + offs_n[:, None] * SW_N

    if SKEW and ITERS == 3:
        # LOAD SKEW: all three GM->UB weight copies and all three activation
        # copies are issued before any vector work, so copy(r+1) no longer waits
        # behind the no_inline vector call of round r.  The three private UB
        # tiles already exist and the fp32 fold order is unchanged, so only the
        # placement of the copies moves.
        w0 = tl.load(wrow + offs_k[None, :])
        x0 = tl.load(X + offs_k)
        w1 = tl.load(wrow + (BLOCK_K + offs_k)[None, :])
        x1 = tl.load(X + BLOCK_K + offs_k)
        w2 = tl.load(wrow + (2 * BLOCK_K + offs_k)[None, :])
        x2 = tl.load(X + 2 * BLOCK_K + offs_k)
        acc = tl.sum(w0.to(tl.float32) * x0[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w1.to(tl.float32) * x1[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w2.to(tl.float32) * x2[None, :].to(tl.float32), axis=1)
    elif SKEW and ITERS == 4:
        # Same load-skew form, four chunks of 3 KiB per row.  Only the per-row
        # contiguous run narrows (the one measured-positive axis at this tile
        # width); issue order, fold order and the fp32 accumulation are the
        # round-1 arithmetic unchanged.
        w0 = tl.load(wrow + offs_k[None, :])
        x0 = tl.load(X + offs_k)
        w1 = tl.load(wrow + (BLOCK_K + offs_k)[None, :])
        x1 = tl.load(X + BLOCK_K + offs_k)
        w2 = tl.load(wrow + (2 * BLOCK_K + offs_k)[None, :])
        x2 = tl.load(X + 2 * BLOCK_K + offs_k)
        w3 = tl.load(wrow + (3 * BLOCK_K + offs_k)[None, :])
        x3 = tl.load(X + 3 * BLOCK_K + offs_k)
        acc = tl.sum(w0.to(tl.float32) * x0[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w1.to(tl.float32) * x1[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w2.to(tl.float32) * x2[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w3.to(tl.float32) * x3[None, :].to(tl.float32), axis=1)
    elif SKEW and ITERS == 6:
        # Same load-skew form, six chunks of 2 KiB per row.  Only the per-row
        # contiguous run narrows further; issue order, fold order and the fp32
        # accumulation are unchanged.
        w0 = tl.load(wrow + offs_k[None, :])
        x0 = tl.load(X + offs_k)
        w1 = tl.load(wrow + (BLOCK_K + offs_k)[None, :])
        x1 = tl.load(X + BLOCK_K + offs_k)
        w2 = tl.load(wrow + (2 * BLOCK_K + offs_k)[None, :])
        x2 = tl.load(X + 2 * BLOCK_K + offs_k)
        w3 = tl.load(wrow + (3 * BLOCK_K + offs_k)[None, :])
        x3 = tl.load(X + 3 * BLOCK_K + offs_k)
        w4 = tl.load(wrow + (4 * BLOCK_K + offs_k)[None, :])
        x4 = tl.load(X + 4 * BLOCK_K + offs_k)
        w5 = tl.load(wrow + (5 * BLOCK_K + offs_k)[None, :])
        x5 = tl.load(X + 5 * BLOCK_K + offs_k)
        acc = tl.sum(w0.to(tl.float32) * x0[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w1.to(tl.float32) * x1[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w2.to(tl.float32) * x2[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w3.to(tl.float32) * x3[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w4.to(tl.float32) * x4[None, :].to(tl.float32), axis=1)
        acc += tl.sum(w5.to(tl.float32) * x5[None, :].to(tl.float32), axis=1)
    else:
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for it in tl.static_range(0, ITERS):
            kk = it * BLOCK_K + offs_k
            w = tl.load(wrow + kk[None, :])
            x = tl.load(X + kk)
            acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(OUT + offs_n, acc)


# --------------------------------------------------------------------------- #
# M == 1: streaming GEMV over the weight rows (divisible fast path)
# --------------------------------------------------------------------------- #
@triton.jit
def _gemv_even(
    X,  # pointer to self row (self[0, :])
    W,  # pointer to mat2 (addressed as W[k, n])
    OUT,  # pointer to out0 row  (out0[0, :])
    N,
    K,
    sx_k,  # self.stride(1)
    sw_k,  # mat2.stride(0)
    sw_n,  # mat2.stride(1)
    so_n,  # out0.stride(1)
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ITERS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    wrow = W + offs_n[:, None] * sw_n

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for it in tl.static_range(0, ITERS):
        kk = it * BLOCK_K + offs_k
        w = tl.load(wrow + kk[None, :] * sw_k)
        x = tl.load(X + kk * sx_k)
        acc += tl.sum(w.to(tl.float32) * x[None, :].to(tl.float32), axis=1)
    tl.store(OUT + offs_n * so_n, acc)


# --------------------------------------------------------------------------- #
# M == 1: masked general path (any N, K, strides)
# --------------------------------------------------------------------------- #
@triton.jit
def _gemv_masked(
    X,
    W,
    OUT,
    N,
    K,
    sx_k,
    sw_k,
    sw_n,
    so_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + offs_k
        mask_k = kk < K
        w = tl.load(
            W + offs_n[:, None] * sw_n + kk[None, :] * sw_k,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        x = tl.load(X + kk * sx_k, mask=mask_k, other=0.0)
        acc += w.to(tl.float32) * x[None, :].to(tl.float32)
    res = tl.sum(acc, axis=1)
    tl.store(OUT + offs_n * so_n, res, mask=mask_n)


# --------------------------------------------------------------------------- #
# Generic fallback for M > 1
# --------------------------------------------------------------------------- #
@triton.jit
def _mm_kernel(
    X,
    W,
    OUT,
    M,
    N,
    K,
    sx_m,
    sx_k,
    sw_k,
    sw_n,
    so_m,
    so_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + offs_k
        mask_k = kk < K
        a = tl.load(
            X + offs_m[:, None] * sx_m + kk[None, :] * sx_k,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        b = tl.load(
            W + kk[:, None] * sw_k + offs_n[None, :] * sw_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc = tl.dot(a, b, acc)

    tl.store(
        OUT + offs_m[:, None] * so_m + offs_n[None, :] * so_n,
        acc.to(OUT.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


def _out_dtype(self_dtype, mat2_dtype, out_dtype):
    if isinstance(out_dtype, torch.dtype):
        return out_dtype
    return torch.promote_types(self_dtype, mat2_dtype)


def _launch(self, mat2, out_dtype):
    # HOST PATH: the device body is byte-identical to the measured best round;
    # only the per-call Python work around the launch changes here.  Measured
    # facts: the profiled kernel is 3.355 us of a 4.6075 us headline, so ~1.25
    # us per call lies outside the kernel, which is consistent with the timed
    # path being launch-throughput bound.  Every avoidable Python operation is
    # therefore removed from the fast path: one shape read per tensor, one
    # stride() tuple per tensor instead of per-axis calls, output allocation
    # through x.new_empty (no separate device/dtype lookups), inline dtype
    # resolution, and constexpr tile parameters passed positionally so Triton
    # does not build a keyword dict, while the correctness-preserving guards and
    # all fallback paths are unchanged.
    x = self
    w = mat2
    if x.dim() != 2:
        x = x.unsqueeze(0) if x.dim() == 1 else x
    if w.dim() != 2:
        w = w.unsqueeze(1) if w.dim() == 1 else w

    xs = x.shape
    M = xs[0]
    K = xs[1]
    N = w.shape[1]

    dt = (
        out_dtype
        if isinstance(out_dtype, torch.dtype)
        else torch.promote_types(x.dtype, w.dtype)
    )
    out = x.new_empty((M, N), dtype=dt)

    if M == 1:
        xt = x.stride()
        wt = w.stride()
        sx_k = xt[1]
        sw_k = wt[0]
        sw_n = wt[1]
        so_n = out.stride()[1]
        if (
            N % _BLOCK_N == 0
            and K % _BLOCK_K == 0
            and sx_k == 1
            and sw_k == 1
            and sw_n == K
            and so_n == 1
        ):
            _gemv_un[(N // _BLOCK_N,)](
                x,
                w,
                out,
                _BLOCK_N,
                _BLOCK_K,
                K // _BLOCK_K,
                sw_n,
                _SKEW,
                num_stages=_NUM_STAGES,
                num_warps=_NUM_WARPS,
            )
        elif N % _BLOCK_N == 0 and K % _BLOCK_K == 0:
            _gemv_even[(N // _BLOCK_N,)](
                x,
                w,
                out,
                N,
                K,
                sx_k,
                sw_k,
                sw_n,
                so_n,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=_BLOCK_K,
                ITERS=K // _BLOCK_K,
                num_stages=_NUM_STAGES,
            )
        else:
            _gemv_masked[(triton.cdiv(N, _BLOCK_N),)](
                x,
                w,
                out,
                N,
                K,
                sx_k,
                sw_k,
                sw_n,
                so_n,
                BLOCK_N=_BLOCK_N,
                BLOCK_K=512,
            )
    else:
        BM = 32
        BN = 64
        BK = 64
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        _mm_kernel[grid](
            x,
            w,
            out,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            out.stride(0),
            out.stride(1),
            BM=BM,
            BN=BN,
            BK=BK,
        )
    return out


def mm_dtype(self, mat2, out_dtype):
    """Matrix multiplication with an explicit output dtype on Atlas A5."""
    if self.ndim != 2 or mat2.ndim != 2 or self.shape[1] != mat2.shape[0]:
        raise ValueError("expected compatible two-dimensional matrices")
    if (
        self.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or self.dtype != mat2.dtype
        or out_dtype not in (self.dtype, torch.float32)
    ):
        raise TypeError("expected matching floating inputs and matching or FP32 output")
    if not self.numel() or not mat2.numel():
        return torch.zeros(
            (self.shape[0], mat2.shape[1]), device=self.device, dtype=out_dtype
        )
    if self.device.type != "npu" or mat2.device != self.device:
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(self.device):
        return _launch(self, mat2, out_dtype)
