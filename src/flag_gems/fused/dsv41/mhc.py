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

# Portions adapted from SGLang (Copyright 2023-2026 SGLang Team),
# including the validated CherryLemon DeepSeek V4.1 H100 serving fork.
# See docs/dsv41-model-operators.md for source identities and numerical scope.
import functools
from typing import Tuple

import torch
import triton
import triton.language as tl

MHC_EXACT_ABI = 1


@triton.jit
def _hc_combine_kernel(
    x_ptr,
    pre_ptr,
    y_ptr,
    H,
    x_stride_m,
    pre_stride_m,
    pre_stride_k,
    y_stride_m,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs_h < H
    acc = tl.zeros([BLOCK_H], dtype=tl.float32)
    for k in tl.static_range(HC):
        pk = tl.load(pre_ptr + pid_m * pre_stride_m + k * pre_stride_k).to(tl.float32)
        xv = tl.load(
            x_ptr + pid_m * x_stride_m + k * H + offs_h, mask=mask, other=0.0
        ).to(tl.float32)
        acc += pk * xv
    tl.store(y_ptr + pid_m * y_stride_m + offs_h, acc, mask=mask)


@triton.jit
def _hc_mix_stats_partial_kernel(
    x_ptr,
    w_ptr,
    part_mix_ptr,
    part_sq_ptr,
    M,
    K,
    x_stride_m,
    w_stride_n,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    """Mixing dot products and row sum of squares over one K slice; the slicing
    and tiles are compile-time constants, so a row's fp32 operation sequence
    does not depend on the batch size."""
    pid_m = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    k_per_slice = K // NUM_SLICES
    k_start = pid_s * k_per_slice
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kb in range(0, k_per_slice, BLOCK_K):
        offs_k = k_start + kb + tl.arange(0, BLOCK_K)
        mask_k = offs_k < k_start + k_per_slice
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * w_stride_n + offs_k[:, None],
            mask=mask_n[None, :] & mask_k[:, None],
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(x_tile, w_tile, input_precision=DOT_PRECISION)
        sq += tl.sum(x_tile * x_tile, axis=1)
    tl.store(
        part_mix_ptr + (pid_s * M + offs_m[:, None]) * MIX + offs_n[None, :],
        acc,
        mask=mask_m[:, None] & mask_n[None, :],
    )
    tl.store(part_sq_ptr + pid_s * M + offs_m, sq, mask=mask_m)


@triton.jit
def _hc_mix_stats_reduce_kernel(
    part_mix_ptr,
    part_sq_ptr,
    mixes_ptr,
    M,
    inv_k,
    eps,
    MIX: tl.constexpr,
    MIX_PAD: tl.constexpr,
    NUM_SLICES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Sum the NUM_SLICES partials in slice order and apply the rms scaling."""
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, MIX_PAD)
    mask_m = offs_m < M
    mask_n = offs_n < MIX
    acc = tl.zeros([BLOCK_M, MIX_PAD], dtype=tl.float32)
    sq = tl.zeros([BLOCK_M], dtype=tl.float32)
    for s in tl.static_range(NUM_SLICES):
        acc += tl.load(
            part_mix_ptr + (s * M + offs_m[:, None]) * MIX + offs_n[None, :],
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        sq += tl.load(part_sq_ptr + s * M + offs_m, mask=mask_m, other=0.0)
    rsqrt = 1.0 / tl.sqrt(sq * inv_k + eps)
    tl.store(
        mixes_ptr + offs_m[:, None] * MIX + offs_n[None, :],
        acc * rsqrt[:, None],
        mask=mask_m[:, None] & mask_n[None, :],
    )


# K slicing, BLOCK_K and dot precision must stay independent of M;
# a row must produce the same bits alone and in any batch.
_HC_MIX_SLICE_CHOICES = (80, 64, 40, 32, 16, 8, 4, 2, 1)
_HC_MIX_BLOCK_M = 32
_HC_MIX_BLOCK_K = 64
_HC_MIX_NUM_WARPS = 4
_HC_MIX_DOT_PRECISION = "tf32x3"
# num_stages only reorders memory issue, not arithmetic; 2 is enough to cover the
# short k_per_slice loop (K=20480 gives 80 slices, i.e. 4 BLOCK_K tiles per CTA).
_HC_MIX_NUM_STAGES = 2

# BLOCK_M 8/16/32 preserve each row's K reduction order; thresholds were measured on GB300.
# Keep BLOCK_M below 64, where Triton lowers tf32x3 to plain TF32 and changes rounding.
_HC_MIX_BLOCK_M_SMALL = 8
_HC_MIX_BLOCK_M_MID = 16
_HC_MIX_MID_MAX_M = 2048


def _block_m_for(m: int) -> int:
    """Row-tile choices preserve each row's arithmetic and may depend on M."""
    if m <= _HC_MIX_BLOCK_M_SMALL:
        return _HC_MIX_BLOCK_M_SMALL
    if m <= _HC_MIX_MID_MAX_M:
        return _HC_MIX_BLOCK_M_MID
    return _HC_MIX_BLOCK_M


def _num_stages_for(m: int, k: int) -> int:
    # GB300 verify batches benefit from a smaller shared-memory footprint.
    # This changes memory scheduling only; K tiles and reduction order stay fixed.
    if torch.cuda.get_device_capability() == (10, 0) and k == 20480 and 64 <= m <= 384:
        return 1
    return _HC_MIX_NUM_STAGES


def _num_slices_for(k: int) -> int:
    """Slice count depends only on K, never on batch size M."""
    blocks = k // _HC_MIX_BLOCK_K
    assert k % _HC_MIX_BLOCK_K == 0, k
    for n in _HC_MIX_SLICE_CHOICES:
        if blocks % n == 0:
            return n
    return 1


def _hc_mix_geometry_for(m: int, k: int) -> Tuple[int, int]:
    # Exact geometry changes measured on H100 serving-derived rows only.
    if torch.cuda.get_device_capability() == (9, 0) and k == 20480:
        if m in (10, 12, 18, 20, 24, 30):
            return 8, 2
        if m in (50, 60):
            return 16, 1
        if m in (70, 72):
            return 8, 1
    return _block_m_for(m), _num_stages_for(m, k)


def hc_mix_stats(
    x_flat: torch.Tensor, hc_fn: torch.Tensor, eps: float, *, tuned=True
) -> torch.Tensor:
    """Batch-invariant F.linear(x_flat.float(), hc_fn) * rsqrt(mean(x_flat^2) + eps).

    x_flat is [M, K] in any float dtype; hc_fn is [MIX, K] fp32; returns [M, MIX] fp32.
    K slicing and reduction order are independent of M, so each row is bitwise
    identical whether computed alone or in a batch.
    """
    assert x_flat.dim() == 2 and hc_fn.dim() == 2
    assert x_flat.stride(1) == 1 and hc_fn.stride(1) == 1
    assert hc_fn.dtype == torch.float32
    m, k = x_flat.shape
    mix = hc_fn.shape[0]
    assert hc_fn.shape[1] == k
    num_slices = _num_slices_for(k)
    mix_pad = max(16, triton.next_power_of_2(mix))
    part_mix = torch.empty(
        (num_slices, m, mix), dtype=torch.float32, device=x_flat.device
    )
    part_sq = torch.empty((num_slices, m), dtype=torch.float32, device=x_flat.device)
    mixes = torch.empty((m, mix), dtype=torch.float32, device=x_flat.device)
    if m == 0:
        return mixes
    block_m, num_stages = (
        _hc_mix_geometry_for(m, k) if tuned else (_block_m_for(m), _HC_MIX_NUM_STAGES)
    )
    grid_m = triton.cdiv(m, block_m)
    _hc_mix_stats_partial_kernel[(grid_m, num_slices)](
        x_flat,
        hc_fn,
        part_mix,
        part_sq,
        m,
        k,
        x_flat.stride(0),
        hc_fn.stride(0),
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        BLOCK_K=_HC_MIX_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        num_warps=_HC_MIX_NUM_WARPS,
        num_stages=num_stages,
    )
    _hc_mix_stats_reduce_kernel[(grid_m,)](
        part_mix,
        part_sq,
        mixes,
        m,
        1.0 / k,
        eps,
        MIX=mix,
        MIX_PAD=mix_pad,
        NUM_SLICES=num_slices,
        BLOCK_M=block_m,
        num_warps=4,
    )
    return mixes


def hc_combine(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    """Fused y[m, h] = sum_k pre[m, k] * x_flat[m, k*H + h]."""
    m = x_flat.shape[0]
    h = x_flat.shape[1] // hc
    y = torch.empty((m, h), dtype=out_dtype, device=x_flat.device)
    block_h = 1024
    _hc_combine_kernel[(m, triton.cdiv(h, block_h))](
        x_flat,
        pre,
        y,
        h,
        x_flat.stride(0),
        pre.stride(0),
        pre.stride(1),
        y.stride(0),
        HC=hc,
        BLOCK_H=block_h,
    )
    return y


_HC_EXACT_REDUCE_CUDA = r"""
__device__ __forceinline__ float hc_mix_exact_rms(float sq, float inv_k, float eps) {
  float val, root, inv;
  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(val) : "f"(sq), "f"(inv_k), "f"(eps));
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(root) : "f"(val));
  asm("div.full.f32 %0, %1, %2;" : "=f"(inv) : "f"(1.0f), "f"(root));
  return inv;
}
__device__ __forceinline__ float hc_mix_exact_mul(float a, float b) {
  float out;
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
  return out;
}
"""


@functools.lru_cache(maxsize=128)
def _exact_kernel(hc, iters, eps, slices, debug_mix):
    # TileLang is optional for package import and required only by this entry.
    import tilelang
    import tilelang.language as T

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def _hc_mix_reduce_sinkhorn_exact_tilelang_kernel(
        hc: int, iters: int, eps: float, slices: int, debug_mix: bool
    ):
        n = T.symbolic("n")
        mix_hc = (2 + hc) * hc
        pdl = True

        @T.prim_func
        def main(
            partial: T.Tensor[(slices, n, mix_hc), T.float32],
            sq_partial: T.Tensor[(slices, n), T.float32],
            scale: T.Tensor[(3,), T.float32],
            base: T.Tensor[(mix_hc,), T.float32],
            pre: T.Tensor[(n, hc), T.float32],
            post: T.Tensor[(n, hc), T.float32],
            comb: T.Tensor[(n, hc, hc), T.float32],
            mixes: T.Tensor[(n, mix_hc), T.float32],
            inv_k: T.float32,
            rms_eps: T.float32,
        ):
            with T.Kernel(n, threads=64) as i:
                T.import_source(_HC_EXACT_REDUCE_CUDA)
                if pdl:
                    T.pdl_sync()
                mix_acc = T.alloc_fragment((mix_hc,), T.float32)
                sq_acc = T.alloc_local((1,), T.float32)
                rsqrt = T.alloc_local((1,), T.float32)
                mixes_shared = T.alloc_shared((mix_hc,), T.float32)
                comb_frag = T.alloc_fragment((hc, hc), T.float32)
                T.clear(mix_acc)
                sq_acc[0] = 0.0
                for s in T.serial(slices):
                    for j in T.Parallel(mix_hc):
                        mix_acc[j] = mix_acc[j] + partial[s, i, j]
                    sq_acc[0] = sq_acc[0] + sq_partial[s, i]
                rsqrt[0] = T.call_pure_extern(
                    T.float32, "hc_mix_exact_rms", sq_acc[0], inv_k, rms_eps
                )
                for j in T.Parallel(mix_hc):
                    mixes_shared[j] = T.call_pure_extern(
                        T.float32, "hc_mix_exact_mul", mix_acc[j], rsqrt[0]
                    )
                if debug_mix:
                    T.copy(mixes_shared, mixes[i, :])

                # Keep the production TileLang expressions and reduction layouts.
                for j in T.Parallel(hc):
                    pre[i, j] = T.sigmoid(mixes_shared[j] * scale[0] + base[j]) + eps
                for j in T.Parallel(hc):
                    post[i, j] = 2 * T.sigmoid(
                        mixes_shared[j + hc] * scale[1] + base[j + hc]
                    )
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = (
                        mixes_shared[j * hc + k + hc * 2] * scale[2]
                        + base[j * hc + k + hc * 2]
                    )
                row_sum = T.alloc_fragment((hc,), T.float32)
                col_sum = T.alloc_fragment((hc,), T.float32)
                row_max = T.alloc_fragment((hc,), T.float32)
                T.reduce_max(comb_frag, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
                for _ in T.serial(iters - 1):
                    T.reduce_sum(comb_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                    T.reduce_sum(comb_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)
                T.copy(comb_frag, comb[i, :, :])
                if pdl:
                    T.pdl_trigger()

        return main

    return _hc_mix_reduce_sinkhorn_exact_tilelang_kernel(
        hc, iters, eps, slices, debug_mix
    )


def hc_mix_stats_sinkhorn_exact(
    x, fn, scale, base, hc, iters, rms_eps, hc_eps, *, debug_mix=False
):
    assert x.dtype == torch.bfloat16 and fn.dtype == torch.float32
    assert x.ndim == fn.ndim == 2 and x.stride(1) == fn.stride(1) == 1
    m, k = x.shape
    assert k == 20480 and hc == 4 and fn.shape == (24, k)
    assert (
        scale.dtype == base.dtype == torch.float32
        and scale.is_contiguous()
        and base.is_contiguous()
    )
    assert m in (36, 40, 42, 48, 50, 60, 80, 84, 96)
    slices = _num_slices_for(k)
    pm = torch.empty((slices, m, 24), dtype=torch.float32, device=x.device)
    ps = torch.empty((slices, m), dtype=torch.float32, device=x.device)
    mixes = torch.empty((m, 24), dtype=torch.float32, device=x.device)
    pre = torch.empty((m, hc), dtype=torch.float32, device=x.device)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc, hc), dtype=torch.float32, device=x.device)
    bm, stages = _hc_mix_geometry_for(m, k)
    _hc_mix_stats_partial_kernel[(triton.cdiv(m, bm), slices)](
        x,
        fn,
        pm,
        ps,
        m,
        k,
        x.stride(0),
        fn.stride(0),
        MIX=24,
        MIX_PAD=32,
        NUM_SLICES=slices,
        BLOCK_M=bm,
        BLOCK_K=_HC_MIX_BLOCK_K,
        DOT_PRECISION=_HC_MIX_DOT_PRECISION,
        num_warps=_HC_MIX_NUM_WARPS,
        num_stages=stages,
    )
    kernel = _exact_kernel(hc, iters, hc_eps, slices, debug_mix)
    kernel(pm, ps, scale, base, pre, post, comb, mixes, 1.0 / k, rms_eps)
    return (mixes, pre, post, comb) if debug_mix else (pre, post, comb)


@functools.lru_cache(maxsize=128)
def _split_kernel(hc, sinkhorn_iters, eps):
    import tilelang
    import tilelang.language as T

    @tilelang.jit(
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        }
    )
    def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
        n = T.symbolic("n")
        mix_hc = (2 + hc) * hc
        threads = 64

        ENABLE_PDL = True

        @T.prim_func
        def hc_split_sinkhorn_kernel_(
            mixes: T.Tensor[(n, mix_hc), T.float32],
            hc_scale: T.Tensor[(3,), T.float32],
            hc_base: T.Tensor[(mix_hc,), T.float32],
            pre: T.Tensor[(n, hc), T.float32],
            post: T.Tensor[(n, hc), T.float32],
            comb: T.Tensor[(n, hc, hc), T.float32],
        ):
            with T.Kernel(n, threads=threads) as i:
                if ENABLE_PDL:
                    T.pdl_sync()

                mixes_shared = T.alloc_shared(mix_hc, T.float32)
                comb_frag = T.alloc_fragment((hc, hc), T.float32)
                T.copy(mixes[i, :], mixes_shared)

                for j in T.Parallel(hc):
                    pre[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
                    )
                for j in T.Parallel(hc):
                    post[i, j] = 2 * T.sigmoid(
                        mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                    )
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = (
                        mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                        + hc_base[j * hc + k + hc * 2]
                    )

                row_sum = T.alloc_fragment(hc, T.float32)
                col_sum = T.alloc_fragment(hc, T.float32)

                row_max = T.alloc_fragment(hc, T.float32)
                T.reduce_max(comb_frag, row_max, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

                for _ in T.serial(sinkhorn_iters - 1):
                    T.reduce_sum(comb_frag, row_sum, dim=1)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                    T.reduce_sum(comb_frag, col_sum, dim=0)
                    for j, k in T.Parallel(hc, hc):
                        comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

                T.copy(comb_frag, comb[i, :, :])
                if ENABLE_PDL:
                    T.pdl_trigger()

        return hc_split_sinkhorn_kernel_

    return hc_split_sinkhorn_kernel(hc, sinkhorn_iters, eps)


def hc_split_sinkhorn(mixes, scale, base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    if mixes.ndim != 2 or mixes.shape[1] != (2 + hc_mult) * hc_mult:
        raise ValueError("mixes must have shape [M,(2+HC)*HC]")
    if (
        mixes.dtype != torch.float32
        or scale.dtype != torch.float32
        or base.dtype != torch.float32
    ):
        raise TypeError("mixes, scale and base must be FP32")
    if (
        not mixes.is_contiguous()
        or not scale.is_contiguous()
        or not base.is_contiguous()
    ):
        raise ValueError("mixes, scale and base must be contiguous")
    m = mixes.shape[0]
    pre = torch.empty((m, hc_mult), dtype=torch.float32, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty((m, hc_mult, hc_mult), dtype=torch.float32, device=mixes.device)
    if m:
        _split_kernel(hc_mult, sinkhorn_iters, eps)(mixes, scale, base, pre, post, comb)
    return pre, post, comb


__all__ = [
    "hc_mix_stats",
    "hc_mix_stats_sinkhorn_exact",
    "hc_combine",
    "hc_split_sinkhorn",
]
