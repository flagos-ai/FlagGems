import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.scaled_grouped_mm import (
    _check_dims,
    _default_out_dtype,
    _normalize_bias,
    _normalize_scale,
    _resolve_shapes,
    _scaled_grouped_mm_fallback,
    _supports_triton_dot,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _scaled_grouped_mm_kernel(
    mat_a,
    mat_b,
    scale_a,
    scale_b,
    offs,
    bias,
    out,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_ag: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_cg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_sag: tl.constexpr,
    stride_sbg: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    MAX_N_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    group_idx = tl.program_id(0).to(tl.int64)
    tile_idx = tl.program_id(1).to(tl.int64)
    pid_m = tile_idx // MAX_N_TILES
    pid_n = tile_idx % MAX_N_TILES

    zero = tl.full((), 0, dtype=tl.int64)
    m_start = zero
    n_start = zero
    k_start = zero
    m_size = tl.full((), M, dtype=tl.int64)
    n_size = tl.full((), N, dtype=tl.int64)
    k_size = tl.full((), K, dtype=tl.int64)

    if A_IS_2D or B_IS_2D:
        offset_start = tl.load(offs + group_idx - 1, mask=group_idx > 0, other=0).to(
            tl.int64
        )
        offset_end = tl.load(offs + group_idx).to(tl.int64)
        group_size = offset_end - offset_start
        if A_IS_2D and not B_IS_2D:
            m_start = offset_start
            m_size = group_size
        elif not A_IS_2D and B_IS_2D:
            n_start = offset_start
            n_size = group_size
        else:
            k_start = offset_start
            k_size = group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    output_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_offset in range(0, k_size, BLOCK_K):
        group_offs_k = k_offset + offs_k
        a_ptrs = (
            mat_a
            + (0 if A_IS_2D else group_idx * stride_ag)
            + (m_start + offs_m[:, None]) * stride_am
            + (k_start + group_offs_k[None, :]) * stride_ak
        )
        b_ptrs = (
            mat_b
            + (0 if B_IS_2D else group_idx * stride_bg)
            + (k_start + group_offs_k[:, None]) * stride_bk
            + (n_start + offs_n[None, :]) * stride_bn
        )
        a_mask = (offs_m[:, None] < m_size) & (group_offs_k[None, :] < k_size)
        b_mask = (group_offs_k[:, None] < k_size) & (offs_n[None, :] < n_size)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    if A_IS_2D and not B_IS_2D:
        scale_a_vals = tl.load(
            scale_a + m_start + offs_m[:, None],
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )
    else:
        scale_a_vals = tl.load(
            scale_a + group_idx * stride_sag + offs_m[:, None],
            mask=offs_m[:, None] < m_size,
            other=0.0,
        )

    if not A_IS_2D and B_IS_2D:
        scale_b_vals = tl.load(
            scale_b + n_start + offs_n[None, :],
            mask=offs_n[None, :] < n_size,
            other=0.0,
        )
    else:
        scale_b_vals = tl.load(
            scale_b + group_idx * stride_sbg + offs_n[None, :],
            mask=offs_n[None, :] < n_size,
            other=0.0,
        )

    result = acc * scale_a_vals * scale_b_vals
    if BIAS_MODE == 1:
        bias_start = n_start if (not A_IS_2D and B_IS_2D) else zero
        bias_vals = tl.load(
            bias + bias_start + offs_n[None, :],
            mask=offs_n[None, :] < n_size,
            other=0.0,
        )
        result += bias_vals
    elif BIAS_MODE == 2:
        bias_vals = tl.load(
            bias + group_idx * N + offs_n[None, :],
            mask=offs_n[None, :] < n_size,
            other=0.0,
        )
        result += bias_vals

    if A_IS_2D and not B_IS_2D:
        out_ptrs = (
            out + (m_start + offs_m[:, None]) * stride_cm + offs_n[None, :] * stride_cn
        )
    elif not A_IS_2D and B_IS_2D:
        out_ptrs = (
            out + offs_m[:, None] * stride_cm + (n_start + offs_n[None, :]) * stride_cn
        )
    else:
        out_ptrs = (
            out
            + group_idx * stride_cg
            + offs_m[:, None] * stride_cm
            + offs_n[None, :] * stride_cn
        )
    tl.store(out_ptrs, result, mask=output_mask)


def scaled_grouped_mm(
    self,
    mat2,
    scale_a,
    scale_b,
    offs=None,
    bias=None,
    scale_result=None,
    out_dtype=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_MTHREADS SCALED_GROUPED_MM")
    if scale_result is not None:
        raise RuntimeError("scale_result is not supported for scaled_grouped_mm")

    _ = use_fast_accum
    _check_dims(self, mat2)
    (
        a_is_2d,
        b_is_2d,
        num_groups,
        M,
        N,
        K,
        out_shape,
        offs,
    ) = _resolve_shapes(self, mat2, offs)

    output_dtype = out_dtype or _default_out_dtype(self.dtype)
    scale_multiplier = num_groups if a_is_2d and b_is_2d else 1
    scale_a = _normalize_scale(
        scale_a,
        self,
        dim=0,
        num_groups=num_groups,
        scale_multiplier=scale_multiplier,
        name="scale_a",
    )
    scale_b = _normalize_scale(
        scale_b,
        mat2,
        dim=1,
        num_groups=num_groups,
        scale_multiplier=scale_multiplier,
        name="scale_b",
    )
    bias, bias_mode = _normalize_bias(
        bias, a_is_2d=a_is_2d, b_is_2d=b_is_2d, num_groups=num_groups, N=N
    )

    if not _supports_triton_dot(self.dtype):
        return _scaled_grouped_mm_fallback(
            self,
            mat2,
            scale_a,
            scale_b,
            offs,
            bias,
            output_dtype,
            a_is_2d,
            b_is_2d,
            num_groups,
        )

    if self.stride(-2) > 1 and self.stride(-1) > 1:
        self = self.contiguous()
    if mat2.stride(-2) > 1 and mat2.stride(-1) > 1:
        mat2 = mat2.contiguous()

    out = torch.empty(out_shape, dtype=output_dtype, device=self.device)
    if out.numel() == 0:
        return out

    block_m = 32
    block_n = 32
    max_m_tiles = triton.cdiv(M, block_m)
    max_n_tiles = triton.cdiv(N, block_n)
    grid = (num_groups, max_m_tiles * max_n_tiles)

    with torch_device_fn.device(self.device):
        _scaled_grouped_mm_kernel[grid](
            self,
            mat2,
            scale_a,
            scale_b,
            offs,
            bias,
            out,
            M,
            N,
            K,
            self.stride(0) if not a_is_2d else 0,
            self.stride(-2),
            self.stride(-1),
            mat2.stride(0) if not b_is_2d else 0,
            mat2.stride(-2),
            mat2.stride(-1),
            out.stride(0) if out.dim() == 3 else 0,
            out.stride(-2),
            out.stride(-1),
            scale_a.stride(0) if scale_a.dim() == 2 else M,
            scale_b.stride(0) if scale_b.dim() == 2 else N,
            A_IS_2D=a_is_2d,
            B_IS_2D=b_is_2d,
            BIAS_MODE=bias_mode,
            MAX_N_TILES=max_n_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=32,
            num_warps=4,
            num_stages=1,
        )
    return out
