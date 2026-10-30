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

"""MetaX W8A8 grouped GEMM with matching INT8 or E4M3 inputs.

Offsets stay on the device. Small ragged-M workloads map CTAs directly to
groups; other ragged M/N groups use a parallel prefix sum. Neither path uses
per-group host dispatch or a floating GEMM fallback for quantized inputs.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_E4M3_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


BIAS_NONE = 0
BIAS_VECTOR = 1
BIAS_GROUPED = 2


@triton.jit
def _decode_e4m3(bits, FNUZ: tl.constexpr):
    # Load FP8 storage as bytes: several backends cannot represent FP8 pointers.
    bits = bits.to(tl.int32)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    if FNUZ:
        power = ((exponent + 119) << 23).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.0009765625
        is_nan = bits == 128
    else:
        power = ((exponent + 120) << 23).to(tl.float32, bitcast=True)
        subnormal = mantissa.to(tl.float32) * 0.001953125
        is_nan = (bits & 127) == 127
    value = tl.where(exponent == 0, subnormal, power * (1.0 + mantissa * 0.125))
    value = tl.where((bits & 128) != 0, -value, value)
    return tl.where(is_nan, float("nan"), value).to(tl.float16)


def _check_dims(mat_a, mat_b):
    if mat_a.dim() not in (2, 3):
        raise RuntimeError("mat_a has to be 2D or 3D")
    if mat_b.dim() not in (2, 3):
        raise RuntimeError("mat_b has to be 2D or 3D")
    if mat_a.dtype != mat_b.dtype:
        raise RuntimeError(
            f"mat_a and mat_b must have the same dtype, got {mat_a.dtype} and {mat_b.dtype}"
        )


def _check_offsets(offs, need_offsets, num_groups):
    if need_offsets:
        if offs is None:
            raise RuntimeError("offs must be provided when either input is 2D")
        if offs.dim() != 1:
            raise RuntimeError("offs has to be 1D")
        if offs.dtype != torch.int32:
            raise RuntimeError("offs has to be int32")
        if offs.numel() != num_groups:
            raise RuntimeError(
                f"offs length must match the group count, got {offs.numel()} and {num_groups}"
            )
        return offs.contiguous()

    if offs is not None:
        raise RuntimeError("offs must be None when both inputs are 3D")
    return None


def _resolve_shapes(mat_a, mat_b, offs):
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    need_offsets = a_is_2d or b_is_2d

    if a_is_2d:
        M, K = mat_a.shape
    else:
        num_groups_a, M, K = mat_a.shape

    if b_is_2d:
        BK, N = mat_b.shape
    else:
        num_groups_b, BK, N = mat_b.shape

    if K != BK:
        raise RuntimeError(
            f"mat_a and mat_b shapes cannot be multiplied ({K} and {BK})"
        )

    if need_offsets:
        if a_is_2d and b_is_2d:
            num_groups = offs.numel() if offs is not None else 0
            out_shape = (num_groups, M, N)
        elif a_is_2d:
            num_groups = num_groups_b
            out_shape = (M, N)
        else:
            num_groups = num_groups_a
            out_shape = (M, N)
    else:
        if num_groups_a != num_groups_b:
            raise RuntimeError(
                f"matrix batch sizes have to match, got {num_groups_a} and {num_groups_b}"
            )
        num_groups = num_groups_a
        out_shape = (num_groups, M, N)

    offs = _check_offsets(offs, need_offsets, num_groups)
    return a_is_2d, b_is_2d, num_groups, M, N, K, out_shape, offs


def _normalize_scale(scale, mat, *, dim, num_groups, scale_multiplier, name):
    if scale.dtype != torch.float32:
        raise RuntimeError(f"{name} must be a float32 tensor")

    if mat.dim() == 2:
        expected = mat.shape[dim] * scale_multiplier
        if scale.dim() != 1 or scale.numel() != expected:
            raise RuntimeError(f"{name} must be a 1D tensor with length {expected}")
        return scale.reshape(expected).contiguous()

    expected_shape = (num_groups, mat.shape[1 + dim])
    if scale.dim() != 2 or tuple(scale.shape) != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}")
    return scale.contiguous()


def _normalize_bias(bias, *, a_is_2d, b_is_2d, num_groups, N):
    if bias is None:
        return None, BIAS_NONE

    if bias.dim() == 1 and bias.numel() == N:
        return bias.contiguous(), BIAS_VECTOR

    can_use_grouped_bias = not (b_is_2d and not a_is_2d)
    if can_use_grouped_bias and bias.numel() == num_groups * N:
        return bias.reshape(num_groups, N).contiguous(), BIAS_GROUPED

    expected = f"({N},)"
    if can_use_grouped_bias:
        expected = f"{expected} or ({num_groups}, {N})"
    raise RuntimeError(f"bias must have shape {expected}")


def _use_sparse_int8(M, N, K, groups):
    # Sparse token batches need less staging and more work per K/N tile.
    return M < groups and (
        (64 <= N <= 1024 and K >= 2048) or (N >= 1024 and 128 <= K <= 512)
    )


def _select_config(M, N, K, num_groups, mode, is_int8):
    if is_int8:
        if mode == 0 and _use_sparse_int8(M, N, K, num_groups):
            return (16, 64, 256, 4) if K >= 2048 else (16, 128, 128, 4)
        # Packed M alone overestimates the useful tile height for small experts.
        if mode == 0 and M <= 16 * num_groups and N >= 64 and K >= 128:
            return (16, 64, 128, 4)
        # Long-K projections reuse operands across a larger reduction tile.
        if mode == 0 and M >= 4096 and 256 <= N <= 512 and K >= 2048:
            return (128, 128, 128, 8)
        # Larger expert groups can amortize wider tiles and their shared memory.
        if mode == 0 and N == 256 and K >= 512 and M >= 256 * num_groups:
            return (256, 128, 64, 8)
        if mode == 0 and M >= 128 * num_groups and K >= 512:
            if N == 256:
                return (128, 128, 64, 4)
            if N >= 1024 and K >= 1024:
                return (128, 256, 64, 8)
        if mode != 0:
            return (32, 64, 64, 4)
        if N >= 512 and K >= 512 and M >= 4096:
            return (128, 128, 64, 4)
        if N >= 256 and K >= 256 and M >= 512:
            return (64, 64, 64, 4)
        if N >= 64 and K >= 128:
            return (32, 64, 128, 4)
        return (32, 64, 64, 4)
    if _use_predecode(M, N, K, num_groups, mode):
        return (64, 64, 64, 4) if N <= 256 else (128, 128, 64, 8)
    # Byte decoding keeps extra values live; narrow M avoids spilling on C550.
    return (16, 64, 64, 4)


def _use_predecode(M, N, K, num_groups, mode):
    # Large M groups reuse B across tiles, amortizing FP16 materialization.
    return mode == 0 and M >= 128 * num_groups and N >= 256 and K >= 256


@libentry()
@triton.jit
def _decode_operand_kernel(
    X,
    Y,
    SIZE: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    stride_xg: tl.constexpr,
    stride_xr: tl.constexpr,
    stride_xc: tl.constexpr,
    COLUMN_MAJOR: tl.constexpr,
    FNUZ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    group = offset // (ROWS * COLS)
    within = offset % (ROWS * COLS)
    if COLUMN_MAJOR:
        row = within % ROWS
        col = within // ROWS
    else:
        row = within // COLS
        col = within % COLS
    bits = tl.load(
        X + group * stride_xg + row * stride_xr + col * stride_xc,
        offset < SIZE,
        other=0,
    )
    tl.store(Y + offset, _decode_e4m3(bits, FNUZ), offset < SIZE)


def _decode_operand(operand: torch.Tensor, fnuz: bool) -> torch.Tensor:
    """Materialize logical FP16 values without copying or reordering via Torch."""
    rows, cols = operand.shape[-2:]
    column_major = operand.stride(-2) == 1 and operand.stride(-1) > 1
    if column_major:
        strides = (1, rows) if operand.ndim == 2 else (rows * cols, 1, rows)
        decoded = torch.empty_strided(
            operand.shape, strides, device=operand.device, dtype=torch.float16
        )
    else:
        decoded = torch.empty(operand.shape, device=operand.device, dtype=torch.float16)
    if operand.numel():
        # Larger spans reduce launch-grid overhead for full expert banks.
        block = 2048 if operand.numel() >= 8 * 1024 * 1024 else 1024
        _decode_operand_kernel[(triton.cdiv(operand.numel(), block),)](
            operand,
            decoded,
            operand.numel(),
            rows,
            cols,
            operand.stride(0) if operand.ndim == 3 else 0,
            operand.stride(-2),
            operand.stride(-1),
            COLUMN_MAJOR=column_major,
            FNUZ=fnuz,
            BLOCK=block,
            num_warps=4,
            num_stages=2,
            pipeline="basic",
            enable_fp_fusion=False,
        )
    return decoded


@libentry()
@triton.jit
def _scaled_grouped_mm_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Offs,
    ActiveGroups,
    Bias,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    stride_ag: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    MODE: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    IS_INT8: tl.constexpr,
    DECODE_FP8: tl.constexpr,
    FNUZ: tl.constexpr,
    GROUP_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    GROUP_MAJOR: tl.constexpr = False,
    DIRECT_GROUP: tl.constexpr = False,
    COMPACT_GROUPS: tl.constexpr = False,
):
    # LibEntry does not key compiler options; make both launch options explicit.
    tl.static_assert(NUM_WARPS > 0 and NUM_STAGES > 0)
    if GROUP_MAJOR:
        tl.static_assert(IS_INT8 and MODE == 0)
    if DIRECT_GROUP:
        tl.static_assert(MODE == 0 and not GROUP_MAJOR)
    # MODE: ragged M, ragged N, ragged K, regular batch.
    tile = tl.program_id(0).to(tl.int64)
    other_tile = tl.program_id(1).to(tl.int64)
    zero = tl.full((), 0, tl.int64)
    m_start, n_start, k_start = zero, zero, zero
    m_size = tl.full((), M, tl.int64)
    n_size = tl.full((), N, tl.int64)
    k_size = tl.full((), K, tl.int64)

    if DIRECT_GROUP:
        # A flattened (group, N tile) grid avoids scanning every group's
        # offsets for each CTA and does not depend on the grid-Y limit.
        DIRECT_TILES_N: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
        group = tile // DIRECT_TILES_N
        if COMPACT_GROUPS:
            active_count = tl.load(ActiveGroups + NUM_GROUPS)
            if group >= active_count:
                return
            group = tl.load(ActiveGroups + group).to(tl.int64)
        pid_n = tile % DIRECT_TILES_N
        pid_m = zero
        m_start = tl.load(Offs + group - 1, group > 0, other=0).to(tl.int64)
        m_size = tl.load(Offs + group).to(tl.int64) - m_start
    elif MODE == 0 or MODE == 1:
        groups = tl.arange(0, GROUP_BLOCK)
        ends = tl.load(Offs + groups, groups < NUM_GROUPS, other=0).to(tl.int64)
        starts = tl.load(
            Offs + groups - 1, (groups > 0) & (groups < NUM_GROUPS), other=0
        ).to(tl.int64)
        sizes = tl.where(groups < NUM_GROUPS, ends - starts, 0)
        if MODE == 0:
            tile_counts = tl.cdiv(sizes, BLOCK_M)
            if GROUP_MAJOR:
                TILES_N: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
                # Preserve the INT64 prefix while assigning every N tile to
                # the same contiguous program-ID interval for its group.
                tile_counts = tile_counts * TILES_N
            tile_ends = tl.cumsum(tile_counts, 0)
        else:
            tile_ends = tl.cumsum(tl.cdiv(sizes, BLOCK_N), 0)
        group = tl.sum(((tile >= tile_ends) & (groups < NUM_GROUPS)).to(tl.int32), 0)
        if group >= NUM_GROUPS:
            return
        group_start = tl.sum(tl.where(groups == group, starts, 0), 0)
        group_size = tl.sum(tl.where(groups == group, sizes, 0), 0)
        first_tile = tl.sum(tl.where(groups == group - 1, tile_ends, 0), 0)
        if MODE == 0:
            m_start, m_size = group_start, group_size
            if GROUP_MAJOR:
                local_tile = tile - first_tile
                # Neighboring programs traverse N before advancing M.
                pid_m = local_tile // TILES_N
                pid_n = local_tile % TILES_N
            else:
                pid_m, pid_n = tile - first_tile, other_tile
        else:
            n_start, n_size = group_start, group_size
            pid_m, pid_n = other_tile, tile - first_tile
        group = group.to(tl.int64)
    else:
        group = other_tile
        pid_m = tile % tl.cdiv(M, BLOCK_M)
        pid_n = tile // tl.cdiv(M, BLOCK_M)
        if MODE == 2:
            k_start = tl.load(Offs + group - 1, group > 0, other=0).to(tl.int64)
            k_size = tl.load(Offs + group).to(tl.int64) - k_start

    # Direct mapping keeps all M tiles for an expert on one CTA. Empty groups
    # execute zero iterations; all existing mappings execute exactly one tile.
    if DIRECT_GROUP:
        m_tile_count = tl.cdiv(m_size, BLOCK_M)
    else:
        m_tile_count = 1
    for m_iteration in range(0, m_tile_count):
        offs_m = (pid_m + m_iteration) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K).to(tl.int64)
        a_base = A + (group * stride_ag if MODE == 1 or MODE == 3 else 0)
        b_base = B + (group * stride_bg if MODE == 0 or MODE == 3 else 0)
        a_rows = (m_start + offs_m) * stride_am
        b_cols = (n_start + offs_n) * stride_bn
        if IS_INT8:
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k_offset in range(0, k_size, BLOCK_K):
            relative_k = k_offset + offs_k
            a = tl.load(
                a_base + a_rows[:, None] + (k_start + relative_k[None, :]) * stride_ak,
                (offs_m[:, None] < m_size) & (relative_k[None, :] < k_size),
                other=0,
            )
            b = tl.load(
                b_base + (k_start + relative_k[:, None]) * stride_bk + b_cols[None, :],
                (relative_k[:, None] < k_size) & (offs_n[None, :] < n_size),
                other=0,
            )
            if IS_INT8:
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
            else:
                if DECODE_FP8:
                    a = _decode_e4m3(a, FNUZ)
                    b = _decode_e4m3(b, FNUZ)
                acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

        sa_base = m_start if MODE == 0 else group * M
        sb_base = n_start if MODE == 1 else group * N
        scale_a = tl.load(ScaleA + sa_base + offs_m, offs_m < m_size, other=0)
        scale_b = tl.load(ScaleB + sb_base + offs_n, offs_n < n_size, other=0)
        value = acc.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
        if BIAS_MODE != 0:
            if BIAS_MODE == 2:
                bias_base = group * N
            elif MODE == 1:
                bias_base = n_start
            else:
                bias_base = 0
            bias = tl.load(Bias + bias_base + offs_n, offs_n < n_size, other=0)
            value += bias.to(tl.float32)[None, :]
        if MODE == 0 or MODE == 1:
            c_ptrs = C + (m_start + offs_m[:, None]) * N + n_start + offs_n[None, :]
        else:
            c_ptrs = C + group * M * N + offs_m[:, None] * N + offs_n[None, :]
        tl.store(c_ptrs, value, (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size))


def _grouped_mm_quantized(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
    *,
    out: Optional[torch.Tensor] = None,
    active_groups: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Grouped signed INT8 or E4M3 GEMM; offsets contain cumulative ends.

    Offsets must be nondecreasing and end at the split-axis extent. These
    value preconditions stay on the caller, avoiding device synchronization.
    INT8 K is limited to 131071 to keep every signed dot in INT32 range.
    """
    if not isinstance(self, torch.Tensor) or not isinstance(mat2, torch.Tensor):
        raise TypeError("mat_a and mat_b must be tensors")
    _check_dims(self, mat2)
    is_int8 = self.dtype == torch.int8
    if not is_int8 and self.dtype not in _E4M3_DTYPES:
        raise TypeError("the quantized grouped GEMM requires INT8 or E4M3 inputs")

    logger.debug("GEMS_METAX GROUPED_MM_W8A8")
    if scale_result is not None:
        raise RuntimeError("scale_result is not supported for quantized grouped GEMM")
    output_dtype = torch.bfloat16 if out_dtype is None else out_dtype
    float_outputs = (torch.float16, torch.bfloat16, torch.float32)
    if output_dtype not in float_outputs:
        raise TypeError("out_dtype must be FP16, BF16 or FP32")
    if self.device.type != "cuda" or mat2.device != self.device:
        raise ValueError("mat_a and mat_b must be on the same MetaX device")
    for name, tensor in (("scale_a", scale_a), ("scale_b", scale_b)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on the input device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, tensor in (("offs", offs), ("bias", bias)):
        if tensor is not None:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a tensor")
            if tensor.device != self.device:
                raise ValueError(f"{name} must be on the input device")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
    if bias is not None and bias.dtype not in float_outputs:
        raise TypeError("bias must be FP16, BF16 or FP32")

    a_is_2d, b_is_2d, groups, M, N, K, out_shape, offs = _resolve_shapes(
        self, mat2, offs
    )
    if is_int8 and K > 131071:
        raise ValueError("INT8 grouped GEMM requires K <= 131071")
    multiplier = groups if a_is_2d and b_is_2d else 1
    scale_a = _normalize_scale(
        scale_a,
        self,
        dim=0,
        num_groups=groups,
        scale_multiplier=multiplier,
        name="scale_a",
    )
    scale_b = _normalize_scale(
        scale_b,
        mat2,
        dim=1,
        num_groups=groups,
        scale_multiplier=multiplier,
        name="scale_b",
    )
    bias, bias_mode = _normalize_bias(
        bias, a_is_2d=a_is_2d, b_is_2d=b_is_2d, num_groups=groups, N=N
    )
    mode = (0 if a_is_2d else 1) if a_is_2d != b_is_2d else (2 if a_is_2d else 3)
    if not groups and ((mode == 0 and M) or (mode == 1 and N)):
        raise ValueError("nonempty ragged extent requires at least one group")
    if out is None:
        out = torch.empty(out_shape, dtype=output_dtype, device=self.device)
    if out.numel() == 0:
        return out

    bm, bn, bk, warps = _select_config(M, N, K, groups, mode, is_int8)
    # The larger INT8 tiles fit 16-bit outputs, but FP32 epilogue staging
    # exceeds C550 shared memory. Keep the established FP32 configurations.
    if is_int8 and output_dtype == torch.float32:
        if (bm, bn) == (256, 128):
            bm, bn, bk, warps = (64, 64, 64, 4)
        elif (bm, bn) == (128, 256):
            bm, bn, bk, warps = (128, 128, 64, 4)
        elif (bm, bn, bk, warps) == (128, 128, 128, 8):
            bm, bn, bk, warps = (128, 128, 64, 4)
    # Direct expert mapping avoids prefix work for small average group sizes;
    # its inner M loop also handles skewed partitions with larger individual groups.
    stages = 2 if is_int8 else 1
    if is_int8 and mode == 0 and _use_sparse_int8(M, N, K, groups):
        stages = 1
    compact_groups = active_groups is not None
    long_k_tiles = is_int8 and mode == 0 and (bm, bn, bk, warps) == (128, 128, 128, 8)
    direct_group = mode == 0 and M <= 64 * groups and not long_k_tiles
    if mode == 0:
        grid = (triton.cdiv(M, bm) + groups - 1, triton.cdiv(N, bn))
    elif mode == 1:
        grid = (triton.cdiv(N, bn) + groups - 1, triton.cdiv(M, bm))
    else:
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn), groups)
    # Keep neighboring CTAs on one expert to reuse long-K operands in cache.
    group_major = (
        not direct_group
        and is_int8
        and mode == 0
        and M >= 4096
        and (N >= 1024 or long_k_tiles)
        and K >= 512
    )
    if group_major:
        # Use the original host upper bound; excess programs return after the
        # compact GPU prefix lookup. No offsets are copied to the host.
        grid = (grid[0] * grid[1],)
    if direct_group:
        grid = (
            (min(groups, M) * triton.cdiv(N, bn),)
            if compact_groups
            else (groups * triton.cdiv(N, bn),)
        )
    fnuz = self.dtype == torch.float8_e4m3fnuz
    a = self if is_int8 else self.view(torch.uint8)
    b = mat2 if is_int8 else mat2.view(torch.uint8)
    predecode = not is_int8 and _use_predecode(M, N, K, groups, mode)
    with torch_device_fn.device(self.device):
        if predecode:
            a = _decode_operand(a, fnuz)
            b = _decode_operand(b, fnuz)
        _scaled_grouped_mm_kernel[grid](
            a,
            b,
            scale_a,
            scale_b,
            offs,
            active_groups,
            bias,
            out,
            M,
            N,
            K,
            groups,
            a.stride(0) if not a_is_2d else 0,
            a.stride(-2),
            a.stride(-1),
            b.stride(0) if not b_is_2d else 0,
            b.stride(-2),
            b.stride(-1),
            MODE=mode,
            BIAS_MODE=bias_mode,
            IS_INT8=is_int8,
            DECODE_FP8=not is_int8 and not predecode,
            FNUZ=fnuz,
            GROUP_BLOCK=triton.next_power_of_2(groups),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            NUM_WARPS=warps,
            NUM_STAGES=stages,
            GROUP_MAJOR=group_major,
            DIRECT_GROUP=direct_group,
            COMPACT_GROUPS=compact_groups,
            num_warps=warps,
            num_stages=stages,
            pipeline="basic",
            enable_fp_fusion=False,
        )
    return out


@libentry()
@triton.jit
def _expert_offsets(
    Rows,
    Offsets,
    ActiveGroups,
    E: tl.constexpr,
    BLOCK: tl.constexpr,
    COMPACT: tl.constexpr = False,
):
    experts = tl.arange(0, BLOCK)
    counts = tl.load(Rows + experts, experts < E, other=0)
    tl.store(Offsets + experts, tl.cumsum(counts, 0), experts < E)
    if COMPACT:
        active = (experts < E) & (counts > 0)
        ranks = tl.cumsum(active.to(tl.int32), 0)
        tl.store(ActiveGroups + ranks - 1, experts, active)
        tl.store(ActiveGroups + E, tl.sum(active.to(tl.int32), 0))


def grouped_mm_w8a8(A, As, B, Bs, C, m_indices, m_rows) -> None:
    """Write unpadded grouped NT GEMM into caller-owned BF16 C.

    A[M,K] and B[E,N,K] must have matching INT8 or E4M3 dtypes.
    As[M,1] and Bs[E,N,1] are FP32. The INT32 m_indices[M] and
    m_rows[E] describe contiguous rows in ascending expert order.
    Counts must be nonnegative, sum to M and agree with m_indices;
    these device-value preconditions belong to the caller, with no host sync.
    Inputs are already quantized and grouped. No routing, activation or
    top-k weighted reduction is performed. All tensors must be contiguous.
    """
    tensors = (A, As, B, Bs, C, m_indices, m_rows)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("all grouped_mm_w8a8 arguments must be tensors")
    if A.ndim != 2 or B.ndim != 3:
        raise ValueError("A must be [M,K] and B must be [E,N,K]")
    m, k = A.shape
    e, n, weight_k = B.shape
    if A.dtype not in (torch.int8, *_E4M3_DTYPES) or B.dtype != A.dtype:
        raise TypeError("A and B must have matching INT8 or E4M3 dtypes")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise TypeError("As and Bs must be FP32")
    if C.dtype != torch.bfloat16:
        raise TypeError("C must be BF16")
    if m_indices.dtype != torch.int32 or m_rows.dtype != torch.int32:
        raise TypeError("m_indices and m_rows must be INT32")
    if A.device.type != "cuda" or any(t.device != A.device for t in tensors):
        raise ValueError("all tensors must be on the same MetaX device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("all tensors must be contiguous")
    expected = ((m, 1), (e, n, 1), (m, n), (m,), (e,))
    if weight_k != k or any(
        tuple(t.shape) != shape
        for t, shape in zip((As, Bs, C, m_indices, m_rows), expected)
    ):
        raise ValueError("incompatible scales, output or expert metadata shapes")
    if A.dtype == torch.int8 and k > 131071:
        raise ValueError("INT8 grouped GEMM requires K <= 131071")
    if m and not e:
        raise ValueError("nonempty input requires at least one expert")
    c_start = C.data_ptr()
    c_end = c_start + C.numel() * C.element_size()
    for tensor in (A, As, B, Bs, m_indices, m_rows):
        start = tensor.data_ptr()
        end = start + tensor.numel() * tensor.element_size()
        if c_start < end and start < c_end:
            raise ValueError("C must not overlap an input")
    if m == 0 or n == 0:
        return
    # The active count is at most min(M, E), so no host readback is needed.
    # Long-K batches compact only when the upper bound is sufficiently sparse.
    compact = (
        A.dtype == torch.int8
        and _use_sparse_int8(m, n, k, e)
        and (k <= 512 or m <= e // 4)
    )
    metadata = torch.empty(
        (2 * e + 1 if compact else e,), dtype=torch.int32, device=A.device
    )
    offsets = metadata[:e]
    active_groups = metadata[e:] if compact else None
    with torch_device_fn.device(A.device):
        _expert_offsets[(1,)](
            m_rows,
            offsets,
            active_groups,
            e,
            triton.next_power_of_2(e),
            COMPACT=compact,
            num_warps=4,
        )
        _grouped_mm_quantized(
            A,
            B.transpose(-1, -2),
            As[:, 0],
            Bs[:, :, 0],
            offs=offsets,
            out_dtype=C.dtype,
            out=C,
            active_groups=active_groups,
        )
