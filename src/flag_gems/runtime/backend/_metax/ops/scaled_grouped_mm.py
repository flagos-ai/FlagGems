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

"""MetaX grouped E4M3/INT8 GEMM with FP32 row and column scales.

Offsets stay on the device. Ragged M/N groups are mapped to output tiles by
a parallel prefix sum; no per-group host dispatch or floating GEMM fallback
is used for these quantized input types.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from flag_gems.ops.scaled_grouped_mm import (
    _check_dims,
    _decode_e4m3,
    _normalize_bias,
    _normalize_scale,
    _resolve_shapes,
)
from flag_gems.ops.scaled_grouped_mm import (
    scaled_grouped_mm as _generic_scaled_grouped_mm,
)
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_E4M3_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def _select_config(M, N, K, num_groups, mode, is_int8):
    if is_int8:
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
    GROUP_MAJOR: tl.constexpr = False,
):
    # LibEntry does not key compiler options; include the warp count explicitly
    # so different launch configurations cannot alias.
    tl.static_assert(NUM_WARPS > 0)
    if GROUP_MAJOR:
        tl.static_assert(IS_INT8 and MODE == 0)
    # MODE: ragged M, ragged N, ragged K, regular batch.
    tile = tl.program_id(0).to(tl.int64)
    other_tile = tl.program_id(1).to(tl.int64)
    zero = tl.full((), 0, tl.int64)
    m_start, n_start, k_start = zero, zero, zero
    m_size = tl.full((), M, tl.int64)
    n_size = tl.full((), N, tl.int64)
    k_size = tl.full((), K, tl.int64)

    if MODE == 0 or MODE == 1:
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
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


def scaled_grouped_mm(
    self: torch.Tensor,
    mat2: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    offs: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_result: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    use_fast_accum: bool = False,
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
        return _generic_scaled_grouped_mm(
            self,
            mat2,
            scale_a,
            scale_b,
            offs=offs,
            bias=bias,
            scale_result=scale_result,
            out_dtype=out_dtype,
            use_fast_accum=use_fast_accum,
        )

    logger.debug("GEMS_METAX SCALED_GROUPED_MM")
    if scale_result is not None:
        raise RuntimeError("scale_result is not supported for scaled_grouped_mm")
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
        raise ValueError("INT8 scaled_grouped_mm requires K <= 131071")
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
    out = torch.empty(out_shape, dtype=output_dtype, device=self.device)
    if out.numel() == 0:
        return out

    bm, bn, bk, warps = _select_config(M, N, K, groups, mode, is_int8)
    if mode == 0:
        grid = (triton.cdiv(M, bm) + groups - 1, triton.cdiv(N, bn))
    elif mode == 1:
        grid = (triton.cdiv(N, bn) + groups - 1, triton.cdiv(M, bm))
    else:
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn), groups)
    # A wider N makes group-local traversal worthwhile; narrower GEMMs retain
    # the original tile order with the tuned four-warp tile configuration.
    group_major = is_int8 and mode == 0 and M >= 4096 and N >= 1024 and K >= 512
    if group_major:
        # Use the original host upper bound; excess programs return after the
        # compact GPU prefix lookup. No offsets are copied to the host.
        grid = (grid[0] * grid[1],)
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
            GROUP_MAJOR=group_major,
            num_warps=warps,
            num_stages=2,
            pipeline="basic",
            enable_fp_fusion=False,
        )
    return out
