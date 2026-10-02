# Kunlunxin backend override for scaled grouped matrix multiplication.

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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils.device_info import get_sm_count

from .contiguous import contiguous

logger = logging.getLogger(__name__)

BIAS_NONE = 0
BIAS_VECTOR = 1
BIAS_GROUPED = 2


@triton.jit
def _decode_e4m3(bits, FNUZ: tl.constexpr):
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


@triton.jit
def _decode_e4m3_tensor(
    X,
    Y,
    NUMEL: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    stride_g: tl.constexpr,
    stride_m: tl.constexpr,
    stride_n: tl.constexpr,
    FNUZ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    group = offsets // (ROWS * COLS)
    row = (offsets // COLS) % ROWS
    col = offsets % COLS
    bits = tl.load(
        X + group * stride_g + row * stride_m + col * stride_n,
        mask=offsets < NUMEL,
        other=0,
    )
    tl.store(Y + offsets, _decode_e4m3(bits, FNUZ), mask=offsets < NUMEL)


def _decode_e4m3_operand(operand, fnuz):
    decoded = torch.empty(operand.shape, dtype=torch.float16, device=operand.device)
    if operand.numel():
        with torch_device_fn.device(operand.device):
            _decode_e4m3_tensor[(triton.cdiv(operand.numel(), 256),)](
                operand.view(torch.uint8),
                decoded,
                operand.numel(),
                operand.shape[-2],
                operand.shape[-1],
                operand.stride(0) if operand.dim() == 3 else 0,
                operand.stride(-2),
                operand.stride(-1),
                fnuz,
                BLOCK=256,
                num_warps=4,
                num_stages=1,
            )
    return decoded


def get_autotune_config():
    logger.debug("GEMS_KUNLUNXIN GET_AUTOTUNE_CONFIG")
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        # [sggemm-klx 2026-09-19] large-M / wide-N candidates
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3,
            num_warps=4,
        ),
    ]


def _prune_scaled_grouped_mm_configs(configs, named_args, **kwargs):
    """Keep the conservative tile for inputs materialized from FP8."""
    e4m3 = kwargs.get("E4M3", named_args.get("E4M3", False))
    if not e4m3:
        return configs
    return [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
            num_stages=1,
            num_warps=2,
        )
    ]


@libentry()
@libtuner(
    configs=get_autotune_config(),
    key=["M", "N", "K", "A_IS_2D", "B_IS_2D", "E4M3"],
    prune_configs_by={"early_config_prune": _prune_scaled_grouped_mm_configs},
    warmup=2,
    rep=4,
)
@triton.jit
def scaled_grouped_mm_kernel(
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
    stride_cg: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cn: tl.constexpr,
    stride_sag: tl.constexpr,
    stride_sbg: tl.constexpr,
    A_IS_2D: tl.constexpr,
    B_IS_2D: tl.constexpr,
    BIAS_MODE: tl.constexpr,
    E4M3: tl.constexpr,
    FNUZ: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    total_grid = tl.num_programs(axis=0).to(tl.int64)
    tile_idx = tl.program_id(axis=0).to(tl.int64)
    zero = tl.full((), 0, dtype=tl.int64)
    iterated_tiles = zero
    offset_end = zero

    for group_idx in tl.range(NUM_GROUPS):
        m_start = zero
        n_start = zero
        k_start = zero
        m_size = tl.full((), M, dtype=tl.int64)
        n_size = tl.full((), N, dtype=tl.int64)
        k_size = tl.full((), K, dtype=tl.int64)
        scale_a_start = zero
        scale_b_start = zero

        if A_IS_2D or B_IS_2D:
            offset_start = offset_end
            offset_end = tl.load(Offs + group_idx).to(tl.int64)
            group_size = offset_end - offset_start

            if A_IS_2D and not B_IS_2D:
                m_start = offset_start
                m_size = group_size
                scale_a_start = offset_start
            elif not A_IS_2D and B_IS_2D:
                n_start = offset_start
                n_size = group_size
                scale_b_start = offset_start
            else:
                k_start = offset_start
                k_size = group_size
                scale_a_start = group_idx * M
                scale_b_start = group_idx * N
        else:
            scale_a_start = group_idx * stride_sag
            scale_b_start = group_idx * stride_sbg

        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        num_tiles = num_m_tiles * num_n_tiles
        current_problem_end = iterated_tiles + num_tiles

        if tile_idx >= iterated_tiles and tile_idx < current_problem_end:
            loop_count = (current_problem_end - tile_idx + total_grid - 1) // total_grid
            for _ in tl.range(loop_count):
                tile_idx_in_group = tile_idx - iterated_tiles
                pid_m = tile_idx_in_group % num_m_tiles
                pid_n = tile_idx_in_group // num_m_tiles

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k_offset in range(0, k_size, BLOCK_K):
                    group_offs_k = k_offset + offs_k
                    a_ptrs = (
                        A
                        + (0 if A_IS_2D else group_idx * stride_ag)
                        + (m_start + offs_m[:, None]) * stride_am
                        + (k_start + group_offs_k[None, :]) * stride_ak
                    )
                    b_ptrs = (
                        B
                        + (0 if B_IS_2D else group_idx * stride_bg)
                        + (k_start + group_offs_k[:, None]) * stride_bk
                        + (n_start + offs_n[None, :]) * stride_bn
                    )
                    a_mask = (offs_m[:, None] < m_size) & (
                        group_offs_k[None, :] < k_size
                    )
                    b_mask = (group_offs_k[:, None] < k_size) & (
                        offs_n[None, :] < n_size
                    )
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)
                    acc += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

                if A_IS_2D:
                    scale_a = tl.load(
                        ScaleA + scale_a_start + offs_m[:, None],
                        mask=offs_m[:, None] < m_size,
                        other=0.0,
                    )
                else:
                    scale_a = tl.load(
                        ScaleA + group_idx * stride_sag + offs_m[:, None],
                        mask=offs_m[:, None] < m_size,
                        other=0.0,
                    )

                if B_IS_2D:
                    scale_b = tl.load(
                        ScaleB + scale_b_start + offs_n[None, :],
                        mask=offs_n[None, :] < n_size,
                        other=0.0,
                    )
                else:
                    scale_b = tl.load(
                        ScaleB + group_idx * stride_sbg + offs_n[None, :],
                        mask=offs_n[None, :] < n_size,
                        other=0.0,
                    )

                c = acc * scale_a * scale_b

                if BIAS_MODE == 1:
                    if B_IS_2D and not A_IS_2D:
                        bias_offset = n_start
                    else:
                        bias_offset = 0
                    bias = tl.load(
                        Bias + bias_offset + offs_n[None, :],
                        mask=offs_n[None, :] < n_size,
                        other=0.0,
                    )
                    c += bias
                elif BIAS_MODE == 2:
                    bias = tl.load(
                        Bias + group_idx * N + offs_n[None, :],
                        mask=offs_n[None, :] < n_size,
                        other=0.0,
                    )
                    c += bias

                c_mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
                if A_IS_2D != B_IS_2D:
                    c_ptrs = (
                        C
                        + (m_start + offs_m[:, None]) * stride_cm
                        + (n_start + offs_n[None, :]) * stride_cn
                    )
                else:
                    c_ptrs = (
                        C
                        + group_idx * stride_cg
                        + offs_m[:, None] * stride_cm
                        + offs_n[None, :] * stride_cn
                    )
                tl.store(c_ptrs, c, mask=c_mask)
                tile_idx += total_grid

        iterated_tiles = current_problem_end


def _float8_dtypes():
    return tuple(
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        )
        if hasattr(torch, name)
    )


def _is_float8_dtype(dtype):
    return dtype in _float8_dtypes()


def _default_out_dtype(dtype):
    if _is_float8_dtype(dtype):
        return torch.bfloat16
    return dtype


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
        return contiguous(offs)

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
        return contiguous(scale.reshape(expected))

    expected_shape = (num_groups, mat.shape[1 + dim])
    if scale.dim() != 2 or tuple(scale.shape) != expected_shape:
        raise RuntimeError(f"{name} must have shape {expected_shape}")
    return contiguous(scale)


def _normalize_bias(bias, *, a_is_2d, b_is_2d, num_groups, N):
    if bias is None:
        return None, BIAS_NONE

    if bias.dim() == 1 and bias.numel() == N:
        return contiguous(bias), BIAS_VECTOR

    can_use_grouped_bias = not (b_is_2d and not a_is_2d)
    if can_use_grouped_bias and bias.numel() == num_groups * N:
        return contiguous(bias.reshape(num_groups, N)), BIAS_GROUPED

    expected = f"({N},)"
    if can_use_grouped_bias:
        expected = f"{expected} or ({num_groups}, {N})"
    raise RuntimeError(f"bias must have shape {expected}")


def _supports_triton_dot(dtype):
    return dtype in (torch.float16, torch.bfloat16, torch.float32) or dtype in tuple(
        getattr(torch, name)
        for name in ("float8_e4m3fn", "float8_e4m3fnuz")
        if hasattr(torch, name)
    )


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
    logger.debug("GEMS_KUNLUNXIN SCALED_GROUPED_MM")
    if scale_result is not None:
        raise RuntimeError("scale_result is not supported for scaled_grouped_mm")

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
        raise NotImplementedError(
            "scaled_grouped_mm: dtype has no Kunlunxin device kernel; "
            "CPU fallback is intentionally disabled"
        )

    out = torch.empty(out_shape, dtype=output_dtype, device=self.device)
    if out.numel() == 0:
        return out

    e4m3 = self.dtype in tuple(
        getattr(torch, name)
        for name in ("float8_e4m3fn", "float8_e4m3fnuz")
        if hasattr(torch, name)
    )
    fnuz = self.dtype == getattr(torch, "float8_e4m3fnuz", None)
    if e4m3:
        # E4M3 values are exactly representable in FP16. Keep decoding separate
        # from the dot loop so its byte conversions retain their semantics.
        self = _decode_e4m3_operand(self, fnuz)
        mat2 = _decode_e4m3_operand(mat2, fnuz)
    else:
        if self.stride(-2) > 1 and self.stride(-1) > 1:
            self = contiguous(self)
        if mat2.stride(-2) > 1 and mat2.stride(-1) > 1:
            mat2 = contiguous(mat2)

    stride_ag = self.stride(0) if not a_is_2d else 0
    stride_am = self.stride(-2)
    stride_ak = self.stride(-1)
    stride_bg = mat2.stride(0) if not b_is_2d else 0
    stride_bk = mat2.stride(-2)
    stride_bn = mat2.stride(-1)
    stride_cg = out.stride(0) if out.dim() == 3 else 0
    stride_cm = out.stride(-2)
    stride_cn = out.stride(-1)
    stride_sag = scale_a.stride(0) if scale_a.dim() == 2 else 0
    stride_sbg = scale_b.stride(0) if scale_b.dim() == 2 else 0

    grid = (get_sm_count(),)
    with torch_device_fn.device(self.device):
        scaled_grouped_mm_kernel[grid](
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
            num_groups,
            stride_ag,
            stride_am,
            stride_ak,
            stride_bg,
            stride_bk,
            stride_bn,
            stride_cg,
            stride_cm,
            stride_cn,
            stride_sag,
            stride_sbg,
            A_IS_2D=a_is_2d,
            B_IS_2D=b_is_2d,
            BIAS_MODE=bias_mode,
            E4M3=e4m3,
            FNUZ=fnuz,
        )
    return out
