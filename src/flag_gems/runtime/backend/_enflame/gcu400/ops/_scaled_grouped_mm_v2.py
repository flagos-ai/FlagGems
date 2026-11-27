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

"""Grouped scaled matrix multiplication (``aten::_scaled_grouped_mm_v2``).

The operator computes a stack of per-group GEMMs whose operands carry
per-row / per-column (``RowWise``) or per-tensor (``TensorWise``) decoding
scales -- the shape of the grouped quantized GEMM used by quantized
Mixture-of-Experts layers.

The jagged ("varying") dimension is described by ``offs``, a monotonically
increasing ``int32`` tensor whose entry ``i`` marks the end of group ``i``:

* ``m_varying`` -- ``mat_a`` is 2D ``(M, K)`` and ``offs`` slices its rows,
  the output is ``(M, N)``.
* ``n_varying`` -- ``mat_b`` is 2D ``(K, N)`` and ``offs`` slices its columns,
  the output is ``(M, N)``.
* ``k_varying`` -- both operands are 2D and ``offs`` slices the contraction
  dimension, the output is ``(G, M, N)``.
* ``batch``     -- both operands are 3D ``(G, M, K)`` / ``(G, K, N)`` and
  ``offs`` must be ``None``.

Each group therefore contributes an ``(m_g, k_g) x (k_g, n_g)`` GEMM whose
result is scaled by the per-row ``scale_a`` and the per-column ``scale_b``
before being stored into the slice of the output that group owns.

All arithmetic happens in the Triton kernels below; the Python wrapper only
validates arguments and does shape/stride bookkeeping.  On GCU400 the fp8
GEMM runs as a persistent kernel over a fixed 48-program grid with two
launch variants: an autotuned even-K kernel (single-dimension load masks,
deep K tiles, hardware fp8 MMA) and a ragged-K fallback whose contraction
tail is split out of the main loop -- a K-tail branch left beside the even
main loop degrades it by two orders of magnitude on this backend.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

logger = logging.getLogger(__name__)

# Decoding recipes accepted in ``recipe_a`` / ``recipe_b``.
RECIPE_TENSOR_WISE = 0
RECIPE_ROW_WISE = 1

# Swizzling patterns accepted in ``swizzle_a`` / ``swizzle_b``.
SWIZZLE_NONE = 0

# How ``offs`` slices the operands.  ``MODE_BATCH`` keeps both operands whole
# and the other three name the jagged dimension.
MODE_BATCH = 0
MODE_M_VARYING = 1
MODE_N_VARYING = 2
MODE_K_VARYING = 3

# Scale granularity as seen by the kernel.
SCALE_TENSOR_WISE = 0
SCALE_ROW_WISE = 1

# The same modes, spelled as Triton compile-time constants.  A global read
# inside a ``@triton.jit`` body has to be an instantiated ``tl.constexpr``.
_MODE_BATCH = tl.constexpr(MODE_BATCH)
_MODE_M_VARYING = tl.constexpr(MODE_M_VARYING)
_MODE_N_VARYING = tl.constexpr(MODE_N_VARYING)
_MODE_K_VARYING = tl.constexpr(MODE_K_VARYING)
_SCALE_TENSOR_WISE = tl.constexpr(SCALE_TENSOR_WISE)

# ``mat_b`` has to be handed over in its transposed (column major) layout.
_MAT_B_TRANSPOSED_ERROR = "Expected mat2 to be transposed"
_OFFS_DTYPE_ERROR = "Offsets have to be int32"
_OUT_DTYPE_ERROR = (
    "Only bf16 high precision output types are supported for grouped gemm"
)
_BIAS_ERROR = "Bias not supported yet"

# GCU400 fixed persistent grid: 4 DIEs x 6 SIPs; 48 concurrent programs beat
# 4/8/24/96 in every sweep (same finding as ops/group_mm.py).  Do NOT use
# ``get_sm_count()`` here: torch_gcu reports ``multi_processor_count == 4``,
# which starves the persistent kernel (measured 8x on the core shapes).
_GCU_NUM_GRID = 48


# Autotuned fp8 tiles for the even-K persistent kernel.  On the Triton GCU
# 4.1.6 stack the fp8 ``tl.dot`` (lowered to tops_matmul_fp8e4m3fn) prefers
# deep K tiles and num_warps=2; nw4/nw8 leave ~20-30% on the table at
# BLOCK_K >= 128 (counter to the older c042-stack "fp8 nw2 is 2x slower"
# finding -- re-measured on this stack, do not carry the old note over).
# num_stages has no measurable effect on this backend (verified ns1..ns5).
def get_autotune_config():
    return [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 256},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 256},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 256},
            num_stages=3,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 128},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 256},
            num_stages=3,
            num_warps=4,
        ),
    ]


def _prune_configs(configs, named_args, **kwargs):
    """Keep only tiles whose K step divides K (the even kernel is launched
    with the no-tail contract) and whose N/M tiles are not wider than the
    operand itself.  The host routes K % 64 != 0 to the ragged kernel before
    the tuner runs, so the BLOCK_K=64 entry always survives."""
    K = named_args["K"]
    N = named_args["N"]
    M = named_args["M"]
    fitting = [
        config
        for config in configs
        if K % config.kwargs["BLOCK_K"] == 0
        and config.kwargs["BLOCK_N"] <= N
        and config.kwargs["BLOCK_M"] <= max(16, M)
    ]
    if not fitting:
        # Every tile is wider than the operand (e.g. tiny N): fall back to
        # any K-dividing config; the masked loads keep them correct.
        fitting = [config for config in configs if K % config.kwargs["BLOCK_K"] == 0]
    if not fitting:
        raise NotImplementedError(
            f"scaled_grouped_mm_v2: no even-K tile fits K={K}, N={N}, M={M}"
        )
    return fitting


@libentry()
@libtuner(
    configs=get_autotune_config(),
    key=["M", "N", "K", "MODE"],
    prune_configs_by={"early_config_prune": _prune_configs},
    warmup=2,
    rep=4,
)
@triton.jit
def _scaled_grouped_mm_v2_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Offs,
    C,
    M,
    N,
    K,
    NUM_GROUPS,
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
    MODE: tl.constexpr,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    WIDE_IDX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Persistent even-K grouped GEMM for GCU400.

    The host only routes calls here whose contraction ``K`` is a multiple of
    ``BLOCK_K`` (enforced by ``_prune_configs``) and whose mode is not
    ``k_varying``; everything else goes to ``_scaled_grouped_mm_v2_ragged_kernel``.
    Keeping every K-tail branch out of this kernel is a hard requirement on
    this backend: a runtime-dead tail ``if`` left next to the K loop degrades
    the whole main loop ~130x and doubles the DSM footprint (re-measured on
    the Triton GCU 4.1.6 stack; same pathology as the vllm cutlass_scaled_mm
    case).

    Tiles are enumerated globally -- all of group 0's tiles, then group 1's,
    ... -- and the ``_GCU_NUM_GRID`` programs stride over that flat
    enumeration, so a jagged group contributes exactly the tiles it owns.
    Loads carry single-dimension row/column masks only; a compound (row &
    K-tail) mask inside a K loop miscompiles fp8 dot on GCU (silent NaN in
    the last output column), which is why ragged K is a separate kernel.
    """
    tl.static_assert(MODE != _MODE_K_VARYING)

    if WIDE_IDX:
        idx_t = tl.int64
    else:
        idx_t = tl.int32

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    tile = pid
    iterated_tiles = 0
    offset_end = 0

    for group_idx in tl.range(NUM_GROUPS):
        m_start = 0
        n_start = 0
        k_start = 0
        m_size = M
        n_size = N
        k_size = K

        if MODE == _MODE_BATCH:
            scale_a_start = group_idx * stride_sag
            scale_b_start = group_idx * stride_sbg
        else:
            offset_start = offset_end
            offset_end = tl.load(Offs + group_idx)
            group_size = offset_end - offset_start
            if MODE == _MODE_M_VARYING:
                m_start = offset_start
                m_size = group_size
                scale_a_start = offset_start
                scale_b_start = group_idx * stride_sbg
            elif MODE == _MODE_N_VARYING:
                n_start = offset_start
                n_size = group_size
                scale_a_start = group_idx * stride_sag
                scale_b_start = offset_start
            else:
                k_start = offset_start
                k_size = group_size
                scale_a_start = group_idx * stride_sag
                scale_b_start = group_idx * stride_sbg

        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        group_end = iterated_tiles + num_m_tiles * num_n_tiles

        if tile < group_end:
            loop_count = (group_end - tile + num_programs - 1) // num_programs
            for _ in tl.range(loop_count):
                tile_in_group = tile - iterated_tiles
                pid_m = tile_in_group % num_m_tiles
                pid_n = tile_in_group // num_m_tiles

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                masks_m = offs_m < m_size
                masks_n = offs_n < n_size

                # 2D operands share one copy across groups; their group
                # stride is 0 and collapses here.  Index arithmetic stays
                # int32 unless the host flagged an overflow risk (WIDE_IDX),
                # in which case only the products with strides widen.
                a_ptrs = (
                    A
                    + group_idx.to(idx_t) * stride_ag
                    + (m_start + offs_m[:, None]).to(idx_t) * stride_am
                    + (k_start + offs_k[None, :]).to(idx_t) * stride_ak
                )
                b_ptrs = (
                    B
                    + group_idx.to(idx_t) * stride_bg
                    + (k_start + offs_k[:, None]).to(idx_t) * stride_bk
                    + (n_start + offs_n[None, :]).to(idx_t) * stride_bn
                )

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for _k in range(0, k_size, BLOCK_K):
                    a = tl.load(a_ptrs, mask=masks_m[:, None], other=0.0)
                    b = tl.load(b_ptrs, mask=masks_n[None, :], other=0.0)
                    acc += tl.dot(a, b, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk

                if SCALE_A_MODE == _SCALE_TENSOR_WISE:
                    scale_a = tl.load(ScaleA).to(tl.float32)
                else:
                    scale_a = tl.load(
                        ScaleA + scale_a_start.to(idx_t) + offs_m,
                        mask=masks_m,
                        other=0.0,
                    ).to(tl.float32)
                if SCALE_B_MODE == _SCALE_TENSOR_WISE:
                    scale_b = tl.load(ScaleB).to(tl.float32)
                else:
                    scale_b = tl.load(
                        ScaleB + scale_b_start.to(idx_t) + offs_n,
                        mask=masks_n,
                        other=0.0,
                    ).to(tl.float32)

                c = acc * scale_a[:, None] * scale_b[None, :]

                if MODE == _MODE_BATCH or MODE == _MODE_K_VARYING:
                    c_ptrs = (
                        C
                        + group_idx.to(idx_t) * stride_cg
                        + offs_m[:, None].to(idx_t) * stride_cm
                        + offs_n[None, :].to(idx_t) * stride_cn
                    )
                else:
                    c_ptrs = (
                        C
                        + (m_start + offs_m[:, None]).to(idx_t) * stride_cm
                        + (n_start + offs_n[None, :]).to(idx_t) * stride_cn
                    )
                tl.store(c_ptrs, c, mask=masks_m[:, None] & masks_n[None, :])

                tile += num_programs

        iterated_tiles = group_end


@triton.jit
def _scaled_grouped_mm_v2_ragged_kernel(
    A,
    B,
    ScaleA,
    ScaleB,
    Offs,
    C,
    M,
    N,
    K,
    NUM_GROUPS,
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
    MODE: tl.constexpr,
    SCALE_A_MODE: tl.constexpr,
    SCALE_B_MODE: tl.constexpr,
    WIDE_IDX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Correctness path for ragged contraction sizes.

    Used when the contraction extent is not a multiple of any tuned
    ``BLOCK_K`` (``m/n/batch`` modes) or always for ``k_varying``, where the
    per-group K comes from ``offs`` and cannot be verified on the host.  The
    even part of K walks a clean unmasked loop and a single masked iteration
    handles the tail *after* the loop; a compound mask inside the K loop
    miscompiles fp8 dot on GCU, and a tail branch inside the even kernel
    slows its main loop ~130x, so the tail lives only in this kernel.  The
    dot here converts the fp8 tiles to fp16 (exact for e4m3) before every
    contraction: fp8 ``tl.dot`` miscompiles on partial-K tiles on this stack
    regardless of where the tail predicate sits (see
    compiler_issues/flaggems_enflame_scaled_mm_v2_fp8_dot_ktail_bk_ceiling.md).
    """
    if WIDE_IDX:
        idx_t = tl.int64
    else:
        idx_t = tl.int32

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    tile = pid
    iterated_tiles = 0
    offset_end = 0

    for group_idx in tl.range(NUM_GROUPS):
        m_start = 0
        n_start = 0
        k_start = 0
        m_size = M
        n_size = N
        k_size = K

        if MODE == _MODE_BATCH:
            scale_a_start = group_idx * stride_sag
            scale_b_start = group_idx * stride_sbg
        else:
            offset_start = offset_end
            offset_end = tl.load(Offs + group_idx)
            group_size = offset_end - offset_start
            if MODE == _MODE_M_VARYING:
                m_start = offset_start
                m_size = group_size
                scale_a_start = offset_start
                scale_b_start = group_idx * stride_sbg
            elif MODE == _MODE_N_VARYING:
                n_start = offset_start
                n_size = group_size
                scale_a_start = group_idx * stride_sag
                scale_b_start = offset_start
            else:
                k_start = offset_start
                k_size = group_size
                scale_a_start = group_idx * stride_sag
                scale_b_start = group_idx * stride_sbg

        num_m_tiles = tl.cdiv(m_size, BLOCK_M)
        num_n_tiles = tl.cdiv(n_size, BLOCK_N)
        group_end = iterated_tiles + num_m_tiles * num_n_tiles

        if tile < group_end:
            loop_count = (group_end - tile + num_programs - 1) // num_programs
            for _ in tl.range(loop_count):
                tile_in_group = tile - iterated_tiles
                pid_m = tile_in_group % num_m_tiles
                pid_n = tile_in_group // num_m_tiles

                offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
                offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                masks_m = offs_m < m_size
                masks_n = offs_n < n_size

                a_ptrs = (
                    A
                    + group_idx.to(idx_t) * stride_ag
                    + (m_start + offs_m[:, None]).to(idx_t) * stride_am
                    + (k_start + offs_k[None, :]).to(idx_t) * stride_ak
                )
                b_ptrs = (
                    B
                    + group_idx.to(idx_t) * stride_bg
                    + (k_start + offs_k[:, None]).to(idx_t) * stride_bk
                    + (n_start + offs_n[None, :]).to(idx_t) * stride_bn
                )

                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                # fp8 dot miscompiles when any K-tail predicate reaches it on
                # this stack (see compiler_issues/
                # flaggems_enflame_scaled_mm_v2_fp8_dot_ktail_bk_ceiling.md):
                # the ragged path converts to fp16 -- exact for e4m3 -- so the
                # dot never sees partial-K operands at all.
                k_main = (k_size // BLOCK_K) * BLOCK_K
                for _k in range(0, k_main, BLOCK_K):
                    a = tl.load(a_ptrs, mask=masks_m[:, None], other=0.0)
                    b = tl.load(b_ptrs, mask=masks_n[None, :], other=0.0)
                    acc += tl.dot(
                        a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32
                    )
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
                if k_main < k_size:
                    k_mask = offs_k < (k_size - k_main)
                    a = tl.load(
                        a_ptrs,
                        mask=masks_m[:, None] & k_mask[None, :],
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=k_mask[:, None] & masks_n[None, :],
                        other=0.0,
                    )
                    acc += tl.dot(
                        a.to(tl.float16), b.to(tl.float16), out_dtype=tl.float32
                    )

                if SCALE_A_MODE == _SCALE_TENSOR_WISE:
                    scale_a = tl.load(ScaleA).to(tl.float32)
                else:
                    scale_a = tl.load(
                        ScaleA + scale_a_start.to(idx_t) + offs_m,
                        mask=masks_m,
                        other=0.0,
                    ).to(tl.float32)
                if SCALE_B_MODE == _SCALE_TENSOR_WISE:
                    scale_b = tl.load(ScaleB).to(tl.float32)
                else:
                    scale_b = tl.load(
                        ScaleB + scale_b_start.to(idx_t) + offs_n,
                        mask=masks_n,
                        other=0.0,
                    ).to(tl.float32)

                c = acc * scale_a[:, None] * scale_b[None, :]

                if MODE == _MODE_BATCH or MODE == _MODE_K_VARYING:
                    c_ptrs = (
                        C
                        + group_idx.to(idx_t) * stride_cg
                        + offs_m[:, None].to(idx_t) * stride_cm
                        + offs_n[None, :].to(idx_t) * stride_cn
                    )
                else:
                    c_ptrs = (
                        C
                        + (m_start + offs_m[:, None]).to(idx_t) * stride_cm
                        + (n_start + offs_n[None, :]).to(idx_t) * stride_cn
                    )
                tl.store(c_ptrs, c, mask=masks_m[:, None] & masks_n[None, :])

                tile += num_programs

        iterated_tiles = group_end


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


def _is_supported_input_dtype(dtype):
    return dtype in _float8_dtypes()


def _default_out_dtype(dtype):
    if _is_float8_dtype(dtype):
        return torch.bfloat16
    return dtype


def _parse_recipe(recipe, name):
    if recipe is None or len(recipe) == 0:
        return RECIPE_ROW_WISE
    if len(recipe) != 1:
        raise ValueError(
            f"{name} must describe exactly one scaling recipe, got {len(recipe)}"
        )
    value = int(recipe[0])
    if value not in (RECIPE_TENSOR_WISE, RECIPE_ROW_WISE):
        raise ValueError(f"unsupported scaling recipe {value} for {name}")
    return value


def _parse_swizzle(swizzle, name):
    if swizzle is None or len(swizzle) == 0:
        return SWIZZLE_NONE
    if len(swizzle) != 1:
        raise ValueError(
            f"{name} must describe exactly one swizzle pattern, got {len(swizzle)}"
        )
    value = int(swizzle[0])
    if value != SWIZZLE_NONE:
        raise ValueError(f"unsupported swizzle pattern {value} for {name}")
    return value


def _parse_scale_list(scales, name):
    if scales is None:
        raise ValueError(f"{name} must be a list of tensors")
    if len(scales) != 1:
        raise ValueError(f"{name} must contain exactly one tensor, got {len(scales)}")
    return scales[0]


def _resolve_mode(mat_a, mat_b, offs):
    a_is_2d = mat_a.dim() == 2
    b_is_2d = mat_b.dim() == 2
    if a_is_2d and b_is_2d:
        mode = MODE_K_VARYING
    elif a_is_2d:
        mode = MODE_M_VARYING
    elif b_is_2d:
        mode = MODE_N_VARYING
    else:
        mode = MODE_BATCH

    if a_is_2d or b_is_2d:
        if offs is None:
            raise ValueError("Have to provide offsets if there is a 2d matrix")
        if offs.dtype != torch.int32:
            raise ValueError(_OFFS_DTYPE_ERROR)
        if offs.dim() != 1:
            raise ValueError("Offsets have to be 1D")
    elif offs is not None:
        raise ValueError("offs must be None when both operands are 3D")

    return mode, a_is_2d, b_is_2d


def _resolve_shapes(mat_a, mat_b, offs, mode):
    if mode == MODE_M_VARYING:
        num_groups = mat_b.shape[0]
        m_size, k_size = mat_a.shape
        k_b, n_size = mat_b.shape[-2], mat_b.shape[-1]
        out_shape = (m_size, n_size)
    elif mode == MODE_N_VARYING:
        num_groups = mat_a.shape[0]
        _, m_size, k_size = mat_a.shape
        k_b, n_size = mat_b.shape
        out_shape = (m_size, n_size)
    elif mode == MODE_K_VARYING:
        num_groups = offs.numel()
        m_size, k_size = mat_a.shape
        k_b, n_size = mat_b.shape
        out_shape = (num_groups, m_size, n_size)
    else:
        num_groups = mat_a.shape[0]
        _, m_size, k_size = mat_a.shape
        groups_b, k_b, n_size = mat_b.shape
        if groups_b != num_groups:
            raise ValueError(
                f"matrix batch sizes have to match, got {num_groups} and {groups_b}"
            )
        out_shape = (num_groups, m_size, n_size)

    if k_size != k_b:
        raise ValueError(
            f"mat_a and mat_b shapes cannot be multiplied ({k_size} and {k_b})"
        )

    if mode != MODE_BATCH and offs.numel() != num_groups:
        raise ValueError(
            f"offs length must match the group count, "
            f"got {offs.numel()} and {num_groups}"
        )

    return num_groups, m_size, n_size, k_size, out_shape


def _prepare_scale(
    scale,
    *,
    recipe,
    row_len,
    num_groups,
    jagged_total,
    name,
):
    """Validate a decoding scale and flatten it into the layout the kernel reads.

    ``RowWise`` scales are 1D when they serve a 2D (jagged) operand and 2D
    ``(num_groups, row_len)`` when they serve a 3D one; ``k_varying`` hands
    both of them over flattened per group.  Either way the kernel indexes the
    result through a flat pointer plus a per-group start, so the tensor is
    collapsed to 1D here.
    """
    if recipe == RECIPE_TENSOR_WISE:
        if scale.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {scale.dtype}")
        if scale.numel() != 1:
            raise ValueError(f"{name} must hold one element, got {scale.numel()}")
        return scale.reshape(1).contiguous()

    if scale.dtype != torch.float32:
        raise ValueError(f"{name} must be float32, got {scale.dtype}")

    if jagged_total is not None:
        # Serves the operand that ``offs`` slices: one factor per jagged index.
        if scale.dim() != 1:
            raise ValueError(f"{name} must be a 1D tensor for a jagged operand")
        if scale.numel() != jagged_total:
            raise ValueError(
                f"{name} must have the same length as mat, "
                f"got {scale.numel()} instead of {jagged_total}"
            )
    elif scale.dim() == 2:
        if tuple(scale.shape) != (num_groups, row_len):
            raise ValueError(
                f"{name} must have shape {(num_groups, row_len)}, "
                f"got {tuple(scale.shape)}"
            )
    elif scale.numel() != num_groups * row_len:
        raise ValueError(
            f"{name} must have the same length as mat, "
            f"got {scale.numel()} instead of {num_groups * row_len}"
        )

    return scale.reshape(-1).contiguous()


def _scaled_grouped_mm_v2(
    mat_a,
    mat_b,
    scale_a,
    recipe_a,
    swizzle_a,
    scale_b,
    recipe_b,
    swizzle_b,
    offs=None,
    bias=None,
    out_dtype=None,
    contraction_dim=None,
    use_fast_accum=False,
):
    logger.debug("GEMS_ENFLAME _SCALED_GROUPED_MM_V2")
    if bias is not None:
        raise ValueError(_BIAS_ERROR)
    if contraction_dim:
        raise ValueError(
            "contraction_dim only applies to operands with more than 3 dimensions, "
            "which are not supported"
        )

    if mat_a.dtype != mat_b.dtype:
        raise ValueError(
            f"mat_a and mat_b must have the same dtype, "
            f"got {mat_a.dtype} and {mat_b.dtype}"
        )
    if not _is_supported_input_dtype(mat_a.dtype):
        raise ValueError(
            f"expected mat_a to be a float8_e4m3 tensor, got {mat_a.dtype}"
        )
    if mat_b.stride(-2) != 1:
        raise ValueError(_MAT_B_TRANSPOSED_ERROR)

    if out_dtype is None:
        out_dtype = _default_out_dtype(mat_a.dtype)
    if out_dtype not in (torch.bfloat16,):
        raise ValueError(_OUT_DTYPE_ERROR)

    recipe_a = _parse_recipe(recipe_a, "recipe_a")
    recipe_b = _parse_recipe(recipe_b, "recipe_b")
    _parse_swizzle(swizzle_a, "swizzle_a")
    _parse_swizzle(swizzle_b, "swizzle_b")

    scale_a = _parse_scale_list(scale_a, "scale_a")
    scale_b = _parse_scale_list(scale_b, "scale_b")

    mode, a_is_2d, b_is_2d = _resolve_mode(mat_a, mat_b, offs)
    num_groups, m_size, n_size, k_size, out_shape = _resolve_shapes(
        mat_a, mat_b, offs, mode
    )

    # Only ``m_varying``/``n_varying`` make a scale jagged; in ``k_varying``
    # both scales carry one row per group though the operands are 2D.
    scale_a = _prepare_scale(
        scale_a,
        recipe=recipe_a,
        row_len=m_size,
        num_groups=num_groups,
        jagged_total=m_size if mode == MODE_M_VARYING else None,
        name="scale_a",
    )
    scale_b = _prepare_scale(
        scale_b,
        recipe=recipe_b,
        row_len=n_size,
        num_groups=num_groups,
        jagged_total=n_size if mode == MODE_N_VARYING else None,
        name="scale_b",
    )

    # Per-group start of each flattened scale buffer.  A jagged scale advances
    # by the group's own offset in the kernel, so its stride is 0 here.
    if recipe_a == RECIPE_TENSOR_WISE or mode == MODE_M_VARYING:
        stride_sag = 0
    else:
        stride_sag = m_size
    if recipe_b == RECIPE_TENSOR_WISE or mode == MODE_N_VARYING:
        stride_sbg = 0
    else:
        stride_sbg = n_size

    stride_ag = 0 if a_is_2d else mat_a.stride(0)
    stride_am, stride_ak = mat_a.stride(-2), mat_a.stride(-1)
    stride_bg = 0 if b_is_2d else mat_b.stride(0)
    stride_bk, stride_bn = mat_b.stride(-2), mat_b.stride(-1)
    if mat_a.stride(-1) != 1:
        mat_a = mat_a.contiguous()
        stride_am, stride_ak = mat_a.stride(-2), mat_a.stride(-1)

    out = torch.empty(out_shape, dtype=out_dtype, device=mat_a.device)
    if out.numel() == 0:
        return out

    stride_cg = out.stride(0) if out.dim() == 3 else 0
    stride_cm, stride_cn = out.stride(-2), out.stride(-1)

    if offs is None:
        # ``Offs`` is part of the kernel signature even for the batched mode;
        # hand it a placeholder that is never dereferenced.
        offs = torch.zeros(1, dtype=torch.int32, device=mat_a.device)
    else:
        offs = offs.contiguous()

    # int32 addressing is safe only while every element offset stays below
    # 2**31; large expert stacks can push B's flat offset over the line, so
    # the kernels take an explicit width switch instead of risking wrap.
    wide_idx = (
        (stride_ag if not a_is_2d else 0) * num_groups + m_size * stride_am >= 2**31
        or (stride_bg if not b_is_2d else 0) * num_groups + n_size * stride_bn >= 2**31
        or stride_cg * num_groups + m_size * stride_cm + n_size * stride_cn >= 2**31
        or (stride_sag if stride_sag else 0) * num_groups + m_size >= 2**31
        or (stride_sbg if stride_sbg else 0) * num_groups + n_size >= 2**31
    )

    scale_a_mode = (
        SCALE_TENSOR_WISE if recipe_a == RECIPE_TENSOR_WISE else SCALE_ROW_WISE
    )
    scale_b_mode = (
        SCALE_TENSOR_WISE if recipe_b == RECIPE_TENSOR_WISE else SCALE_ROW_WISE
    )

    grid = (_GCU_NUM_GRID,)
    with torch_device_fn.device(mat_a.device):
        if mode == MODE_K_VARYING or k_size % 64 != 0:
            # Ragged contraction (or per-group K from ``offs``): correctness
            # path with fp16-converted dot; the tuned kernel must stay free
            # of any K-tail branch and of partial-K fp8 dot (miscompiles).
            _scaled_grouped_mm_v2_ragged_kernel[grid](
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                offs,
                out,
                m_size,
                n_size,
                k_size,
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
                MODE=mode,
                SCALE_A_MODE=scale_a_mode,
                SCALE_B_MODE=scale_b_mode,
                WIDE_IDX=wide_idx,
                BLOCK_M=64,
                BLOCK_N=128,
                BLOCK_K=64,
                num_warps=4,
                num_stages=3,
            )
        else:
            _scaled_grouped_mm_v2_kernel[grid](
                mat_a,
                mat_b,
                scale_a,
                scale_b,
                offs,
                out,
                m_size,
                n_size,
                k_size,
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
                MODE=mode,
                SCALE_A_MODE=scale_a_mode,
                SCALE_B_MODE=scale_b_mode,
                WIDE_IDX=wide_idx,
            )
    return out
