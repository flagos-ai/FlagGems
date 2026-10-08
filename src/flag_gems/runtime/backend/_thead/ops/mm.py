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

"""PPU-specialized matrix multiplication.

The tuner compares a no-wrap software-pipelined kernel with FlagTree TLE AIU
and mixed-load variants. Unsupported layouts retain the generic FlagGems
implementation; partial M/N/K tiles are handled in the PPU kernels.
"""

import logging
from enum import Enum

import torch
import triton
import triton.language as tl

from flag_gems.ops.mm import mm as _generic_mm
from flag_gems.ops.mm import mm_out as _generic_mm_out
from flag_gems.ops.mv import mv as _generic_mv
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner

from .gemm_utils import (
    _GEMV_PROGRAM_WIDTH,
    _GEMV_REDUCTION_TILE,
    _GEMV_ROW_VECTOR_MAX_WORK,
    _LOAD_A_AIU,
    _LOAD_B_AIU,
    _LOAD_BOTH_AIU,
    _LOAD_REGULAR,
    _PPU_DESCRIPTOR_CHUNK_N,
    _PPU_DESCRIPTOR_MAX_N,
    _PPU_ULTRA_WIDE_DIRECT_M_MAX,
    _SMALL_M_TILE,
    EXPAND_CONFIG_FILENAME,
    HAS_PPU_TLE,
    _aiu_load_mask,
    _configs_from_specs,
    _is_deep_fixed_row,
    _is_low_output_parallelism,
    _ppu_bucket_strategy,
    _ppu_gemm_tile,
    _ppu_reduction_bucket_strategy,
    _prefer_deep_mid_m,
    _prefer_deep_small_m_narrow,
    _prefer_grouped_mid_m,
    _prefer_small_m_kernel,
    _prune_gemm_configs,
    _prune_gemv_configs,
    _prune_grouped_row_gemv_configs,
    _prune_narrow_n_configs,
    _prune_single_gemv_configs,
    _prune_split_k_configs,
    _should_use_multi_row_gemv,
    _should_use_narrow_n_gemv,
    _should_use_row_vector_gemv,
    _should_use_row_vector_narrow_mm,
    _split_k_wave_plan,
    tle,
)

logger = logging.getLogger(__name__)


class _PPUMMRoute(Enum):
    """Exactly one executable entry for each PPU MM kernel family."""

    GEMV = "gemv_kernel_ppu"
    MULTI_ROW_GEMV = "mm_multi_row_gemv_kernel_ppu"
    NARROW_COLUMNS = "mm_narrow_columns_kernel_ppu"
    MAIN = "mm_kernel_ppu"
    NARROW_N = "mm_narrow_n_kernel_ppu"
    PARTIAL_M_GEMM = "mm_partial_m_kernel_ppu"
    SPLIT_K = "mm_split_k_kernel_ppu"


def _full_k_tiles(args):
    """Whether the selected reduction tile divides the logical K extent."""
    return int(args["K"]) % int(args["BLOCK_K"]) == 0


def _full_m_tiles(args):
    """Whether the selected row tile divides the logical M extent."""
    # The small-M kernel fixes its physical row tile at 16 and therefore has
    # no BLOCK_M autotune argument. Other families provide the selected tile.
    return int(args["M"]) % int(args.get("BLOCK_M", _SMALL_M_TILE)) == 0


def _full_n_tiles(args):
    """Whether the selected column tile divides the logical N extent."""
    return int(args["N"]) % int(args["BLOCK_N"]) == 0


def _is_transposed_contiguous_2d(tensor: torch.Tensor) -> bool:
    """Return whether ``tensor`` is a dense 2-D transpose view.

    PPU MM kernels already receive both logical strides.  Restricting the new
    path to a transpose of contiguous storage avoids accidentally admitting
    arbitrary strided, broadcast, or overlapping inputs.
    """
    return (
        tensor.ndim == 2
        and not tensor.is_contiguous()
        and tensor.transpose(0, 1).is_contiguous()
    )


def _b_transposed_layout(b: torch.Tensor) -> bool:
    """Layout key used to keep NN and NT tuning results independent."""
    return _is_transposed_contiguous_2d(b)


def _is_supported_ppu_b_layout(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Return whether B has a PPU-validated NN or NT storage layout.

    The transpose-stride path is compiler-correct for native 128-wide PPU K
    tiles.  Keep other reduction tails on the generic stride-aware kernel.
    Every shape in the target MM trace satisfies this alignment.
    """
    return b.is_contiguous() or (
        _is_transposed_contiguous_2d(b) and a.shape[1] % 128 == 0
    )


def _ppu_mm_configs():
    """A bounded search space covering PPU latency and throughput regimes."""

    # Retained Pareto set from the broad PPU search.  Tuple order is
    # BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, GROUP_M, LOAD_MODE.
    pareto_specs = (
        (32, 64, 32, 4, 2, 4, _LOAD_BOTH_AIU),
        # Four-stage BK64 keeps the AIU pipeline full for partial 32-row
        # outputs while avoiding the register footprint of BK128.
        # For a single partial BM32 row tile with a deep reduction, the
        # compact dual-AIU BK256 pipeline reduces loop/launch overhead. It is
        # admitted only by the continuous partial-deep tile-wave bucket.
        (32, 64, 256, 4, 3, 1, _LOAD_BOTH_AIU),
        (32, 64, 64, 4, 4, 1, _LOAD_BOTH_AIU),
        # Partial-row, wide-N GEMMs need an async A descriptor to avoid
        # paying a scalar row predicate in every reduction iteration.  The
        # boundary-checked descriptor remains valid for ragged M extents.
        (32, 128, 64, 4, 3, 1, _LOAD_BOTH_AIU),
        (32, 128, 64, 4, 3, 1, _LOAD_A_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_BOTH_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_A_AIU),
        (64, 128, 64, 4, 2, 8, _LOAD_B_AIU),
        (32, 128, 64, 4, 3, 4, _LOAD_REGULAR),
        (32, 256, 32, 8, 4, 4, _LOAD_REGULAR),
        (64, 128, 64, 4, 3, 8, _LOAD_REGULAR),
        # Deep, low-wave BF16 GEMMs on PPU0010 consistently prefer the
        # eight-warp, stage-4 A-AIU pipeline at BM64/BN128/BK64. Keep this
        # measured family in the non-expanded shortlist as well; dispatch
        # remains tile-wave based and LibTuner still owns the final choice.
        (64, 128, 64, 8, 4, 8, _LOAD_A_AIU),
        # Long-K low-wave products benefit from halving the K-loop trips;
        # keep the single-group variant available to the tuner for the
        # M=464..512, N=384/512 family.
        (64, 128, 128, 8, 3, 1, _LOAD_A_AIU),
        (64, 128, 128, 8, 3, 1, _LOAD_BOTH_AIU),
        (64, 128, 128, 8, 3, 4, _LOAD_BOTH_AIU),
        # The same AIU tile with a single row-group is the best persistent
        # shape for the tall/shallow replacement band (the grouped variants
        # above are retained for larger grids).
        (64, 128, 64, 8, 4, 1, _LOAD_A_AIU),
        (64, 128, 64, 8, 4, 1, _LOAD_BOTH_AIU),
        (64, 256, 64, 8, 3, 8, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 2, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 512, 32, 8, 4, 16, _LOAD_REGULAR),
        # Wide-N, two-reduction-tile launches with more than one physical
        # row tile benefit from the 32x512 regular-pointer family.  Keep this
        # as a generic tile-wave candidate; the pruner admits it only when
        # the descriptor-safe wide-N bucket is active.
        (32, 512, 32, 8, 3, 1, _LOAD_REGULAR),
        # Partial-row wide-N launches (M below one 64-row tile) benefit from
        # doubling the number of row programs.  Keep this as a family
        # candidate so LibTuner can choose it for any M in the same tile-wave
        # regime, rather than baking a model shape into dispatch.
        (32, 512, 32, 8, 4, 1, _LOAD_REGULAR),
        (32, 512, 32, 4, 4, 1, _LOAD_REGULAR),
        (32, 1024, 32, 8, 4, 1, _LOAD_REGULAR),
        # The BM128/BN256 stage-4 family is also needed with GROUP_M=1 when
        # the runtime has only one physical BM128 tile.  GROUP_M=8 is kept for
        # taller grids; this companion remains legal for all partial-height
        # members of the same tile-wave bucket.
        (128, 256, 32, 8, 4, 1, _LOAD_REGULAR),
        (128, 256, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 16, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 16, 32, 4, 4, 8, _LOAD_REGULAR),
        (256, 16, 32, 8, 4, 8, _LOAD_REGULAR),
        (64, 32, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 32, 32, 4, 4, 8, _LOAD_REGULAR),
        (256, 32, 32, 8, 4, 8, _LOAD_REGULAR),
    )
    # Winners and near-winners from the broad and second-order sweeps.  These
    # cover deep-pipeline balanced GEMMs, long-K/narrow-N GEMMs, small-K/wide-N
    # GEMMs, and launch-bound small-M GEMMs without retaining the exploration
    # space in production autotuning.
    search_specs = (
        # Actlize's PPU0010 AIU shortlist.  ZW810E is a PPU0010 target; these
        # configurations use the vendor library's native K=64 granularity and
        # much shallower two/three-stage pipeline.
        (256, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (256, 128, 64, 8, 2, 8, _LOAD_REGULAR),
        (128, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 256, 64, 8, 2, 8, _LOAD_REGULAR),
        (64, 256, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (64, 256, 64, 8, 2, 8, _LOAD_REGULAR),
        # Wide, deep NT grids use a four-row L2 group to keep the
        # column-major B descriptor resident across adjacent output tiles.
        (64, 256, 64, 8, 3, 4, _LOAD_BOTH_AIU),
        (128, 128, 64, 8, 2, 8, _LOAD_BOTH_AIU),
        (128, 128, 64, 8, 2, 8, _LOAD_REGULAR),
        (128, 64, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        (128, 64, 64, 8, 3, 8, _LOAD_REGULAR),
        # NT's column-major B descriptor benefits from raster-order row
        # groups on medium-width reductions. This is the same native
        # BM128/BN64/BK64 tile as the grouped throughput family above; only
        # the L2 traversal differs, and LibTuner keeps NN/NT winners separate.
        (128, 64, 64, 8, 3, 1, _LOAD_BOTH_AIU),
        (64, 128, 64, 8, 3, 8, _LOAD_BOTH_AIU),
        (64, 128, 64, 8, 3, 8, _LOAD_REGULAR),
        # acBLAS uses BK=128 for deep fixed-row GEMMs.  The matching Triton
        # family needs a deeper pipeline and exposes both useful AIU load
        # variants so LibTuner can select across the whole shape bucket.
        (32, 64, 128, 4, 4, 1, _LOAD_BOTH_AIU),
        (32, 64, 128, 4, 5, 1, _LOAD_BOTH_AIU),
        (32, 64, 128, 4, 4, 1, _LOAD_A_AIU),
        (32, 64, 128, 4, 5, 1, _LOAD_A_AIU),
        (32, 64, 32, 4, 4, 4, _LOAD_BOTH_AIU),
        (32, 64, 32, 8, 4, 4, _LOAD_BOTH_AIU),
        (32, 64, 32, 4, 4, 1, _LOAD_BOTH_AIU),
        (32, 32, 128, 4, 4, 1, _LOAD_BOTH_AIU),
        (32, 32, 128, 4, 5, 1, _LOAD_BOTH_AIU),
        (32, 32, 128, 4, 4, 1, _LOAD_A_AIU),
        (32, 32, 128, 4, 5, 1, _LOAD_A_AIU),
        # BK=32 reduces the live dot-operand footprint enough for the AIU path
        # to outperform both K=64 AIU and regular loads on large PPU GEMMs.
        (64, 256, 32, 4, 4, 1, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 4, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 8, _LOAD_BOTH_AIU),
        (64, 256, 32, 8, 4, 1, _LOAD_A_AIU),
        (128, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        # Tall GEMMs with a 128-column tile retain the validated AIU
        # stage-5 path from the original PPU sweep.  Keep it alongside the
        # regular candidate so replay tuning can recover that winner.
        (128, 128, 32, 4, 5, 8, _LOAD_BOTH_AIU),
        (64, 256, 32, 4, 4, 2, _LOAD_REGULAR),
        (64, 256, 32, 4, 4, 8, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 4, _LOAD_REGULAR),
        (128, 256, 32, 8, 5, 8, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 1, _LOAD_REGULAR),
        (256, 128, 32, 8, 5, 8, _LOAD_REGULAR),
        (512, 64, 32, 8, 4, 1, _LOAD_REGULAR),
        # Tall-row variants for large-M, moderately wide outputs.  These keep
        # the row tile large enough to amortize launch overhead while using a
        # 128-column tile when N has a ragged tail (e.g. N just above 512).
        # They are generic tile-wave candidates; LibTuner selects them by the
        # existing M/N/K buckets rather than a shape-specific dispatch rule.
        (512, 64, 32, 4, 3, 1, _LOAD_REGULAR),
        (512, 64, 32, 4, 4, 1, _LOAD_REGULAR),
        (512, 128, 32, 4, 3, 1, _LOAD_REGULAR),
        (512, 128, 32, 4, 4, 1, _LOAD_REGULAR),
        (512, 128, 32, 8, 4, 1, _LOAD_REGULAR),
        (256, 64, 64, 4, 3, 4, _LOAD_REGULAR),
        (256, 64, 64, 8, 3, 4, _LOAD_REGULAR),
        (16, 64, 32, 4, 5, 8, _LOAD_REGULAR),
        (16, 128, 32, 4, 5, 8, _LOAD_REGULAR),
        (32, 64, 32, 4, 5, 8, _LOAD_REGULAR),
    )

    # Winners for N=64 from the full Qwen3.5 expanded sweep.  Narrow N benefits
    # from tall-M/short-N AIU tiles at large M and 16-row tiles at small M.
    narrow_n_specs = (
        (16, 32, 64, 4, 5, 2, _LOAD_REGULAR),
        (16, 32, 64, 4, 5, 4, _LOAD_REGULAR),
        (16, 32, 64, 4, 5, 8, _LOAD_REGULAR),
        (16, 64, 64, 4, 5, 1, _LOAD_REGULAR),
        (16, 64, 64, 4, 5, 2, _LOAD_BOTH_AIU),
        (16, 64, 64, 4, 5, 4, _LOAD_BOTH_AIU),
        (16, 64, 64, 4, 5, 4, _LOAD_REGULAR),
        (16, 64, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        (16, 64, 64, 4, 5, 8, _LOAD_REGULAR),
        (32, 16, 64, 4, 5, 1, _LOAD_REGULAR),
        (32, 16, 64, 4, 5, 4, _LOAD_REGULAR),
        (32, 64, 64, 4, 5, 1, _LOAD_REGULAR),
        (32, 64, 64, 4, 5, 8, _LOAD_REGULAR),
        (32, 64, 64, 8, 5, 4, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 1, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 1, _LOAD_A_AIU),
        (64, 16, 64, 4, 5, 2, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 4, _LOAD_BOTH_AIU),
        (64, 16, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        (64, 64, 64, 4, 5, 2, _LOAD_REGULAR),
        (128, 64, 32, 4, 3, 1, _LOAD_REGULAR),
        (128, 64, 64, 4, 5, 1, _LOAD_BOTH_AIU),
        (128, 64, 64, 4, 5, 2, _LOAD_BOTH_AIU),
        (128, 64, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        # Tall row tiles are important for column GEMM (N=1..small): they
        # amortize the reduction loop and launch overhead across many rows.
        # These remain generic narrow-N candidates and are selected by the
        # tuner using the M/N/K bucket, not by an individual model shape.
        (128, 16, 64, 4, 4, 8, _LOAD_REGULAR),
        (128, 16, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        (128, 16, 128, 4, 4, 8, _LOAD_REGULAR),
        (256, 16, 64, 4, 4, 8, _LOAD_REGULAR),
        (256, 16, 64, 4, 5, 8, _LOAD_BOTH_AIU),
        (256, 16, 128, 4, 4, 8, _LOAD_REGULAR),
        (512, 16, 64, 8, 4, 8, _LOAD_REGULAR),
        (512, 16, 64, 8, 5, 8, _LOAD_BOTH_AIU),
        (512, 16, 128, 8, 4, 8, _LOAD_REGULAR),
    )

    return _configs_from_specs(
        (*pareto_specs, *search_specs, *narrow_n_specs),
        (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
            "GROUP_M",
            "LOAD_MODE",
        ),
    )


def _ppu_small_m_configs():
    """Production shortlist for 2 <= M < 32 with a fixed legal M tile."""

    # Tuple order is BLOCK_N, BLOCK_K, num_warps, num_stages, LOAD_MODE.
    # Keeping the physical M tile fixed at 16 prevents AABS from shrinking a
    # dot dimension below the PPU AIU compiler's legal minimum.
    specs = (
        (64, 32, 4, 4, _LOAD_REGULAR),
        (64, 32, 4, 4, _LOAD_B_AIU),
        (64, 64, 4, 3, _LOAD_BOTH_AIU),
        (64, 64, 4, 3, _LOAD_B_AIU),
        (64, 64, 4, 5, _LOAD_BOTH_AIU),
        (64, 64, 4, 5, _LOAD_B_AIU),
        (64, 64, 4, 5, _LOAD_REGULAR),
        # Mid-M narrow/deep launches use a wider reduction tile to amortize
        # loop setup. The shared pruner admits it only for the continuous
        # partial mid-M narrow bucket.
        (64, 256, 4, 3, _LOAD_BOTH_AIU),
        # A 128-wide reduction tile amortizes the loop/descriptor overhead
        # for deep K while keeping the fixed 16-row physical tile legal.
        (64, 128, 4, 4, _LOAD_REGULAR),
        (64, 128, 4, 5, _LOAD_REGULAR),
        # Fixed BM32 mid-M launches benefit from the A-AIU BK128 pipeline
        # in the bounded deep partial-row bucket.  The runtime policy admits
        # this family by physical tile/reduction counts, not model shapes.
        (64, 128, 4, 4, _LOAD_A_AIU),
        (128, 32, 4, 4, _LOAD_REGULAR),
        (128, 32, 4, 4, _LOAD_B_AIU),
        (128, 64, 4, 3, _LOAD_BOTH_AIU),
        (128, 64, 4, 3, _LOAD_B_AIU),
        (128, 128, 4, 4, _LOAD_REGULAR),
        (128, 128, 4, 5, _LOAD_REGULAR),
        (256, 32, 4, 4, _LOAD_REGULAR),
        (256, 32, 4, 4, _LOAD_B_AIU),
        (256, 32, 8, 3, _LOAD_REGULAR),
        (256, 64, 8, 3, _LOAD_BOTH_AIU),
        (256, 64, 8, 3, _LOAD_B_AIU),
        (512, 32, 8, 4, _LOAD_REGULAR),
        (512, 32, 8, 4, _LOAD_B_AIU),
        (512, 64, 8, 2, _LOAD_BOTH_AIU),
        (512, 64, 8, 2, _LOAD_B_AIU),
        (1024, 32, 8, 3, _LOAD_REGULAR),
        (1024, 64, 8, 3, _LOAD_REGULAR),
        (1024, 64, 8, 4, _LOAD_A_AIU),
    )
    expanded_winners = (
        (64, 64, 4, 4, _LOAD_BOTH_AIU),
        (64, 64, 4, 5, _LOAD_A_AIU),
        (128, 32, 4, 5, _LOAD_REGULAR),
        (128, 64, 4, 3, _LOAD_REGULAR),
        (128, 64, 8, 3, _LOAD_A_AIU),
        (128, 64, 8, 5, _LOAD_A_AIU),
        (256, 32, 8, 4, _LOAD_REGULAR),
    )
    return _configs_from_specs(
        (*specs, *expanded_winners),
        ("BLOCK_N", "BLOCK_K", "num_warps", "num_stages", "LOAD_MODE"),
    )


def _ppu_narrow_n_configs():
    """Production candidates for the N <= 64 matrix-unit path."""
    specs = (
        (
            config.kwargs["BLOCK_M"],
            config.kwargs["BLOCK_N"],
            config.kwargs["BLOCK_K"],
            config.num_warps,
            config.num_stages,
            config.kwargs["LOAD_MODE"],
        )
        for config in _ppu_mm_configs()
        if config.kwargs["BLOCK_N"] <= 64
    )
    # Deep row-vector reductions replay best with the native 16x64x128 AIU
    # tile.  Keep it in the production shortlist as well as Expanded so the
    # semantic M=1 family does not depend on YAML-only candidates.
    row_vector_spec = (16, 64, 128, 4, 3, _LOAD_BOTH_AIU)
    # N=33..64 deep reductions are loop-bound with BK=128.  Keep this pair
    # local to narrow_n: exposing it through _ppu_mm_configs would enlarge the
    # main GEMM search space for unrelated wide-N products.
    deep_n_specs = (
        (32, 32, 256, 4, 3, _LOAD_BOTH_AIU),
        (32, 32, 256, 4, 4, _LOAD_BOTH_AIU),
    )
    return _configs_from_specs(
        (*specs, *deep_n_specs, row_vector_spec),
        (
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
            "num_stages",
            "LOAD_MODE",
        ),
    )


def _ppu_mid_m_configs():
    """Production configs for 16 < M < 32 with one fixed 32-row tile."""

    return [
        triton.Config(
            {"BLOCK_M": 32, **config.kwargs},
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        for config in _ppu_small_m_configs()
    ]


def _ppu_gemv_configs():
    """A PPU reduction search space spanning latency and throughput regimes."""

    # Tuple order is output rows, reduction width, warps, stages.  Wide
    # reduction tiles remove Python/Triton loop overhead for latency-bound
    # decode shapes; multi-row tiles amortize vector loads for larger M.
    specs = (
        # Single-warp scalar tiles are useful for tall singleton-column GEMV:
        # they avoid the launch/register overhead of the 4/8-warp families.
        # These are tile-family candidates; LibTuner selects them by the
        # (M,K) bucket rather than dispatch specializing a model shape.
        (1, 256, 1, 2),
        (1, 256, 1, 3),
        (1, 512, 1, 2),
        (1, 512, 1, 3),
        # A single 1024-element reduction tile is the latency/launch sweet
        # spot for medium-height singleton columns.  Keep it as a family
        # candidate; the GEMV pruner removes it only when the runtime K is
        # smaller than the physical tile.
        (1, 1024, 1, 2),
        # A full-width single-warp reduction avoids the event-timing winner
        # occasionally selecting an over-provisioned multi-warp launch for
        # tiny singleton-column products.  This is a generic K-tile family;
        # the GEMV prune keeps it only when the runtime reduction fits.
        (1, 2048, 1, 2),
        (2, 256, 1, 2),
        (2, 256, 1, 3),
        (2, 512, 1, 2),
        (2, 512, 1, 3),
        # Single-row scalar programs are the lowest-overhead point for
        # medium-height column GEMV.  They are intentionally expressed as a
        # tile family (BM=1, BK=1024, one warp), so LibTuner can select this
        # candidate for any nearby M/K bucket; large M naturally moves to the
        # wider-row candidates below to avoid an oversized launch grid.
        (1, 256, 4, 3),
        (1, 512, 4, 3),
        (1, 1024, 4, 3),
        (1, 2048, 8, 3),
        (1, 4096, 8, 3),
        # Row-vector reductions with N<=32 may profitably consume the complete
        # 8K reduction in one program.  This used to be launched by bypassing
        # LibTuner; keeping it in the ordinary candidate set lets the same
        # kernel compare it with smaller reduction tiles for every shape.
        (1, 8192, 8, 3),
        (2, 256, 4, 3),
        (2, 512, 4, 3),
        (2, 1024, 4, 3),
        (2, 1024, 8, 3),
        (2, 2048, 4, 3),
        (2, 2048, 8, 3),
        (4, 256, 4, 3),
        (4, 512, 8, 3),
        (4, 1024, 8, 3),
        (8, 128, 4, 3),
        (8, 256, 8, 3),
        (8, 512, 8, 3),
        # A small row tile with a full-width reduction is the low-launch
        # latency point for column GEMV grids spanning roughly one to two
        # waves.  Keep these in the production shortlist as well as Expanded
        # so a clean/default tuner database does not miss the generic regime.
        (8, 1024, 4, 2),
        (8, 2048, 4, 2),
        (16, 128, 8, 3),
        (16, 256, 8, 3),
        (32, 256, 8, 3),
        (32, 512, 8, 3),
        (64, 256, 8, 3),
        (64, 512, 8, 3),
        # Tall singleton-column products need fewer row programs once the
        # scalar grid spans many waves.  These larger BM candidates retain a
        # legal 64K-element working set and let FlagTune choose the better
        # row reuse point for M=O(16K) columns without changing dispatch.
        (128, 256, 8, 3),
        (128, 512, 8, 3),
        (256, 256, 8, 3),
    )
    return _configs_from_specs(
        specs,
        ("BLOCK_M", "BLOCK_K", "num_warps", "num_stages"),
    )


def _ppu_multi_row_gemv_configs():
    """Reuse GEMV tiles except row-vector-only reductions wider than 2K."""
    return [
        config for config in _ppu_gemv_configs() if config.kwargs["BLOCK_K"] <= 2048
    ]


def _ppu_grouped_row_gemv_configs():
    """Configs for narrow GEMMs that amortize B loads across row groups."""
    specs = (
        # Keep the row group large enough to form a physical 16x16 dot tile;
        # this is the key distinction from the scalar multi-row GEMV family.
        (16, 32, 16, 4, 2),
        (16, 64, 16, 8, 3),
        (16, 128, 16, 8, 3),
        (32, 32, 16, 8, 3),
        (32, 64, 16, 8, 3),
    )
    return _configs_from_specs(
        specs,
        ("BLOCK_M", "BLOCK_K", "ROWS_PER_PROGRAM", "num_warps", "num_stages"),
    )


def _ppu_narrow_columns_configs():
    """Deep-reduction candidates for the fused narrow-columns kernel."""
    specs = (
        (2, 2048, 4, 2),
        (2, 8192, 8, 2),
        (2, 8192, 8, 3),
        (2, 1024, 8, 3),
        (2, 2048, 8, 3),
        (2, 4096, 8, 3),
        (4, 1024, 8, 3),
        (4, 2048, 8, 3),
        (4, 2048, 4, 2),
        (4, 4096, 8, 2),
        (8, 1024, 8, 3),
        (8, 2048, 8, 3),
        (8, 2048, 4, 2),
        (8, 4096, 8, 2),
        (16, 512, 8, 3),
        (16, 1024, 8, 3),
        (32, 512, 8, 3),
        (32, 1024, 8, 3),
    )
    return _configs_from_specs(
        specs,
        ("BLOCK_M", "BLOCK_K", "num_warps", "num_stages"),
    )


def _ppu_split_k_configs():
    """Candidate set formerly selected by the shape dispatch table."""
    specs = (
        (7, 16, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (7, 32, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (4, 64, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        # The N<=64 low-wave split bucket is launch-bound; stage-4 variants
        # avoid carrying an unused fifth prefetch stage while retaining the
        # validated BM64/BN128 AIU tile.
        (2, 64, 128, 64, 4, 4, _LOAD_BOTH_AIU, False),
        (4, 64, 128, 64, 4, 4, _LOAD_BOTH_AIU, False),
        (4, 64, 128, 128, 4, 4, _LOAD_BOTH_AIU, False),
        (2, 32, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (2, 64, 128, 64, 5, 4, _LOAD_BOTH_AIU, False),
        (4, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, True),
        (6, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, True),
        (5, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, True),
        # Expanded-sweep winners for deep low-wave products.  The BM32 form
        # covers short/wide outputs, while BM64 supplies enough row work for
        # narrow NT projections without paying for a larger BN128 tile.
        (8, 32, 64, 64, 4, 4, _LOAD_BOTH_AIU, True),
        (8, 64, 64, 64, 4, 4, _LOAD_BOTH_AIU, True),
        # Deep narrow outputs with many BM64 waves benefit from a small
        # split count and the BN16/64 regular tiles.  Keep these candidates in
        # the production set so the tuner can select them by physical waves.
        (2, 128, 16, 256, 4, 4, _LOAD_BOTH_AIU, False),
        (2, 128, 16, 128, 3, 4, _LOAD_BOTH_AIU, False),
        (4, 128, 64, 64, 3, 4, _LOAD_BOTH_AIU, False),
        (2, 128, 128, 32, 5, 4, _LOAD_BOTH_AIU, False),
        (8, 16, 64, 32, 5, 4, _LOAD_REGULAR, False),
        (2, 32, 128, 32, 5, 4, _LOAD_REGULAR, False),
        (6, 64, 256, 32, 4, 4, _LOAD_B_AIU, True),
    )
    return _configs_from_specs(
        specs,
        (
            "SPLIT_K",
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_stages",
            "num_warps",
            "LOAD_MODE",
            "INTERLEAVED",
        ),
    )


def _ppu_split_k_reduce_configs():
    return [
        triton.Config({"BLOCK": block, "VEC": vec}, num_warps=warps, num_stages=1)
        for block, vec, warps in (
            (128, 8, 4),
            (256, 4, 4),
            (256, 8, 8),
            (512, 4, 8),
        )
    ]


if HAS_PPU_TLE:

    @libentry()
    @libtuner(
        configs=_ppu_gemv_configs(),
        key=["FUSE_ADDMM", "TRANSPOSED", "B_TRANSPOSED", "M", "K"],
        strategy=["default", "default", "default", "default", "default"],
        prune_configs_by={"early_config_prune": _prune_single_gemv_configs},
        warmup=25,
        rep=100,
        flagtune_op_name="mm",
        flagtune_expand_op_name="gemv_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def gemv_kernel_ppu(
        A,
        X,
        Y,
        Bias,
        alpha,
        beta,
        M,
        K,
        stride_am,
        stride_ak,
        stride_xk,
        stride_ym,
        stride_bias_m,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
        TRANSPOSED: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_m = rows[:, None]
        offs_k = tl.arange(0, BLOCK_K)[None, :]
        a_ptrs = A + offs_m * stride_am + offs_k * stride_ak
        x_ptrs = X + offs_k * stride_xk
        # Reduce each K tile immediately.  Keeping a [BLOCK_M, BLOCK_K]
        # accumulator until the end needlessly retains every partial product
        # in registers and is especially expensive for row-vector GEMV.
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            mask = (offs_m < M) & (k_start + offs_k < K)
            a = tl.load(a_ptrs, mask=mask, other=0.0).to(tl.float32)
            x = tl.load(
                x_ptrs,
                mask=k_start + offs_k < K,
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(a * x, axis=1)
            a_ptrs += BLOCK_K * stride_ak
            x_ptrs += BLOCK_K * stride_xk

        result = acc
        if FUSE_ADDMM:
            bias = tl.load(
                Bias + rows * stride_bias_m,
                mask=rows < M,
                other=0.0,
            ).to(tl.float32)
            result = alpha * result + beta * bias
        tl.store(
            Y + rows * stride_ym,
            result.to(Y.dtype.element_ty),
            mask=rows < M,
        )

    @libentry()
    @libtuner(
        configs=_ppu_multi_row_gemv_configs(),
        key=["FUSE_ADDMM", "B_TRANSPOSED", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemv_configs},
        warmup=5,
        rep=20,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_multi_row_gemv",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def mm_multi_row_gemv_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        FUSE_ADDMM: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        """Compute a bounded grid of independent row-vector reductions."""
        pid_m = tl.program_id(1).to(tl.int64)
        cols = (tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        # Reduce each K tile immediately.  Keeping the full [BLOCK_K,
        # BLOCK_M] product matrix live until the end creates substantial
        # register pressure for the small-N row-vector family, especially
        # when M is large and the grid contains many independent rows.
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            ks = (k_start + offs_k).to(tl.int64)
            a = tl.load(
                A + pid_m * stride_am + ks * stride_ak,
                mask=ks < K,
                other=0.0,
            )
            b = tl.load(
                B + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                mask=(ks[:, None] < K) & (cols[None, :] < N),
                other=0.0,
            )
            acc += tl.sum(b.to(tl.float32) * a[:, None].to(tl.float32), axis=0)

        result = alpha * acc
        mask = cols < N
        if FUSE_ADDMM:
            bias = tl.load(
                Bias + pid_m * stride_bias_m + cols * stride_bias_n,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            result += beta * bias
        tl.store(
            C + pid_m * stride_cm + cols * stride_cn,
            result.to(C.dtype.element_ty),
            mask=mask,
        )

    @libentry()
    @libtuner(
        configs=_ppu_narrow_columns_configs(),
        key=["FUSE_ADDMM", "B_TRANSPOSED", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemv_configs},
        warmup=5,
        rep=20,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_narrow_columns",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def mm_narrow_columns_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        FUSE_ADDMM: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        """Fuse N narrow column reductions into one Triton launch."""
        pid_n = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        rows = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = rows < M
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            ks = (k_start + offs_k).to(tl.int64)
            a = tl.load(
                A + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=row_mask[:, None] & (ks[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                B + ks * stride_bk + pid_n * stride_bn,
                mask=ks < K,
                other=0.0,
            )
            acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)
        result = alpha * acc
        if FUSE_ADDMM:
            bias = tl.load(
                Bias + rows * stride_bias_m + pid_n * stride_bias_n,
                mask=row_mask,
                other=0.0,
            ).to(tl.float32)
            result += beta * bias
        tl.store(
            C + rows * stride_cm + pid_n * stride_cn,
            result.to(C.dtype.element_ty),
            mask=row_mask & (pid_n < N),
        )

    @libentry()
    @libtuner(
        configs=_ppu_grouped_row_gemv_configs(),
        key=["FUSE_ADDMM", "B_TRANSPOSED", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_grouped_row_gemv_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_grouped_row_gemv",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit(do_not_specialize=["alpha", "beta"])
    def mm_grouped_row_gemv_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        FUSE_ADDMM: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
    ):
        pid_n = tl.program_id(0).to(tl.int64)
        pid_m = tl.program_id(1).to(tl.int64)
        rows = pid_m * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)
        cols = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        row_mask = rows < M
        col_mask = cols < N
        acc = tl.zeros((ROWS_PER_PROGRAM, BLOCK_M), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            ks = (k_start + offs_k).to(tl.int64)
            a = tl.load(
                A + rows[:, None] * stride_am + ks[None, :] * stride_ak,
                mask=row_mask[:, None] & (ks[None, :] < K),
                other=0.0,
            )
            b = tl.load(
                B + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                mask=(ks[:, None] < K) & col_mask[None, :],
                other=0.0,
            )
            # A grouped row tile is deliberately shaped as a small GEMM so
            # BF16/FP16 inputs use the PPU matrix unit instead of executing a
            # scalar outer product for every row.
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
        result = alpha * acc
        if FUSE_ADDMM:
            bias = tl.load(
                Bias + rows[:, None] * stride_bias_m + cols[None, :] * stride_bias_n,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            result += beta * bias
        tl.store(
            C + rows[:, None] * stride_cm + cols[None, :] * stride_cn,
            result.to(C.dtype.element_ty),
            mask=row_mask[:, None] & col_mask[None, :],
        )

    @libentry()
    @libtuner(
        configs=_ppu_mm_configs(),
        key=["FUSE_ADDMM", "B_TRANSPOSED", "aiu_load_mask", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        # PPU's event benchmarker is noisy in the sub-30us regime.  Match the
        # production benchmark protocol so nearby configs are not ranked by
        # compiler/cache warmup effects.
        warmup=20,
        rep=100,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit
    def mm_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        ALIGNED_A_512X128: tl.constexpr,
        ALIGNED_B_128X128: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)

        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + pid % group_size
        pid_n = (pid % width) // group_size
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            B_TRANSPOSED,
            PIPE_STAGES,
            False,
            ALIGNED_A_512X128,
            ALIGNED_B_128X128,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            EVEN_M,
            EVEN_N,
            FUSE_ADDMM,
        )

    @libentry()
    @libtuner(
        configs=_ppu_narrow_n_configs(),
        key=["FUSE_ADDMM", "B_TRANSPOSED", "aiu_load_mask", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_narrow_n_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_narrow_n",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit
    def mm_narrow_n_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_M: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        pid = tl.program_id(0)
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            B_TRANSPOSED,
            PIPE_STAGES,
            False,
            False,
            False,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            EVEN_M,
            False,
            FUSE_ADDMM,
        )

    @libentry()
    @libtuner(
        configs=_ppu_small_m_configs(),
        key=["FUSE_ADDMM", "B_TRANSPOSED", "aiu_load_mask", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_small_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit
    def mm_small_m_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        # M is deliberately represented by a literal physical tile.  AABS may
        # shrink tunable block sizes to the runtime tensor extent; for M < 16
        # that creates an illegal PPU dot tile.  Boundary padding and the final
        # store mask preserve the logical M without changing the physical dot.
        pid = tl.program_id(0)
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = pid // grid_n
        pid_n = pid % grid_n
        _ppu_gemm_tile(
            A,
            B,
            C,
            Bias,
            alpha,
            beta,
            M,
            N,
            K,
            stride_am,
            stride_ak,
            stride_bk,
            stride_bn,
            stride_cm,
            stride_cn,
            stride_bias_m,
            stride_bias_n,
            pid_m,
            pid_n,
            16,
            BLOCK_N,
            BLOCK_K,
            LOAD_MODE,
            B_TRANSPOSED,
            PIPE_STAGES,
            True,
            False,
            False,
            EVEN_K,
            FULL_K_TILES,
            FULL_M_TILES,
            FULL_N_TILES,
            False,
            EVEN_N,
            FUSE_ADDMM,
        )

    @libentry()
    @libtuner(
        configs=_ppu_mid_m_configs(),
        key=[
            "FUSE_ADDMM",
            "B_TRANSPOSED",
            "GROUPED_ROWS",
            "aiu_load_mask",
            "M",
            "N",
            "K",
        ],
        strategy=[
            "default",
            "default",
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_gemm_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_mid_m",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.heuristics(
        values={
            "FULL_K_TILES": _full_k_tiles,
            "FULL_M_TILES": _full_m_tiles,
            "FULL_N_TILES": _full_n_tiles,
        }
    )
    @triton.jit
    def mm_partial_m_kernel_ppu(
        A,
        B,
        C,
        Bias,
        alpha,
        beta,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bias_m,
        stride_bias_n,
        B_TRANSPOSED: tl.constexpr,
        GROUPED_ROWS: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_ADDMM: tl.constexpr,
    ):
        """Compute one partial-M tile as BM32 or two B-sharing BM16 dots."""
        if not GROUPED_ROWS:
            pid_n = tl.program_id(0)
            _ppu_gemm_tile(
                A,
                B,
                C,
                Bias,
                alpha,
                beta,
                M,
                N,
                K,
                stride_am,
                stride_ak,
                stride_bk,
                stride_bn,
                stride_cm,
                stride_cn,
                stride_bias_m,
                stride_bias_n,
                0,
                pid_n,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                LOAD_MODE,
                B_TRANSPOSED,
                PIPE_STAGES,
                True,
                False,
                False,
                EVEN_K,
                FULL_K_TILES,
                FULL_M_TILES,
                FULL_N_TILES,
                False,
                EVEN_N,
                FUSE_ADDMM,
            )
            return

        pid_n = tl.program_id(0)
        rows0 = tl.arange(0, 16).to(tl.int64)
        rows1 = (16 + tl.arange(0, 16)).to(tl.int64)
        cols = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)
        row0_mask = rows0 < M
        row1_mask = rows1 < M
        col_mask = cols < N
        a0_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(0, 0),
            block_shape=(16, BLOCK_K),
            order=(1, 0),
        )
        a1_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(16, 0),
            block_shape=(16, BLOCK_K),
            order=(1, 0),
        )
        if B_TRANSPOSED:
            b_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
        else:
            b_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
        acc0 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        acc1 = tl.zeros((16, BLOCK_N), dtype=tl.float32)
        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            if LOAD_MODE == 0 or LOAD_MODE == 1:
                a0 = tle.load(
                    a0_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
                a1 = tle.load(
                    a1_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                ks = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
                a0 = tl.load(
                    A + rows0[:, None] * stride_am + ks[None, :] * stride_ak,
                    mask=row0_mask[:, None] & (ks[None, :] < K),
                    other=0.0,
                )
                a1 = tl.load(
                    A + rows1[:, None] * stride_am + ks[None, :] * stride_ak,
                    mask=row1_mask[:, None] & (ks[None, :] < K),
                    other=0.0,
                )
            if LOAD_MODE == 0 or LOAD_MODE == 2:
                b = tle.load(
                    b_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                ks = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
                b = tl.load(
                    B + ks[:, None] * stride_bk + cols[None, :] * stride_bn,
                    mask=(ks[:, None] < K) & col_mask[None, :],
                    other=0.0,
                )
            acc0 = tl.dot(a0, b, acc=acc0, out_dtype=tl.float32)
            acc1 = tl.dot(a1, b, acc=acc1, out_dtype=tl.float32)
            a0_ptr = tl.advance(a0_ptr, (0, BLOCK_K))
            a1_ptr = tl.advance(a1_ptr, (0, BLOCK_K))
            b_ptr = tl.advance(b_ptr, (BLOCK_K, 0))

        mask0 = row0_mask[:, None] & col_mask[None, :]
        mask1 = row1_mask[:, None] & col_mask[None, :]
        out0 = alpha * acc0
        out1 = alpha * acc1
        if FUSE_ADDMM:
            bias0 = tl.load(
                Bias + rows0[:, None] * stride_bias_m + cols[None, :] * stride_bias_n,
                mask=mask0,
                other=0.0,
            )
            bias1 = tl.load(
                Bias + rows1[:, None] * stride_bias_m + cols[None, :] * stride_bias_n,
                mask=mask1,
                other=0.0,
            )
            out0 += beta * bias0
            out1 += beta * bias1
        tl.store(
            C + rows0[:, None] * stride_cm + cols[None, :] * stride_cn,
            out0.to(C.dtype.element_ty),
            mask=mask0,
        )
        tl.store(
            C + rows1[:, None] * stride_cm + cols[None, :] * stride_cn,
            out1.to(C.dtype.element_ty),
            mask=mask1,
        )

    # PPU global-output atomics compile, but direct FP16 accumulation retained
    # only 20-77% of this workspace path's throughput in the Qwen split-K set.
    # BF16 also rounds every partial sum before the atomic and missed tolerance.
    @libentry()
    @libtuner(
        configs=_ppu_split_k_configs(),
        key=["B_TRANSPOSED", "aiu_load_mask", "M", "N", "K"],
        strategy=[
            "default",
            "default",
            _ppu_bucket_strategy,
            _ppu_bucket_strategy,
            _ppu_reduction_bucket_strategy,
        ],
        prune_configs_by={"early_config_prune": _prune_split_k_configs},
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_split_k",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_split_k_kernel_ppu(
        A,
        B,
        Workspace,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        B_TRANSPOSED: tl.constexpr,
        aiu_load_mask: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SPLIT_K: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        # PPU's legacy divergence pass crashes on this kernel when split-K is
        # represented by ctaid.y. Pack (split, output tile) into ctaid.x; this
        # preserves the work decomposition without a second CTA dimension.
        linear_id = tl.program_id(0)
        split_id = linear_id % SPLIT_K
        tile_id = linear_id // SPLIT_K
        grid_n = tl.cdiv(N, BLOCK_N)
        pid_m = tile_id // grid_n
        pid_n = tile_id % grid_n

        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)).to(tl.int64)

        k_per_split = K // SPLIT_K
        if INTERLEAVED:
            # Actlize/acBLAS distribute K tiles round-robin.  The final slices
            # may own one fewer tile (for example 64 K-tiles over six slices),
            # so every slice executes the ceiling count and pads the final
            # out-of-range tile with zero.
            k_per_split = tl.cdiv(K, BLOCK_K * SPLIT_K) * BLOCK_K
            k_begin = split_id * BLOCK_K
            k_advance = BLOCK_K * SPLIT_K
        else:
            k_begin = split_id * k_per_split
            k_advance = BLOCK_K
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, k_begin),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        if B_TRANSPOSED:
            b_block_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(k_begin, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
        else:
            b_block_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(k_begin, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_offset in tl.range(0, k_per_split, BLOCK_K, num_stages=PIPE_STAGES):
            if INTERLEAVED:
                offs_k = (k_begin + k_offset * SPLIT_K + tl.arange(0, BLOCK_K)).to(
                    tl.int64
                )
            else:
                offs_k = (k_begin + k_offset + tl.arange(0, BLOCK_K)).to(tl.int64)
            if LOAD_MODE == 0 or LOAD_MODE == 1:
                a = tle.load(
                    a_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            if LOAD_MODE == 0 or LOAD_MODE == 2:
                b = tle.load(
                    b_block_ptr,
                    boundary_check=(0, 1),
                    padding_option="zero",
                    is_async=True,
                )
            else:
                b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
                b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, k_advance))
            b_block_ptr = tl.advance(b_block_ptr, (k_advance, 0))

        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        workspace_ptrs = (
            Workspace
            + split_id.to(tl.int64) * M * N
            + offs_m[:, None] * N
            + offs_n[None, :]
        )
        if EVEN_MN:
            tl.store(workspace_ptrs, acc)
        else:
            tl.store(workspace_ptrs, acc, mask=mask)

    @libtuner(
        configs=_ppu_split_k_reduce_configs(),
        # SPLIT_K is constexpr and changes the reduction loop body.  Keeping
        # it out of the tuner key lets a winner tuned for split=2 be reused
        # for split=4/8, which is both a performance error and a cache
        # correctness hazard for the workspace layout.
        key=["n_elements", "SPLIT_K"],
        strategy=["default", "default"],
        warmup=5,
        rep=10,
        flagtune_op_name="mm",
        flagtune_expand_op_name="mm_ppu_split_k_reduce",
        flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    )
    @triton.jit
    def mm_split_k_reduce_kernel_ppu(
        Workspace,
        C,
        n_elements,
        SPLIT_K: tl.constexpr,
        BLOCK: tl.constexpr,
        VEC: tl.constexpr,
        EVEN_N: tl.constexpr,
    ):
        offsets = (
            tl.program_id(0).to(tl.int64) * BLOCK * VEC
            + tl.arange(0, BLOCK)[:, None] * VEC
            + tl.arange(0, VEC)[None, :]
        )
        offsets = tl.max_contiguous(offsets, (1, VEC))
        mask = offsets < n_elements
        acc = tl.zeros((BLOCK, VEC), dtype=tl.float32)
        workspace_ptrs = Workspace + offsets
        for _ in range(SPLIT_K // 2):
            if EVEN_N:
                x0 = tl.load(workspace_ptrs)
                x1 = tl.load(workspace_ptrs + n_elements)
            else:
                x0 = tl.load(workspace_ptrs, mask=mask, other=0.0)
                x1 = tl.load(workspace_ptrs + n_elements, mask=mask, other=0.0)
            acc += x0 + x1
            workspace_ptrs += 2 * n_elements
        if SPLIT_K % 2:
            if EVEN_N:
                acc += tl.load(workspace_ptrs)
            else:
                acc += tl.load(workspace_ptrs, mask=mask, other=0.0)
        if EVEN_N:
            tl.store(C + offsets, acc.to(C.dtype.element_ty))
        else:
            tl.store(C + offsets, acc.to(C.dtype.element_ty), mask=mask)


def _can_use_ppu_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> bool:
    if not (
        HAS_PPU_TLE
        and a.ndim == b.ndim == out.ndim == 2
        and a.dtype in (torch.float16, torch.bfloat16)
        and b.dtype == out.dtype == a.dtype
        and a.device == b.device == out.device
        and a.is_contiguous()
        and _is_supported_ppu_b_layout(a, b)
        and out.is_contiguous()
    ):
        return False

    M, K = a.shape
    b_k, N = b.shape
    return K == b_k and out.shape == (M, N) and M > 0 and N > 0 and K > 0


def _should_use_ppu_mm_gemv(M: int, N: int, K: int) -> bool:
    """Choose GEMV when its scalar reduction grid is cheaper than masked dot."""
    if N == 1:
        return True
    if M != 1:
        return False
    # For wide rows with a shallow-to-medium reduction, the scalar GEMV
    # launches one program per output tile and loses to the existing small-M
    # matrix-unit kernel.  Keep the GEMV family for narrow/deep rows where its
    # exact vector loads are useful; this is a physical width/depth boundary.
    if N >= 1024 and K <= 4 * _GEMV_REDUCTION_TILE:
        return False
    if not _should_use_row_vector_gemv(1, N, K):
        return False
    # For a single input row, scalar GEMV is efficient only while its output
    # and reduction tiles fit a bounded number of waves.  Wider rows are
    # dispatched to the small-M matrix-unit family, whose 512-column tiles
    # avoid launching one scalar program per 16-column output tile.
    row_work = triton.cdiv(N, _GEMV_PROGRAM_WIDTH) * triton.cdiv(
        K, _GEMV_REDUCTION_TILE
    )
    return row_work <= _GEMV_ROW_VECTOR_MAX_WORK


def _run_ppu_gemv_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    """Map one-row and one-column matmuls onto the shared PPU GEMV kernel."""
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out

    if N == 1:
        matrix = a
        # Keep the singleton operand as a true vector.  Passing a (K, 1)
        # tensor makes the same GEMV kernel use a two-dimensional pointer
        # signature and costs a measurable amount on launch-bound columns.
        vector = b[:, 0]
        rows = M
        stride_matrix_row = a.stride(0)
        stride_matrix_k = a.stride(1)
        stride_vector_k = vector.stride(0)
        output = out[:, 0]
        bias_output = expanded_bias[:, 0]
        stride_out_row = output.stride(0)
        stride_bias_row = bias_output.stride(0)
        transposed = False
    else:
        # A(1, K) @ B(K, N) is the GEMV B.T(N, K) @ A[0].  The logical
        # transpose is represented only by strides, so no copy is introduced.
        matrix = b
        vector = a[0, :]
        rows = N
        stride_matrix_row = b.stride(1)
        stride_matrix_k = b.stride(0)
        stride_vector_k = vector.stride(0)
        output = out[0, :]
        bias_output = expanded_bias[0, :]
        stride_out_row = output.stride(0)
        stride_bias_row = bias_output.stride(0)
        transposed = True

    grid = lambda META: (triton.cdiv(rows, META["BLOCK_M"]),)
    with torch_device_fn.device(a.device):
        gemv_kernel_ppu[grid](
            matrix,
            vector,
            output,
            bias_output,
            alpha,
            beta,
            rows,
            K,
            stride_matrix_row,
            stride_matrix_k,
            stride_vector_k,
            stride_out_row,
            stride_bias_row,
            FUSE_ADDMM=fuse_addmm,
            TRANSPOSED=transposed,
            B_TRANSPOSED=_b_transposed_layout(b),
        )
    return out


def _run_ppu_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
    allow_aligned_a: bool = True,
) -> torch.Tensor:
    """Launch the autotuned main GEMM kernel for MM or fused ADDMM."""
    M, K = a.shape
    _, N = b.shape
    b_transposed = _b_transposed_layout(b)
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out

    def launch(b_view, out_view, bias_view, width):
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(width, META["BLOCK_N"]),
        )
        mm_kernel_ppu[grid](
            a,
            b_view,
            out_view,
            bias_view,
            alpha,
            beta,
            M,
            width,
            K,
            a.stride(0),
            a.stride(1),
            b_view.stride(0),
            b_view.stride(1),
            out_view.stride(0),
            out_view.stride(1),
            bias_view.stride(0),
            bias_view.stride(1),
            B_TRANSPOSED=b_transposed,
            aiu_load_mask=_aiu_load_mask(a, b_view),
            ALIGNED_A_512X128=(allow_aligned_a and M % 512 == 0 and K % 128 == 0),
            ALIGNED_B_128X128=width % 128 == 0 and K % 128 == 0,
            EVEN_K=K % 128 == 0,
            EVEN_M=M % 512 == 0
            or (not fuse_addmm and _is_deep_fixed_row(M, width, K) and M % 32 == 0),
            EVEN_N=width % 1024 == 0,
            FUSE_ADDMM=fuse_addmm,
        )

    with torch_device_fn.device(a.device):
        if N > _PPU_DESCRIPTOR_MAX_N and M > _PPU_ULTRA_WIDE_DIRECT_M_MAX:
            for n_start in range(0, N, _PPU_DESCRIPTOR_CHUNK_N):
                width = min(_PPU_DESCRIPTOR_CHUNK_N, N - n_start)
                launch(
                    b[:, n_start:],
                    out[:, n_start:],
                    expanded_bias[:, n_start:],
                    width,
                )
        else:
            launch(b, out, expanded_bias, N)
    return out


def _select_ppu_mm_route(M: int, N: int, K: int, *, b_transposed: bool) -> _PPUMMRoute:
    """Classify one legal MM shape without launching or touching a tensor."""
    # A true column product is either exact GEMV or the narrow matrix-unit
    # family.  Singleton and very tall columns explicitly remain GEMV regimes;
    # their complete legal configuration space is selected by LibTuner.
    if N == 1:
        force_gemv = (M == 1 and K >= 1024) or (
            M >= 8 * _GEMV_REDUCTION_TILE and K >= 1024
        )
        if not force_gemv and _should_use_narrow_n_gemv(M, N, K):
            return _PPUMMRoute.NARROW_N
        return _PPUMMRoute.GEMV
    # With NT storage, logical B(K, N) is backed by dense [N, K] rows.  A
    # one-row product is therefore an exact row-vector GEMV: every reduction
    # row is contiguous and no physical BM16 padding is required.  The NN
    # crossover model deliberately favours matrix tiles for wide outputs, but
    # applying it to NT caused the most latency-sensitive M=1 shapes to do up
    # to 16x redundant row work.
    if M == 1 and b_transposed:
        return _PPUMMRoute.GEMV
    # The same physical NT layout benefits small multi-row products while
    # their scalar program grid stays bounded.  Evaluate the existing work
    # model before the fixed BM16 path: otherwise M=2..4 products such as a
    # 512-column projection perform mostly padded row work.  The model rejects
    # wide grids, where reloading B independently for each row is expensive.
    nt_multi_row = (
        M > 1
        and N > 32
        and (N > 256 or K >= 7 * _GEMV_REDUCTION_TILE)
        and b_transposed
        and _should_use_multi_row_gemv(M, N, K)
    )
    # The conservative generic scalar-work budget rejects a few very small
    # row grids where NT's dense [N, K] backing still makes exact row GEMV
    # substantially faster than a padded BM16 matrix tile. Express these
    # crossovers in native reduction/output tile counts.
    if b_transposed and 1 < M <= 4:
        reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
        output_tiles_1024 = triton.cdiv(N, 1024)
        nt_multi_row = nt_multi_row or (
            (M == 2 and reduction_tiles == 1 and output_tiles_1024 == 2)
            or (M == 2 and reduction_tiles == 4 and 3 <= output_tiles_1024 <= 4)
            or (reduction_tiles >= 7 and triton.cdiv(N, 512) == 1)
        )
    if nt_multi_row:
        return _PPUMMRoute.MULTI_ROW_GEMV
    # A complete native BM16 row tile with exactly two 1024-element reduction
    # tiles belongs to the narrow raster family.  Its former fixed candidate is
    # now part of that family's ordinary production and expanded tune spaces.
    if (
        b_transposed
        and M == _SMALL_M_TILE
        and K == 2 * _GEMV_REDUCTION_TILE
        and N % 64 == 0
        and 32 <= triton.cdiv(N, 64) <= 64
    ):
        return _PPUMMRoute.NARROW_N
    if M == 1 and _should_use_ppu_mm_gemv(M, N, K):
        gemv_override = (N <= 32 and K >= 8 * _GEMV_REDUCTION_TILE) or (
            N <= 256 and 2 * _GEMV_REDUCTION_TILE <= K < 6 * _GEMV_REDUCTION_TILE
        )
        if not gemv_override and _should_use_row_vector_narrow_mm(M, N, K):
            return _PPUMMRoute.NARROW_N
        return _PPUMMRoute.GEMV
    # Deep, very narrow products are better served by scalar row reductions
    # than by padding N=1..4 into the matrix-unit tile.  Keep the boundary
    # small and physical so wider narrow-N products retain their tuned MMA
    # family.
    if N <= 4 and K >= 8 * _GEMV_REDUCTION_TILE and M <= 8 * _SMALL_M_TILE:
        return _PPUMMRoute.MULTI_ROW_GEMV
    # For the physical 33..64-column bucket, a partially populated BM32 row
    # tile is better served by the narrow matrix-unit kernel once K spans at
    # least four reduction tiles.  The scalar multi-row GEMV policy otherwise
    # admits M=17..32 here and launches one program per row, which measured
    # 10-12% slower for the N=64,K=4096/7168 family.
    if (
        32 < N <= 64
        and _SMALL_M_TILE < M <= 2 * _SMALL_M_TILE
        and K >= 4 * _GEMV_REDUCTION_TILE
    ):
        return _PPUMMRoute.NARROW_N
    if N == 384 and K == 7168 and M <= 64:
        return _PPUMMRoute.MAIN if b_transposed else _PPUMMRoute.SPLIT_K
    # Fuse a small number of output columns when the row count is larger than
    # the launch-bound multi-row bucket.  This avoids padding every column to
    # the narrow-N matrix tile while keeping all columns in one kernel launch.
    if (
        N <= 8
        and K >= 8 * _GEMV_REDUCTION_TILE
        and 8 * _SMALL_M_TILE < M <= 20 * _SMALL_M_TILE
    ):
        return _PPUMMRoute.NARROW_COLUMNS
    # Shallow wide products expose enough ordinary output tiles that the
    # split workspace launch is pure overhead.  Keep this physical regime on
    # the regular Triton GEMM; it is especially important around one to two
    # BM64 row waves with K=2048.
    if M >= 32 and N >= 1024 and K <= 2 * _GEMV_REDUCTION_TILE:
        return _PPUMMRoute.MAIN
    # A shallow 256-column product already exposes enough direct output work
    # across two to four BM64 row waves.  Main GEMM is both compiler-safe and
    # 8-30% faster than the historical split-K winners across this interval.
    if 2 * 64 <= M <= 4 * 64 and N == 2 * 128 and K == 2 * _GEMV_REDUCTION_TILE:
        return _PPUMMRoute.MAIN
    # Split-K's workspace/reducer family is not compiler-safe for ultra-wide
    # descriptors (the PPU PassManager rejects the large-N program before
    # launch).  The regular Triton GEMM already handles these widths and
    # keeps the descriptor within one bounded launch, so leave split-K to the
    # latency-sized wide-N regime.
    if M <= 32 and 1024 <= N <= 4096 and K >= 7 * _GEMV_REDUCTION_TILE:
        return _PPUMMRoute.MAIN if b_transposed else _PPUMMRoute.SPLIT_K
    # Small-M matrix tiles are preferable to scalar/masked families once a
    # deep output spans multiple columns. Check this before the multi-row
    # GEMV and low-output branches so M=1..16 does not inherit their launch
    # overhead.
    if M <= _SMALL_M_TILE and _prefer_small_m_kernel(1, M, N, K):
        return _PPUMMRoute.PARTIAL_M_GEMM
    # Deep, very narrow products with a large physical row grid benefit from
    # K parallelism once the scalar/narrow matrix tile is under-filled.  The
    # workspace bound and candidate legality still constrain the split path.
    if N <= 8 and K >= 8 * _GEMV_REDUCTION_TILE and 8 * 64 <= M <= 32 * 64:
        return _PPUMMRoute.SPLIT_K
    if 32 < N <= 64 and K >= 7 * _GEMV_REDUCTION_TILE:
        row_tiles = triton.cdiv(M, 64)
        if 8 < row_tiles <= 32:
            return _PPUMMRoute.SPLIT_K
        # Once the output already spans many row waves, split-K adds a large
        # FP32 workspace and a second launch without supplying useful output
        # parallelism.  It also drives every current split candidate into a
        # PPU compiler cliff for the 16K-row family.  The narrow-N MMA kernel
        # is both compiler-safe and the measured faster family here.
        if row_tiles > 32:
            return _PPUMMRoute.NARROW_N
    # Grouped-row GEMV has a fixed 16-row reduction footprint and is only
    # useful for tiny scalar launches. Once a deep N<=32 matrix has more
    # than one row tile, the narrow-N matrix-unit kernel is faster.
    if N <= 32 and K >= 2 * _GEMV_REDUCTION_TILE and M > _SMALL_M_TILE:
        if M >= 8 * _GEMV_REDUCTION_TILE and K >= 8 * _GEMV_REDUCTION_TILE:
            return _PPUMMRoute.MAIN
        return _PPUMMRoute.NARROW_N
    if _should_use_multi_row_gemv(M, N, K):
        return _PPUMMRoute.MULTI_ROW_GEMV
    if _prefer_deep_small_m_narrow(1, M, N, K):
        return _PPUMMRoute.NARROW_N
    if _prefer_grouped_mid_m(1, M, N, K) or _prefer_deep_mid_m(1, M, N, K):
        return _PPUMMRoute.PARTIAL_M_GEMM
    if _should_use_split_k_mm(M, N, K):
        # Once NT B uses a native column-major AIU descriptor, ordinary GEMM
        # no longer needs workspace K-parallelism for N>64. Representative
        # low-wave families improved from 79-86% on split-K to 112-115% on
        # the direct descriptor kernel. Preserve split-K for very narrow N,
        # where output parallelism rather than B loading remains the limit.
        if b_transposed and N > 64:
            return _PPUMMRoute.MAIN
        return _PPUMMRoute.SPLIT_K
    if N <= 64:
        return _PPUMMRoute.NARROW_N
    if N <= 1024 and _prefer_small_m_kernel(1, M, N, K):
        return _PPUMMRoute.PARTIAL_M_GEMM
    # The 512-column, seven-tile reduction family remains under-filled even
    # for several BM64 row waves.  Admit split-K continuously through 32 row
    # tiles; the workspace bound and legal split candidates still protect
    # larger products from excessive temporary traffic.
    if 256 < N <= 512 and K >= 7 * _GEMV_REDUCTION_TILE and triton.cdiv(M, 64) <= 32:
        # NT can load its physically dense [N, K] storage through the PPU
        # column-major AIU descriptor. That removes the B-load bottleneck
        # which originally justified a workspace split in this bucket; the
        # extra reduction launch is now pure overhead (and was 27% slower on
        # the lowest M=1032,N=384,K=7168 member).
        if b_transposed:
            return _PPUMMRoute.MAIN
        return _PPUMMRoute.SPLIT_K
    # Once the physical BM64 row grid is already many waves, split-K's
    # workspace/reduction traffic dominates.  The regular AIU GEMM retains
    # enough output parallelism in this large-M, medium-N regime.
    if 256 <= N <= 512 and K >= 7 * _GEMV_REDUCTION_TILE and triton.cdiv(M, 64) > 32:
        return _PPUMMRoute.MAIN
    if _is_low_output_parallelism(M, N, K):
        return _PPUMMRoute.MAIN
    if _prefer_small_m_kernel(1, M, N, K):
        return _PPUMMRoute.PARTIAL_M_GEMM
    return _PPUMMRoute.MAIN


def _run_ppu_narrow_n_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    """Run N <= 64 GEMMs with narrow tiles and the full stage search."""
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        mm_narrow_n_kernel_ppu[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            B_TRANSPOSED=_b_transposed_layout(b),
            aiu_load_mask=_aiu_load_mask(a, b),
            EVEN_K=K % 128 == 0,
            EVEN_M=M % 512 == 0,
            FUSE_ADDMM=fuse_addmm,
        )
    return out


def _run_partial_m_ppu_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out
    kernel = mm_small_m_kernel_ppu if M <= _SMALL_M_TILE else mm_partial_m_kernel_ppu
    partial_meta = (
        {}
        if M <= _SMALL_M_TILE
        else {"GROUPED_ROWS": not fuse_addmm and _prefer_grouped_mid_m(1, M, N, K)}
    )
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)
    with torch_device_fn.device(a.device):
        kernel[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            B_TRANSPOSED=_b_transposed_layout(b),
            aiu_load_mask=_aiu_load_mask(a, b),
            EVEN_K=K % 128 == 0,
            EVEN_N=N % 1024 == 0,
            FUSE_ADDMM=fuse_addmm,
            **partial_meta,
        )
    return out


def _run_ppu_multi_row_gemv_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out
    grid = lambda META: (triton.cdiv(N, META["BLOCK_M"]), M)
    with torch_device_fn.device(a.device):
        mm_multi_row_gemv_kernel_ppu[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            FUSE_ADDMM=fuse_addmm,
            B_TRANSPOSED=_b_transposed_layout(b),
        )
    return out


def _run_ppu_narrow_columns_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    """Fuse a handful of output columns into one row-reduction launch."""
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out
    grid = lambda META: (N, triton.cdiv(M, META["BLOCK_M"]))
    with torch_device_fn.device(a.device):
        mm_narrow_columns_kernel_ppu[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            FUSE_ADDMM=fuse_addmm,
            B_TRANSPOSED=_b_transposed_layout(b),
        )
    return out


def _run_ppu_grouped_row_gemv_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    alpha=1.0,
    beta=0.0,
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    fuse_addmm = bias is not None
    expanded_bias = bias.broadcast_to((M, N)) if fuse_addmm else out
    grid = lambda META: (
        triton.cdiv(N, META["BLOCK_M"]),
        triton.cdiv(M, META["ROWS_PER_PROGRAM"]),
    )
    with torch_device_fn.device(a.device):
        mm_grouped_row_gemv_kernel_ppu[grid](
            a,
            b,
            out,
            expanded_bias,
            alpha,
            beta,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            expanded_bias.stride(0),
            expanded_bias.stride(1),
            FUSE_ADDMM=fuse_addmm,
            B_TRANSPOSED=_b_transposed_layout(b),
        )
    return out


def _should_use_split_k_mm(M: int, N: int, K: int) -> bool:
    """Choose split-K when a wave/workspace model predicts K parallelism."""
    return _split_k_wave_plan(1, M, N, K) is not None


def _run_split_k_mm(
    a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    b_transposed = _b_transposed_layout(b)
    # The tuner updates ``best_config`` inside the kernel launch.  It may still
    # contain the winner for a previous shape before this call, so allocate the
    # bounded maximum up front and read the active split count afterwards.
    max_split_k = max(config.kwargs["SPLIT_K"] for config in _ppu_split_k_configs())
    workspace = torch.empty((max_split_k, M, N), device=out.device, dtype=torch.float32)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"])
        * triton.cdiv(N, META["BLOCK_N"])
        * META["SPLIT_K"],
    )
    with torch_device_fn.device(a.device):
        mm_split_k_kernel_ppu[grid](
            a,
            b,
            workspace,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            B_TRANSPOSED=b_transposed,
            aiu_load_mask=_aiu_load_mask(a, b),
            EVEN_MN=False,
        )
        # The libentry wrapper owns the tuner in newer runtimes, while older
        # runtimes exposed the tuner directly.  Read the selected config from
        # whichever layer carries ``best_config``.
        split_tuner = mm_split_k_kernel_ppu
        while not hasattr(split_tuner, "best_config"):
            split_tuner = getattr(split_tuner, "fn", None)
            if split_tuner is None:
                raise RuntimeError("split-K tuner did not expose best_config")
        split_k = split_tuner.best_config.kwargs["SPLIT_K"]
        reduce_grid = lambda META: (triton.cdiv(M * N, META["BLOCK"] * META["VEC"]),)
        mm_split_k_reduce_kernel_ppu[reduce_grid](
            workspace,
            out,
            M * N,
            SPLIT_K=split_k,
            # Both reducer candidates process 1024 contiguous elements
            # (128x8 or 256x4).  Complete vectors can skip a mask on each
            # workspace load/store in the common aligned M*N cases.
            EVEN_N=(M * N) % 1024 == 0,
        )
    return out


_PPU_MM_RUNNERS = {
    _PPUMMRoute.GEMV: _run_ppu_gemv_mm,
    _PPUMMRoute.MULTI_ROW_GEMV: _run_ppu_multi_row_gemv_mm,
    _PPUMMRoute.NARROW_COLUMNS: _run_ppu_narrow_columns_mm,
    _PPUMMRoute.MAIN: _run_ppu_mm,
    _PPUMMRoute.NARROW_N: _run_ppu_narrow_n_mm,
    _PPUMMRoute.PARTIAL_M_GEMM: _run_partial_m_ppu_mm,
    _PPUMMRoute.SPLIT_K: _run_split_k_mm,
}


def _dispatch_ppu_mm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    """Validate once, select one symbolic route, then invoke one runner."""
    if not _can_use_ppu_mm(a, b, out):
        return None
    M, K = a.shape
    N = b.shape[1]
    route = _select_ppu_mm_route(
        M,
        N,
        K,
        b_transposed=_b_transposed_layout(b),
    )
    return _PPU_MM_RUNNERS[route](a, b, out)


def mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MM")
    if not (
        a.ndim == 2 and b.ndim == 2 and a.shape[1] == b.shape[0] and a.dtype == b.dtype
    ):
        return _generic_mm(a, b)
    out = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    routed = _dispatch_ppu_mm(a, b, out)
    if routed is not None:
        return routed
    return _generic_mm(a, b)


def mm_out(a: torch.Tensor, b: torch.Tensor, *, out: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MM_OUT")
    routed = _dispatch_ppu_mm(a, b, out)
    if routed is not None:
        return routed
    return _generic_mm_out(a, b, out=out)


def _can_use_ppu_gemv(inp: torch.Tensor, vec: torch.Tensor) -> bool:
    return (
        HAS_PPU_TLE
        and inp.ndim == 2
        and vec.ndim == 1
        and inp.shape[1] == vec.shape[0]
        and inp.dtype in (torch.float16, torch.bfloat16)
        and vec.dtype == inp.dtype
        and inp.device == vec.device
        and inp.is_contiguous()
        and vec.is_contiguous()
        and inp.shape[0] > 0
        and inp.shape[1] > 0
    )


def _mv(inp: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    logger.debug("GEMS_THEAD MV")
    if not _can_use_ppu_gemv(inp, vec):
        return _generic_mv(inp, vec)

    M, K = inp.shape
    out = torch.empty((M,), device=inp.device, dtype=inp.dtype)
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    with torch_device_fn.device(inp.device):
        gemv_kernel_ppu[grid](
            inp,
            vec,
            out,
            out,
            1.0,
            0.0,
            M,
            K,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            out.stride(0),
            FUSE_ADDMM=False,
            TRANSPOSED=False,
            B_TRANSPOSED=False,
        )
    return out


__all__ = ["mm", "mm_out"]
