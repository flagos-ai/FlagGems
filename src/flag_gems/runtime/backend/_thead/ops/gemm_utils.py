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

"""Dispatch, pruning, and shared kernel helpers for THead PPU MM."""

import math
import os

import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    HAS_PPU_TLE = True
except ImportError:
    tle = None
    HAS_PPU_TLE = False


_LOAD_BOTH_AIU = 0
_LOAD_A_AIU = 1
_LOAD_B_AIU = 2
_LOAD_REGULAR = 3
_AIU_LOAD_A = 1
_AIU_LOAD_B = 2
_SMALL_M_TILE = 16
_PPU_SMS = 64
_PPU_GEMM_FLOPS_PER_US = 64_000_000
_SPLIT_K_SETUP_US = 2
_SPLIT_K_MIN_REDUCTION_PER_SLICE = 1024
_PPU_MAX_SPLIT_K = 8
_GEMV_REDUCTION_TILE = 1024
_GEMV_PROGRAM_WIDTH = 16
_GEMV_SCALAR_TILE_BUDGET = _PPU_SMS * _PPU_SMS
# A row-vector GEMV becomes launch-bound once its scalar output/reduction
# tiles span more than eight PPU waves.  The bound is expressed in work, so
# the crossover moves continuously with N and K instead of matching model
# dimensions.
_GEMV_ROW_VECTOR_MAX_WORK = 8 * _PPU_SMS
# Column GEMV emits one scalar program per output-row tile.  Its work budget
# is intentionally smaller than the row-vector budget because a matrix-unit
# tile can reuse the singleton N dimension without materializing a vector
# launch for every row wave.
# Column GEMV has one output column and therefore little independent work.
# Keep a bounded, eight-wave-equivalent scalar budget before switching to the
# narrow-N matrix-unit family.  Expressing this as work rather than a model
# shape threshold lets the crossover move monotonically with M and K.
# Column GEMV needs a wider scalar work envelope than row-vector GEMV.  A
# 16-row tile is not competitive for singleton-N products, while BM=2/4 with
# BK=1024 remains efficient through several output waves.  Keep enough work
# for those tiles without admitting the BM=1 candidates once the row grid
# grows; the boundary is expressed in wave-equivalent work, not shape points.
_GEMV_COLUMN_TILE_BUDGET = 32 * _PPU_SMS
_MULTI_ROW_GEMV_PROGRAM_BUDGET = 2 * _PPU_SMS
_MULTI_ROW_GEMV_TILE_BUDGET = 8 * _PPU_SMS
_SINGLE_PROGRAM_ROW_BUDGET = 3 * _PPU_SMS
_SINGLE_PROGRAM_ROW_TILE_BUDGET = 32 * _PPU_SMS
# The descriptor field is 17-bit but the all-bits-set extent is reserved by
# the PPU lowering.  Keep chunk widths strictly below 2**17.
_PPU_DESCRIPTOR_MAX_N = (1 << 17) - 1
# Keep descriptor-safe chunks aligned to the native 128-column tile.  Using
# the raw 17-bit maximum (131071) makes every ultra-wide launch carry a large
# ragged tail and prevents the tuned even-N configurations from being used.
_PPU_DESCRIPTOR_CHUNK_N = (_PPU_DESCRIPTOR_MAX_N // 128) * 128
# A single physical 64-row tile can consume a full ultra-wide regular-pointer
# launch without losing output-wave occupancy.  Above this extent,
# descriptor-safe chunks retain the AIU/load family and avoid excessive
# accumulator footprints.  This is a physical tile boundary, not a
# model-shape lookup.  A full-N launch for two or more row tiles was measured
# slower than the descriptor-chunked path under the production CUDA-Graph
# protocol, so keep the validated one-tile cutoff.
_PPU_ULTRA_WIDE_DIRECT_M_MAX = 64
_PPU_ULTRA_WIDE_DIRECT_BLOCK_M = 64
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_ppu_expand.yaml")
)


def _ppu_bucket_strategy(value):
    """Coarsen PPU GEMM keys at tile-wave boundaries.

    This keeps nearby model trace shapes on one libtuner result while
    preserving the boundaries that change the number of output waves.  It is
    intentionally dtype-independent; dtype remains part of LibTuner's key.
    """
    value = int(value)
    if value <= 0:
        return value
    # Keep partially populated fixed row tiles separate from their full-tile
    # neighbours.  Collapsing M=17..31 with M=32 (or M=40..63 with M=64)
    # lets a winner measured without a row mask be reused for the partial-row
    # launch.  The affected decode projections are latency-sized, so these
    # small extra buckets are more valuable than a marginal cache reduction.
    if 16 < value < 32:
        return 24
    if value <= 32:
        return 1 << (value - 1).bit_length()
    if value < 64:
        return 48
    if value <= 256:
        quantum = 32
    elif value <= 1024:
        quantum = 256
    elif value <= 4096:
        quantum = 512
    else:
        quantum = 1024
    return int(math.ceil(value / quantum) * quantum)


def _ppu_reduction_bucket_strategy(value):
    """Bucket K by native reduction tiles without merging distinct depths."""
    value = int(value)
    if value <= 0:
        return value
    if value <= 256:
        quantum = 32
    elif value <= 4096:
        quantum = 128
    elif value <= 16384:
        quantum = 256
    else:
        quantum = 1024
    return int(math.ceil(value / quantum) * quantum)


def _is_deep_fixed_row(M: int, N: int, K: int) -> bool:
    """Return whether one BM32 row tile leaves fewer than one wave of work."""
    n_tiles = triton.cdiv(N, 64)
    return K >= 4096 and M > 16 and triton.cdiv(M, 32) == 1 and 8 <= n_tiles <= _PPU_SMS


def _latency_tile_width(N: int, K: int) -> int:
    """Return the output-tile width used by the low-wave work estimate.

    Wide, deep reductions are serviced by the main 128-column tile family;
    skinny or shallow reductions use the 16-column latency family.  Keeping
    this policy in one helper makes the unbatched and batched prune paths use
    the same shape-generic estimate.
    """
    reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
    # Two native K tiles are enough for the wide throughput family.  Requiring
    # four tiles made K=2048 shapes look artificially well-parallelized and
    # sent small-M/wide-N launches to the skinny family.
    return 128 if N >= 1024 and reduction_tiles >= 2 else 16


def _is_low_output_parallelism(M: int, N: int, K: int) -> bool:
    """Identify reductions whose *physical* output grid is at most one wave.

    The old estimate used ``ceil(N / 16)`` for every GEMM.  Sixteen columns is
    the narrow-N tile, but it is not the tile used by the throughput families
    for a wide, deep reduction.  That proxy consequently classified a wide
    GEMM as well-parallelized even when its actual 128-column grid had only a
    handful of programs.  Use the tile width of the relevant family: wide
    outputs with at least two native K tiles use the 128-column family; all
    other shapes retain the latency-oriented 16-column estimate.  Both terms
    vary monotonically with the problem dimensions and describe tile waves,
    rather than individual model shapes.
    """
    if K < 1024:
        return False
    tile_n = _latency_tile_width(N, K)
    latency_tiles = triton.cdiv(M, 64) * triton.cdiv(N, tile_n)
    return latency_tiles <= _PPU_SMS or _is_deep_fixed_row(M, N, K)


def _estimate_output_tiles(
    batch: int, M: int, N: int, block_m: int, block_n: int
) -> int:
    """Estimate physical output programs for a candidate tile family."""
    return max(int(batch), 1) * triton.cdiv(M, block_m) * triton.cdiv(N, block_n)


def _prefer_small_m_kernel(batch: int, M: int, N: int, K: int) -> bool:
    """Use the fixed legal M tile when a small-M launch cannot fill a wave.

    The K guard avoids replacing the low-latency dense path for shallow
    reductions, while the tile estimate remains independent of model shapes.
    """
    if M <= 0 or N <= 0 or K <= 0 or M >= 32:
        return False
    # A full physical 16-row tile remains useful for small output grids.  Once
    # the fixed-row family would need many 512-column waves, the ordinary
    # GEMM family exposes a 32-row/512-column program and avoids repeatedly
    # launching masked row work.  The boundary is expressed in physical tile
    # waves, not a model-shape exception; it also preserves the fast path for
    # narrow decode projections such as M=2,N=1024.
    if M <= _SMALL_M_TILE:
        # For medium/wide outputs, the fixed 16-row family launches a single
        # heavily masked row tile and cannot amortize its pipeline setup.  Let
        # the ordinary GEMM family use a wider physical row tile instead.  The
        # boundary is expressed in output width, so adjacent shapes follow the
        # same policy rather than introducing model-specific points.
        # A singleton input row may still use this family for wide output
        # grids after the row-GEMV work model rejects its scalar launch.  For
        # multiple rows, retain the dense path for deep reductions where
        # repeatedly masking a 16-row tile is not profitable.
        small_grid = triton.cdiv(M, _SMALL_M_TILE) * triton.cdiv(N, 512)
        if small_grid < _PPU_SMS // 2:
            return True
        return False
    if N >= 64:
        return False
    # A single partially populated 32-row tile remains launch-bound even for
    # a two-tile reduction.  Use the same physical output-wave estimate for
    # K=2048 as for deeper reductions; this covers medium small-M projections
    # without enumerating model-specific N values.
    if K < _GEMV_REDUCTION_TILE:
        return False
    output_tiles = _estimate_output_tiles(batch, M, N, 32, 256)
    return output_tiles <= _PPU_SMS


def _prefer_deep_small_m_narrow(batch: int, M: int, N: int, K: int) -> bool:
    """Use the narrow matrix-unit family for a deep, partial-row reduction.

    The regular BM32 family wins when the N grid is either too small to fill
    the machine or large enough to provide several waves. Between those
    regimes, a native BM16/BN64 tile avoids carrying a second fully masked row
    tile through a deep reduction. Express the crossover in physical output
    tiles so it applies equally to nearby widths and batched inputs.
    """
    if batch <= 0 or not (1 < M <= _SMALL_M_TILE):
        return False
    if N <= 0 or K < 4 * _GEMV_REDUCTION_TILE:
        return False
    output_tiles = _estimate_output_tiles(batch, M, N, _SMALL_M_TILE, 64)
    return _PPU_SMS // 2 <= output_tiles <= _PPU_SMS


def _prefer_deep_mid_m(batch: int, M: int, N: int, K: int) -> bool:
    """Use the fixed-BM32 mid-M family for bounded deep partial rows.

    The regular raster kernel carries GROUP_M/indexing overhead even when
    ``16 < M < 32`` guarantees one physical BM32 row tile.  A fixed-row
    launch removes that overhead and is beneficial while the output grid is
    still latency-sized.  The reduction-depth bound is intentional: at very
    deep K the fixed-row launch loses to the regular pipeline's larger K
    tiles.  All bounds are tile counts/ranges so adjacent shapes share the
    same dispatch bucket.
    """
    if batch <= 0 or not (_SMALL_M_TILE < M < 2 * _SMALL_M_TILE):
        return False
    if N <= 0 or K <= 0:
        return False
    output_tiles = _estimate_output_tiles(batch, M, N, 32, 64)
    reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
    return 16 <= output_tiles <= _PPU_SMS and 3 <= reduction_tiles <= 4


def _prefer_grouped_mid_m(batch: int, M: int, N: int, K: int) -> bool:
    """Reuse each B tile across two physical BM16 row tiles for deep mid-M."""
    if batch <= 0 or not (_SMALL_M_TILE < M < 2 * _SMALL_M_TILE):
        return False
    if N <= 0 or K < 4 * _GEMV_REDUCTION_TILE:
        return False
    output_tiles = _estimate_output_tiles(batch, M, N, _SMALL_M_TILE, 64)
    return _PPU_SMS // 2 <= output_tiles <= _PPU_SMS


def _should_use_narrow_n_gemv(M: int, N: int, K: int) -> bool:
    """Switch only oversized column reductions to the matrix-unit path.

    A matrix-unit tile pads the singleton N dimension to its physical tile
    width, while GEMV computes exactly one output column.  The old
    ``ceil(M / 16) >= 64`` cutoff therefore sent many latency-bound decode
    shapes to a needlessly wide MMA tile.  Compare the scalar GEMV work in
    program/reduction tiles against one wave budget; this remains monotonic in
    M and K and lets the vector path cover ordinary column products.
    """
    if M <= 0 or N != 1 or K <= 0:
        return False
    gemv_programs = triton.cdiv(M, _GEMV_PROGRAM_WIDTH)
    reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
    return (
        gemv_programs * reduction_tiles > _GEMV_COLUMN_TILE_BUDGET
        and K >= _GEMV_REDUCTION_TILE
    )


def _should_use_row_vector_gemv(batch: int, N: int, K: int) -> bool:
    """Prefer scalar row-GEMV while its reduction grid remains bounded.

    The alternative is a masked 16-row matrix-unit tile.  This model compares
    the number of scalar output programs and K reduction tiles instead of
    encoding model-specific N/K values, so increasing batch, width, or
    reduction depth moves monotonically toward the matrix-unit path.
    """
    if batch <= 0 or N <= 0 or K <= 0:
        return False
    output_programs = batch * triton.cdiv(N, _GEMV_PROGRAM_WIDTH)
    reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
    return output_programs * reduction_tiles <= _GEMV_SCALAR_TILE_BUDGET


def _should_use_row_vector_narrow_mm(M: int, N: int, K: int) -> bool:
    """Use a matrix-unit tile for deep, wide one-row reductions.

    Scalar row GEMV is latency-efficient until its output grid and reduction
    depth together exceed several PPU waves.  At that point a 16x64 matrix
    tile amortizes the strided B loads better.  This is a continuous work
    threshold; M=1 is the semantic row-vector family, not a model shape.
    """
    if M != 1 or N <= 0 or K <= 0:
        return False
    output_programs = triton.cdiv(N, _GEMV_PROGRAM_WIDTH)
    reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
    return (
        output_programs >= 2 * _PPU_SMS
        and reduction_tiles >= 4
        and output_programs * reduction_tiles > 4 * _PPU_SMS
    )


def _should_use_multi_row_gemv(M: int, N: int, K: int) -> bool:
    """Prefer scalar row reductions only while their grid stays launch-sized."""
    if M <= 1 or N <= 1 or K <= 0:
        return False
    programs_per_row = triton.cdiv(N, _GEMV_PROGRAM_WIDTH)
    output_programs = M * programs_per_row
    reduction_tiles = triton.cdiv(K, _GEMV_REDUCTION_TILE)
    # Scalar row reductions stop being competitive once a row spans more than
    # eight 16-column programs and the resulting grid exceeds two physical
    # waves.  Allowing the second wave covers small row counts whose padded
    # BM16 matrix tile wastes most lanes (for example M=4,N=512 and
    # M=8,N=256), while still handing wider grids to the dense family.  Both
    # terms are tile counts, so neighbouring M/N values transition
    # continuously rather than introducing shape-specific exceptions.
    if programs_per_row > 8 and M * programs_per_row > 2 * _PPU_SMS:
        return False
    if programs_per_row == 1:
        program_budget = _SINGLE_PROGRAM_ROW_BUDGET
        tile_budget = _SINGLE_PROGRAM_ROW_TILE_BUDGET
    else:
        program_budget = _MULTI_ROW_GEMV_PROGRAM_BUDGET
        tile_budget = _MULTI_ROW_GEMV_TILE_BUDGET
    return (
        output_programs <= program_budget
        and output_programs * reduction_tiles <= tile_budget
    )


def _should_use_grouped_row_gemv(M: int, N: int, K: int) -> bool:
    """Use grouped rows when scalar row-GEMV would launch too many programs.

    A grouped program computes a small, compile-time number of adjacent rows
    against the same B tile.  The policy is expressed entirely in physical
    tile counts and wave capacity, so neighbouring dimensions move smoothly
    between scalar, grouped, and matrix-unit families.
    """
    # A single reduction tile leaves too little B reuse to amortize the
    # extra row-group register footprint; require at least two physical K
    # tiles before entering this family.
    if M < 32 or N <= 1 or N > 32 or K < 2 * _GEMV_REDUCTION_TILE:
        return False
    n_tiles = triton.cdiv(N, _GEMV_PROGRAM_WIDTH)
    scalar_programs = M * n_tiles
    # Use a physical 16-row group so the reduction maps to the PPU's native
    # matrix-unit tile.  Wider output tiles are still bounded by the config
    # footprint and wave budget below.
    rows_per_program = 16
    grouped_programs = triton.cdiv(M, rows_per_program) * n_tiles
    # Grouping is useful after scalar row launches exceed two waves, but avoid
    # turning very large matrices into an unbounded number of tiny reduction
    # programs.  The limits are expressed as wave multiples so they apply
    # smoothly to neighbouring M/N/K extents.
    return (
        scalar_programs > 2 * _PPU_SMS
        and grouped_programs <= 128 * _PPU_SMS
        and grouped_programs * triton.cdiv(K, _GEMV_REDUCTION_TILE) <= 512 * _PPU_SMS
    )


def _aiu_load_mask(a, b) -> int:
    """Return the AIU operands whose base and row starts are 32-byte aligned."""
    mask = 0
    if a.data_ptr() % 32 == 0 and a.stride(-2) * a.element_size() % 32 == 0:
        mask |= _AIU_LOAD_A
    # A dense 2-D transpose view is physically row-major [N, K]. Its logical
    # row stride is one element, while logical columns identify the aligned
    # physical rows. PPU's order=(0,1) descriptor supports this column-major
    # view directly, so validate that physical row stride as an AIU operand.
    b_row_stride = b.stride(-2)
    if b.ndim == 2 and b.stride(-2) == 1:
        b_row_stride = b.stride(-1)
    if b.data_ptr() % 32 == 0 and b_row_stride * b.element_size() % 32 == 0:
        mask |= _AIU_LOAD_B
    return mask


def _load_mode_supported(load_mode: int, aiu_load_mask: int) -> bool:
    if load_mode == _LOAD_BOTH_AIU:
        return aiu_load_mask == (_AIU_LOAD_A | _AIU_LOAD_B)
    if load_mode == _LOAD_A_AIU:
        return bool(aiu_load_mask & _AIU_LOAD_A)
    if load_mode == _LOAD_B_AIU:
        return bool(aiu_load_mask & _AIU_LOAD_B)
    return True


def _split_k_wave_plan(batch: int, M: int, N: int, K: int):
    """Estimate a useful split count from output waves and reduction cost.

    This is deliberately a continuous shape model.  LibTuner still selects the
    physical tiles, load mode, and exact split count; this function only decides
    whether the split-K implementation family is worth considering.
    """
    if batch <= 0 or M <= 0 or N <= 0 or K < 2048 or K % 256 != 0:
        return None

    # Very narrow outputs generally have a dedicated matrix-unit family.  The
    # workspace split path duplicates that tile and pays an additional
    # reduction launch; only the physically under-filled N=64 bucket below is
    # admitted. N==1 is handled by the GEMV policy before reaching this model.
    if N <= 32:
        return None
    # For the newly admitted N=33..64 band, split-K only fills the machine
    # when the 64-row raster has seven or eight physical tiles.  Smaller
    # grids regress because workspace allocation and reduction dominate;
    # larger grids already expose enough independent output programs.
    if N <= 64 and not (7 <= triton.cdiv(M, 64) <= 8):
        return None

    # These representative tiles follow the broad PPU throughput regimes.
    # They are estimates, not dispatch requirements; only the explicitly
    # admitted low-wave narrow buckets reach this point.
    if N <= 512:
        block_m, block_n = 64, 128
    else:
        # Split-K is evaluated against the physical 64-row family used by
        # the low-wave PPU GEMM candidates.  Using BM128 here makes a partial
        # 32-row output look artificially under-filled and can enable an
        # expensive split for a shape that is better served by one regular
        # launch.  This is a tile-family choice, not a model-shape cutoff.
        block_m, block_n = 64, 256

    tiles_per_matrix = triton.cdiv(M, block_m) * triton.cdiv(N, block_n)
    output_tiles = batch * tiles_per_matrix
    # Split-K is useful only when independent output tiles do not already fill
    # the machine.  Batch contributes independent output tiles exactly like M
    # and N, so no batch-specific shape table is needed here.
    # Once an eighth-wave of independent output tiles is available, the
    # regular GEMM already keeps enough PPU SMs busy.  Splitting K there only
    # adds workspace allocation and a reduction launch.  The eighth-wave
    # bound is continuous and leaves split-K for genuinely under-filled
    # grids, where its extra K parallelism can amortize the setup cost.
    if output_tiles > _PPU_SMS // 8:
        # A four-tile N=512 launch still benefits from split-K while its
        # physical row grid fits six BM64 tiles.  Beyond that boundary the
        # regular GEMM exposes enough work and split-K overhead dominates.
        if (
            not (256 < N <= 512 and triton.cdiv(M, block_m) <= 6)
            and not (
                256 < N <= 512
                and triton.cdiv(M, block_m) <= 8
                and 4 * _GEMV_REDUCTION_TILE <= K <= 7 * _GEMV_REDUCTION_TILE
            )
            and not (
                16 * 128 <= N <= 17 * 128
                and M <= 4 * 64
                and K >= 7 * _GEMV_REDUCTION_TILE
            )
            and not (
                20 * 128 <= N <= 21 * 128
                and M <= 4 * 64
                and K >= 4 * _GEMV_REDUCTION_TILE
            )
            and not (
                16 * 16 <= N <= 2 * 128
                and triton.cdiv(M, block_m) <= 8
                and K >= 4 * _GEMV_REDUCTION_TILE
            )
            and not (
                3 * 128 <= N < 4 * 128
                and triton.cdiv(M, block_m) <= 8
                and K >= 7 * _GEMV_REDUCTION_TILE
            )
            and not (
                4 * 256 <= N < 5 * 256
                and triton.cdiv(M, block_m) <= 8
                and K >= 2 * _GEMV_REDUCTION_TILE
            )
        ):
            return None
    # A partially populated physical row tile has insufficient independent
    # M work to amortize workspace allocation and the reduction launch.  The
    # boundary follows the selected physical tile, so M moves continuously
    # into split-K once it can fill at least one row tile.
    if M < block_m and not (
        32 <= M < block_m and N >= 8 * 256 and K >= 4 * _GEMV_REDUCTION_TILE
    ):
        return None
    # A partial row tile with a medium/wide output already has enough
    # independent N work for the regular family.  Forcing split-K on this
    # small-M band adds workspace traffic without increasing useful waves.
    if M < 32 and N >= 1024:
        return None
    # Aim to bring one output wave up to the physical SM count.  Rounding the
    # ratio to a power of two matches the kernel's legal split candidates and
    # avoids over-splitting medium-size grids.
    wave_ratio = triton.cdiv(_PPU_SMS, max(output_tiles, 1))
    split = 1 << max(0, (wave_ratio - 1).bit_length())
    split = min(_PPU_MAX_SPLIT_K, max(2, split))
    # Once the regular grid already exposes two low-wave groups (16 output
    # tiles on the 64-SM PPU), a two-way reduction is sufficient to add useful
    # K parallelism.  Larger fan-out only multiplies workspace traffic and
    # reducer work; cap it continuously by output-tile pressure rather than
    # allowing adjacent M buckets to drift to split4/8.
    if output_tiles >= _PPU_SMS // 4:
        split = min(split, 2)
    split = min(split, max(1, K // 1024))
    if split < 2:
        return None
    if K // split < max(_SPLIT_K_MIN_REDUCTION_PER_SLICE, 4 * 32):
        return None

    workspace_bytes = split * batch * M * N * 4
    if workspace_bytes > 128 * 1024 * 1024:
        return None

    # Split-K needs a temporary allocation, an extra kernel launch, and a
    # reduction. PPU allocation alone has a substantial fixed latency, so a
    # low-wave GEMM must contain enough arithmetic to amortize that setup.
    # Model the regular kernel at a conservative 64 TFLOP/s and require enough
    # compute to cover the modeled setup threshold. LibTuner then selects the
    # concrete split count and physical tiles.
    regular_compute_us = 2 * batch * M * N * K / _PPU_GEMM_FLOPS_PER_US
    if regular_compute_us < _SPLIT_K_SETUP_US:
        return None

    return split


def _configs_from_specs(specs, fields):
    """Build Triton configs while keeping tuple shortlists easy to audit."""
    configs = []
    for spec in dict.fromkeys(specs):
        if len(spec) != len(fields):
            raise ValueError(
                "PPU tuning spec has "
                f"{len(spec)} values but {len(fields)} fields were declared: "
                f"spec={spec!r}, fields={fields!r}"
            )
        kwargs = dict(zip(fields, spec))
        num_warps = kwargs.pop("num_warps")
        num_stages = kwargs.pop("num_stages")
        kwargs["PIPE_STAGES"] = num_stages
        configs.append(
            triton.Config(kwargs, num_warps=num_warps, num_stages=num_stages)
        )
    return configs


def _prune_split_k_configs(configs, named_args, **kwargs):
    """Keep split-K candidates that are addressable and workspace-safe.

    The split count is a performance decision made by LibTuner.  This prune
    only rejects candidates whose non-interleaved slices cannot be represented
    by the kernel or whose temporary workspace exceeds a generic bound.
    """
    K = named_args.get("K")
    if K is None:
        return configs
    K = int(K)
    M = int(named_args.get("M", 1))
    N = int(named_args.get("N", 1))
    batch = max(int(named_args.get("batch", 1)), 1)
    aiu_load_mask = named_args.get("aiu_load_mask", _AIU_LOAD_A | _AIU_LOAD_B)
    filtered = []
    for config in configs:
        split_k = int(config.kwargs.get("SPLIT_K", 1))
        block_k = int(config.kwargs.get("BLOCK_K", 1))
        interleaved = bool(config.kwargs.get("INTERLEAVED", False))
        load_mode = config.kwargs.get("LOAD_MODE", _LOAD_REGULAR)
        if split_k <= 0 or block_k <= 0:
            continue
        if not _load_mode_supported(load_mode, aiu_load_mask):
            continue
        if not interleaved and (K % split_k != 0 or (K // split_k) % block_k != 0):
            continue
        if split_k * batch * max(M, 1) * max(N, 1) * 4 > 128 * 1024 * 1024:
            continue
        filtered.append(config)
    return filtered or configs


def _prune_gemv_configs(configs, named_args, **kwargs):
    """Keep GEMV candidates unless they violate compile/resource limits.

    Candidate ranking is intentionally left to LibTuner.  In particular, do
    not use runtime extents or a measured tall-column winner to shrink this
    list: nearby shapes and different PPU targets must be able to benchmark
    the complete legal family.
    """
    filtered = []
    for config in configs:
        block_m = int(config.kwargs.get("BLOCK_M", 1))
        block_k = int(config.kwargs.get("BLOCK_K", 128))
        stages = int(config.kwargs.get("PIPE_STAGES", config.num_stages or 1))
        if block_m <= 0 or block_k <= 0 or stages <= 0:
            continue
        # BK4096/8192 are row-vector-only candidates.  Pairing them with
        # multiple output rows creates huge scalar IR (BM8/BK8192 took more
        # than six minutes in ppu-llc) without adding useful K parallelism.
        if block_k > 2048 and block_m != 1:
            continue
        # Large output-row x reduction tiles scalarize into pathological LLVM
        # on PPU.  The production Pareto set stays at or below a 16K-element
        # working tile once BK reaches 1024; larger Cartesian combinations add
        # multi-minute compiler work without representing a deployable kernel.
        if block_k >= 1024 and block_m * block_k > 16384:
            continue
        if block_m * block_k > 65536:
            continue
        if stages * (block_m + block_k) * 2 > 192 * 1024:
            continue
        filtered.append(config)
    return filtered or configs


def _prune_single_gemv_configs(configs, named_args, **kwargs):
    """Add the exact-output-row dominance rule for the single GEMV kernel."""
    filtered = _prune_gemv_configs(configs, named_args, **kwargs)
    M = named_args.get("M")
    if M is None or int(M) <= 0:
        return filtered
    max_block_m = 1 << (int(M) - 1).bit_length()
    row_filtered = [
        config
        for config in filtered
        if int(config.kwargs.get("BLOCK_M", 1)) <= max_block_m
    ]
    return row_filtered or filtered


def _prune_grouped_row_gemv_configs(configs, named_args, **kwargs):
    """Keep grouped-row GEMV candidates that fit compile-time resources."""
    filtered = []
    for config in configs:
        block_n = int(config.kwargs.get("BLOCK_M", 16))
        block_k = int(config.kwargs.get("BLOCK_K", 128))
        rows = int(config.kwargs.get("ROWS_PER_PROGRAM", 1))
        stages = int(config.kwargs.get("PIPE_STAGES", config.num_stages or 1))
        if block_n <= 0 or block_k <= 0 or rows <= 0 or stages <= 0:
            continue
        if rows * block_n * block_k > 65536:
            continue
        if stages * (block_n + block_k) * rows * 2 > 192 * 1024:
            continue
        filtered.append(config)
    return filtered or configs


def _prune_gemm_configs(configs, named_args, **kwargs):
    """Apply only generic PPU legality checks to GEMM candidates.

    Candidate selection belongs to LibTuner.  In particular, do not encode
    measured tile winners, dimension buckets, or next-power-of-two extent
    limits here: those rules make the expanded search behave like a hidden
    dispatch table and can regress a neighbouring shape.  The remaining
    checks describe addressability and compiler/resource limits only.
    """
    M = named_args.get("M")
    N = named_args.get("N")
    aiu_load_mask = named_args.get("aiu_load_mask", _AIU_LOAD_A | _AIU_LOAD_B)

    filtered = []
    for config in configs:
        bm = int(config.kwargs.get("BLOCK_M", 16))
        bn = int(config.kwargs.get("BLOCK_N", 16))
        bk = int(config.kwargs.get("BLOCK_K", 32))
        stages = int(config.kwargs.get("PIPE_STAGES", config.num_stages or 1))
        load_mode = config.kwargs.get("LOAD_MODE", _LOAD_REGULAR)

        if bm <= 0 or bn <= 0 or bk <= 0 or stages <= 0:
            continue
        if not _load_mode_supported(load_mode, aiu_load_mask):
            continue

        # PPU's masked TSM path faults at launch when a physical row tile is
        # more than four times the logical M extent (for example BM256 at
        # M=32). Such severely under-filled tiles are not useful GEMM
        # candidates and must be removed before benchmarking because a device
        # launch fault poisons the CUDA context instead of failing one config.
        if M is not None and int(M) * 4 < bm:
            continue

        # Very tall launches exceed the PPU address/launch safety envelope
        # with narrow row tiles.  Keep BM128+ for M above 32K; for smaller
        # matrices, BM64 is still a valid and useful throughput candidate.
        # This avoids illegal-page faults seen during graph-replay tuning of
        # huge-M workloads while retaining the normal GEMM family.
        if M is not None and int(M) > 32768 and bm < 128:
            continue

        # TLE descriptors cannot address an async tile past the PPU's
        # descriptor extent.  Ultra-wide launches are split into descriptor-
        # sized chunks by the caller, and regular pointer loads remain valid.
        if N is not None and int(N) > _PPU_DESCRIPTOR_MAX_N:
            if load_mode != _LOAD_REGULAR or bn > 256:
                continue

        # These are compiler/resource bounds, shared by default and expanded
        # candidates.  They do not select a tile family or depend on a model
        # shape; candidates exceeding them cannot compile reliably on PPU.
        if stages * (bm + bn) * bk * 2 > 192 * 1024:
            continue
        if bm * bn > config.num_warps * 32 * 128:
            continue
        filtered.append(config)

    # Never return an empty candidate list solely because an optional
    # metadata field was unavailable.  LibTuner can then report/skip a
    # compile-invalid candidate instead of silently selecting a hidden
    # shape-specific fallback.
    return filtered or configs


def _prune_narrow_n_configs(configs, named_args, **kwargs):
    """Keep narrow-N candidates based only on legality/resource checks."""
    aiu_load_mask = named_args.get("aiu_load_mask", _AIU_LOAD_A | _AIU_LOAD_B)
    filtered = []
    for config in configs:
        bm = int(config.kwargs.get("BLOCK_M", 16))
        bn = int(config.kwargs.get("BLOCK_N", 16))
        bk = int(config.kwargs.get("BLOCK_K", 32))
        stages = int(config.kwargs.get("PIPE_STAGES", config.num_stages or 1))
        load_mode = config.kwargs.get("LOAD_MODE", _LOAD_REGULAR)
        if bm <= 0 or bn <= 0 or bk <= 0 or stages <= 0:
            continue
        if not _load_mode_supported(load_mode, aiu_load_mask):
            continue
        if bm * bn > config.num_warps * 32 * 128:
            continue
        if stages * (bm + bn) * bk * 2 > 192 * 1024:
            continue
        filtered.append(config)
    return filtered or configs


if HAS_PPU_TLE:

    @triton.jit
    def _ppu_gemm_tile(
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
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        LOAD_MODE: tl.constexpr,
        B_TRANSPOSED: tl.constexpr,
        PIPE_STAGES: tl.constexpr,
        MASK_M: tl.constexpr,
        ALIGNED_A_512X128: tl.constexpr,
        ALIGNED_B_128X128: tl.constexpr,
        EVEN_K: tl.constexpr,
        FULL_K_TILES: tl.constexpr,
        FULL_M_TILES: tl.constexpr,
        FULL_N_TILES: tl.constexpr,
        EVEN_M: tl.constexpr,
        EVEN_N: tl.constexpr,
        FUSE_BIAS: tl.constexpr,
    ):
        """Compute and store one general GEMM tile; inlined by Triton."""
        a_block_ptr = tl.make_block_ptr(
            base=A,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        if B_TRANSPOSED:
            # Logical B is column-major because its dense backing storage is
            # [N, K]. PPU AIU descriptors represent that layout with order
            # (0, 1), preserving the logical [K, N] tile consumed by dot.
            b_block_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(0, 1),
            )
        else:
            b_block_ptr = tl.make_block_ptr(
                base=B,
                shape=(K, N),
                strides=(stride_bk, stride_bn),
                offsets=(0, pid_n * BLOCK_N),
                block_shape=(BLOCK_K, BLOCK_N),
                order=(1, 0),
            )
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        load_m = offs_m.to(tl.int64)
        load_n = offs_n.to(tl.int64)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in tl.range(0, K, BLOCK_K, num_stages=PIPE_STAGES):
            offs_k = (k_start + tl.arange(0, BLOCK_K)).to(tl.int64)
            if LOAD_MODE == 0 or LOAD_MODE == 1:
                if (ALIGNED_A_512X128 or FULL_M_TILES) and FULL_K_TILES:
                    a = tle.load(a_block_ptr, is_async=True)
                else:
                    a = tle.load(
                        a_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        is_async=True,
                    )
            else:
                a_ptrs = A + load_m[:, None] * stride_am + offs_k[None, :] * stride_ak
                # ``EVEN_M``/``EVEN_K`` describe logical extents, while the
                # selected constexpr tile may be larger (for example M=32
                # with BLOCK_M=64 or K divisible by 64 with BLOCK_K=128).
                # FULL_K_TILES is a config-dependent heuristic, so aligned
                # candidates compile to the original branch-free load.  For
                # a ragged candidate, test each iteration and predicate only
                # its final partial K tile.  TLE descriptor loads above
                # already perform equivalent boundary checks.
                if (
                    (EVEN_M or FULL_M_TILES)
                    and EVEN_K
                    and (FULL_K_TILES or k_start + BLOCK_K <= K)
                ):
                    a = tl.load(a_ptrs)
                else:
                    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

            if LOAD_MODE == 0 or LOAD_MODE == 2:
                if (
                    (ALIGNED_B_128X128 or FULL_N_TILES)
                    and FULL_K_TILES
                    and BLOCK_N <= 128
                ):
                    b = tle.load(b_block_ptr, is_async=True)
                else:
                    b = tle.load(
                        b_block_ptr,
                        boundary_check=(0, 1),
                        padding_option="zero",
                        is_async=True,
                    )
            else:
                b_ptrs = B + offs_k[:, None] * stride_bk + load_n[None, :] * stride_bn
                # As above, keep complete K tiles on the unmasked path while
                # guarding the final partial tile.  EVEN_N is a logical-shape
                # hint and cannot prove divisibility by the selected BLOCK_N;
                # the masked branch therefore checks both axes.
                if (
                    EVEN_K
                    and (EVEN_N or FULL_N_TILES)
                    and (FULL_K_TILES or k_start + BLOCK_K <= K)
                ):
                    b = tl.load(b_ptrs)
                else:
                    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

        store_m = offs_m.to(tl.int64)
        store_n = offs_n.to(tl.int64)
        c_ptrs = C + store_m[:, None] * stride_cm + store_n[None, :] * stride_cn
        c_complete = (EVEN_M or FULL_M_TILES) and (EVEN_N or FULL_N_TILES)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        if FUSE_BIAS:
            bias_ptrs = (
                Bias
                + store_m[:, None] * stride_bias_m
                + store_n[None, :] * stride_bias_n
            )
            bias_value = tl.load(bias_ptrs, mask=c_mask, other=0.0)
            acc = alpha * acc + beta * bias_value
        # Store masks are cheap relative to the reduction and prevent a
        # mismatched constexpr tile from writing a ragged M/N tail.  The
        # previous EVEN_M/EVEN_N shortcut was unsafe because those flags were
        # computed from coarse 512-wide buckets rather than BLOCK_M/BLOCK_N.
        if c_complete:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty))
        else:
            tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)

else:
    _ppu_gemm_tile = None
