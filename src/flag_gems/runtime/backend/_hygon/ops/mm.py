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

"""Hygon floating-point matrix multiplication."""

import copy
import logging
import os
import statistics
from dataclasses import asdict, dataclass
from functools import cache, lru_cache
from math import inf, isfinite, prod

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    HAS_HYGON_TLE = True
except ImportError:
    # HCU FlagTree builds provide TLE.  Keep module importable with a stock
    # Triton so non-HCU tooling can still inspect the backend.
    tle = None
    HAS_HYGON_TLE = False

try:
    from triton.language.extra.hip import accumulator_fence
    from triton.language.extra.hip import cta_barrier as cta_phase
except ImportError:
    accumulator_fence = cta_phase = None

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

from .gemm_utils import (
    GemmChoice,
    GemmChoiceCache,
    I32_GEMM_COORD_LIMIT,
    benchmark_gemm_candidate,
    capture_gemm_fallback,
    gemm_has_wide_storage,
    gemm_replay_benchmarker,
    gemm_storage_is_wide,
    gemm_tuning_key,
    supports_hcu_options,
    try_gemm_candidate,
)

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_hygon_expand.yaml")
)


# Ordinary GEMM tuning and kernel.
def _hygon_mm_bucket(value):
    """Bucket dimensions at tile/work boundaries for config reuse.

    The kernel keeps M/N/K runtime-dynamic, so nearby shapes can safely share
    a tuned tile family. Small dimensions stay exact because launch overhead
    dominates there; larger dimensions are bucketed at boundaries that match
    the 16/32/64 output tiles and 64/128 reduction tiles.
    """
    value = int(value)
    if value <= 16:
        return value
    if value <= 128:
        quantum = 16
    elif value <= 512:
        quantum = 32
    elif value <= 2048:
        quantum = 64
    elif value <= 8192:
        quantum = 256
    else:
        quantum = 1024
    return ((value + quantum - 1) // quantum) * quantum


def _hygon_mm_tile_bucket(value):
    # Keep cached full-tile configurations valid within each shape bucket.
    value = int(value)
    alignment = min(256, value & -value)
    return f"{_hygon_mm_bucket(value)}:{alignment}"


def _hygon_mm_large_options_supported(*additional_options):
    return supports_hcu_options(
        ("mmac_warps_m", "optimize_epilogue", "schedule_hint", *additional_options)
    )


_HYGON_MM_LARGE_OPTIONS_SUPPORTED = _hygon_mm_large_options_supported()
_HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED = _hygon_mm_large_options_supported(
    "mmac_transposed"
)
_HYGON_MM_SINGLE_WAVE_OPTIONS_SUPPORTED = _hygon_mm_large_options_supported(
    "mmac_transposed", "kpack"
)
_HYGON_MM_WAVES_OPTIONS_SUPPORTED = _hygon_mm_large_options_supported("waves_per_eu")
_HYGON_MM_PACKED_ASYNC_OPTIONS_SUPPORTED = _hygon_mm_large_options_supported(
    "mmac_transposed", "kpack"
)
_HYGON_SPLITK_OPTIONS_SUPPORTED = supports_hcu_options(
    ("optimize_epilogue", "schedule_hint")
)


def _hygon_mm_is_packed_async_config(config):
    kwargs = dict(config.kwargs)
    if not kwargs.pop("ASYNC_LOAD", False) or kwargs.pop("kpack", 1) != 2:
        return False
    if kwargs.pop("waves_per_eu", 1) != 1:
        return False
    return (config.num_warps, config.num_stages, config.num_ctas) == (
        4,
        2,
        1,
    ) and kwargs in (
        dict(
            BLOCK_M=block_m,
            BLOCK_N=128,
            BLOCK_K=32,
            GROUP_M=4,
            mmac_warps_m=0,
            mmac_transposed=True,
            optimize_epilogue=True,
            schedule_hint="local-prefetch",
        )
        for block_m in (64, 128)
    )


def _hygon_mm_is_large_config(config):
    kwargs = dict(config.kwargs)
    async_load = kwargs.pop("ASYNC_LOAD", False)
    waves = kwargs.pop("waves_per_eu", 1)
    if waves not in (1, 4) or (
        async_load and (waves != 1 or not kwargs.get("mmac_transposed", False))
    ):
        return False
    transposed = kwargs.pop("mmac_transposed", False)
    if kwargs.pop("kpack", 1) != 1:
        return False
    return (
        (config.num_warps, config.num_stages, config.num_ctas) == (8, 2, 1)
        and kwargs in (
            dict(
                BLOCK_M=256,
                BLOCK_N=256,
                BLOCK_K=16,
                GROUP_M=group_m,
                mmac_warps_m=4,
                optimize_epilogue=transposed,
                schedule_hint="local-prefetch",
            )
            for group_m in (1, 4)
        )
    )


def _hygon_mm_is_dense_config(config):
    kwargs = dict(config.kwargs)
    if kwargs.pop("ASYNC_LOAD", False):
        return _hygon_mm_is_packed_async_config(config) or _hygon_mm_is_large_config(
            config
        )
    if kwargs.pop("waves_per_eu", 1) not in (1, 4):
        return False
    if kwargs.pop("kpack", 1) != 1:
        return False
    if _hygon_mm_is_large_config(config):
        return True
    return (
        (config.num_warps, config.num_stages, config.num_ctas) == (4, 2, 1)
        and kwargs in (
            dict(
                BLOCK_M=block_m,
                BLOCK_N=128,
                BLOCK_K=32,
                GROUP_M=group_m,
                mmac_warps_m=0,
                mmac_transposed=True,
                optimize_epilogue=True,
                schedule_hint="local-prefetch",
            )
            for block_m in (64, 128)
            for group_m in (1, 4)
        )
    )


def _hygon_mm_is_single_wave_config(config):
    kwargs = dict(config.kwargs)
    if kwargs.pop("ASYNC_LOAD", False):
        return False
    if kwargs.pop("waves_per_eu", 1) != 1:
        return False
    return (
        (config.num_warps, config.num_stages, config.num_ctas) == (1, 2, 1)
        and kwargs in (
            dict(
                BLOCK_M=block_m,
                BLOCK_N=64,
                BLOCK_K=block_k,
                GROUP_M=1,
                mmac_warps_m=0,
                mmac_transposed=False,
                optimize_epilogue=True,
                schedule_hint="local-prefetch",
                kpack=kpack,
            )
            for block_m in (16, 32)
            for block_k in (32, 64)
            for kpack in (1, 2)
        )
    )


def _hygon_mm_async_load_mask(a, b, c, M, N, K):
    """Describe legal direct-to-LDS operands independently of tuning choices."""
    if not (
        _HYGON_MM_LARGE_OPTIONS_SUPPORTED
        and HAS_HYGON_TLE
        and a is not None and b is not None and c is not None
        and a.dtype == b.dtype == c.dtype
        and a.dtype in (torch.float16, torch.bfloat16)
        and a.device == b.device == c.device
        and a.device.type == "cuda"
        and c.shape == (M, N)
        and M > 0 and N > 0 and K >= 16 and K % 16 == 0
        and c.is_contiguous() and c.data_ptr() % 16 == 0
        and c.untyped_storage().nbytes() <= 2**31 - 1
    ):
        return 0
    from triton import knobs

    target = triton.runtime.driver.active.get_current_target()
    if not (
        target.backend == "hip" and target.arch == "gfx936"
        and knobs.hcu.use_buffer_ops and not knobs.hcu.use_async_copy
        and not knobs.hcu.use_block_pingpong
        and not knobs.hcu.use_in_thread_transpose
    ):
        return 0
    a_async = (
        a.stride() == (K, 1) and a.data_ptr() % 16 == 0
        and a.untyped_storage().nbytes() <= 2**31 - 1
    )
    b_async = (
        (b.stride() == (1, K) or (b.stride() == (N, 1) and N % 16 == 0))
        and b.data_ptr() % 16 == 0
        and b.untyped_storage().nbytes() <= 2**31 - 1
    )
    return int(a_async) | (int(b_async) << 1)


def _hygon_mm_large_tile_eligible(a, M, N, K, load_mask):
    # Preserve the production shortlist's domain. Expanded uses the separate
    # operand capability mask and does not inherit these size preferences.
    masked_family = (
        _HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED
        and M >= 64 and N >= 128 and M * N >= 65536 and K >= 128
    )
    minimum_tile = 128 if _HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED else 256
    return (
        a.dtype == torch.bfloat16
        and bool(load_mask)
        and (masked_family or (
            M >= minimum_tile and N >= minimum_tile
            and M % minimum_tile == N % minimum_tile == 0
        ))
    )


def _hygon_mm_output_tile_fits(M, N, bm, bn, *, allow_tail=False):
    """Require one full output tile and exact coverage unless tails are allowed."""
    return M >= bm and N >= bn and (allow_tail or (M % bm == 0 and N % bn == 0))


def _hygon_mm_expanded_prune(configs, named_args, **kwargs):
    """Prune correctness/capability limits, leaving ranking to tuning."""
    M, N = int(named_args["M"]), int(named_args["N"])
    masked = kwargs.get("MASK_EDGES", named_args.get("MASK_EDGES", False))
    load_mask = int(kwargs.get("ASYNC_LOAD_MASK", named_args.get("ASYNC_LOAD_MASK", 0)))
    a, b = named_args.get("a_ptr"), named_args.get("b_ptr")
    a_bytes = a.element_size() if a is not None else 2
    b_bytes = b.element_size() if b is not None else 2
    half_inputs = a_bytes == b_bytes == 2
    result = []
    for config in configs:
        meta = config.kwargs
        bm, bn = (int(meta[k]) for k in ("BLOCK_M", "BLOCK_N"))
        mode, stages = int(meta["LOAD_MODE"]), int(config.num_stages)
        if mode not in (0, 1, 2, 3) or mode & ~load_mask:
            continue
        if not masked and not _hygon_mm_output_tile_fits(M, N, bm, bn):
            continue
        # The supported pipeline depths are validated by offline tests.
        # The runtime tuner ranks executable candidates by latency.
        if mode and stages not in (2, 3, 4):
            continue
        if not half_inputs and (
            meta["mmac_transposed"] or meta["mmac_warps_m"]
            or meta["kpack"] != 1 or meta["schedule_hint"] != "none"
        ):
            continue
        warp_m = int(meta["mmac_warps_m"])
        if warp_m and (
            config.num_warps % warp_m or bm < 16 * warp_m
            or bn < 16 * (config.num_warps // warp_m)
        ):
            continue
        # Mixed loads can use different LDS buffer counts. Check exact shared
        # and register resources during compilation instead of rejecting a
        # legal candidate from a uniform-buffer footprint estimate.
        result.append(config)
    return result


def _hygon_mm_configs():
    """Keep legacy tiles and gated gfx936 dense layout candidates."""
    configs = []
    for config in runtime.get_tuned_config("mm"):
        kwargs = config.kwargs
        if (
            int(kwargs.get("BLOCK_M", 0)) not in (16, 32, 64)
            or int(kwargs.get("BLOCK_N", 0)) not in (16, 32, 64, 128)
            or int(kwargs.get("BLOCK_K", 0)) not in (32, 64, 128)
            or int(getattr(config, "num_stages", 1)) > 2
            or int(getattr(config, "num_warps", 1)) not in (2, 4)
        ):
            continue
        # The loader caches Config objects shared with other tuner instances.
        configs.append(copy.deepcopy(config))
    for config in configs:
        config.kwargs.setdefault("GROUP_M", 8)
    if _HYGON_MM_LARGE_OPTIONS_SUPPORTED:
        from triton import knobs

        # ConfigCache uses one SQL schema for the whole candidate family.
        # Preserve the legacy backend defaults while keeping fields uniform.
        for config in configs:
            config.kwargs.setdefault("mmac_warps_m", 0)
            config.kwargs.setdefault("optimize_epilogue", knobs.hcu.optimize_epilogue)
            config.kwargs.setdefault("schedule_hint", "none")
        configs.extend(
            triton.Config(
                dict(
                    BLOCK_M=256,
                    BLOCK_N=256,
                    BLOCK_K=16,
                    GROUP_M=group_m,
                    mmac_warps_m=4,
                    optimize_epilogue=False,
                    schedule_hint="local-prefetch",
                ),
                num_warps=8,
                num_stages=2,
            )
            for group_m in (1, 4)
        )
    if _HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED:
        # Every candidate needs the same SQL option fields, including defaults.
        for config in configs:
            config.kwargs.setdefault("mmac_transposed", False)
        for block, warps, warp_m, block_k in ((64, 4, 0, 32), (128, 4, 0, 32), (256, 8, 4, 16)):
            configs.extend(
                triton.Config(
                    dict(
                        BLOCK_M=block,
                        BLOCK_N=max(128, block),
                        BLOCK_K=block_k,
                        GROUP_M=group_m,
                        mmac_warps_m=warp_m,
                        mmac_transposed=True,
                        optimize_epilogue=True,
                        schedule_hint="local-prefetch",
                    ),
                    num_warps=warps,
                    num_stages=2,
                )
                for group_m in (1, 4)
            )
    if _HYGON_MM_SINGLE_WAVE_OPTIONS_SUPPORTED:
        # Config-set hashes isolate the new SQL schema from older caches.
        for config in configs:
            config.kwargs.setdefault("kpack", 1)
        configs.extend(
            triton.Config(
                dict(
                    BLOCK_M=block_m,
                    BLOCK_N=64,
                    BLOCK_K=block_k,
                    GROUP_M=1,
                    mmac_warps_m=0,
                    mmac_transposed=False,
                    optimize_epilogue=True,
                    schedule_hint="local-prefetch",
                    kpack=kpack,
                ),
                num_warps=1,
                num_stages=2,
            )
            for block_m in (16, 32)
            for block_k in (32, 64)
            for kpack in (1, 2)
        )
    if _HYGON_MM_WAVES_OPTIONS_SUPPORTED:
        # Masked 256x256 tiles can cross a register-allocation boundary. Keep
        # the default and let the tuner compare the existing occupancy hint.
        # Every candidate needs the same SQL option fields, including defaults.
        for config in configs:
            config.kwargs.setdefault("waves_per_eu", 1)
        extra = []
        for config in configs:
            if _hygon_mm_is_large_config(config) and config.kwargs.get(
                "mmac_transposed", False
            ):
                candidate = copy.deepcopy(config)
                candidate.kwargs["waves_per_eu"] = 4
                extra.append(candidate)
        configs.extend(extra)
    # Keep one SQL option schema and isolate the new family from old caches.
    for config in configs:
        config.kwargs.setdefault("ASYNC_LOAD", False)
    if _HYGON_MM_PACKED_ASYNC_OPTIONS_SUPPORTED:
        # Packed K reduces direct-to-LDS and register/shared read instructions.
        # Keep both measured shapes alongside all existing choices; pruning
        # admits them only for eligible full-output BF16 tiles on gfx936.
        for block_m in (64, 128):
            kwargs = dict(
                BLOCK_M=block_m,
                BLOCK_N=128,
                BLOCK_K=32,
                GROUP_M=4,
                mmac_warps_m=0,
                mmac_transposed=True,
                optimize_epilogue=True,
                schedule_hint="local-prefetch",
                kpack=2,
                ASYNC_LOAD=True,
            )
            if _HYGON_MM_WAVES_OPTIONS_SUPPORTED:
                kwargs["waves_per_eu"] = 1
            configs.append(triton.Config(kwargs, num_warps=4, num_stages=2))
    if _HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED:
        # Compare direct-to-LDS copies for the existing full 256x256 family.
        # Keep the synchronous choices and the same pruning and cache schema.
        extra = []
        for config in configs:
            if (
                not config.kwargs["ASYNC_LOAD"]
                and _hygon_mm_is_large_config(config)
                and config.kwargs.get("mmac_transposed", False)
                and config.kwargs.get("waves_per_eu", 1) == 1
            ):
                candidate = copy.deepcopy(config)
                candidate.kwargs["ASYNC_LOAD"] = True
                extra.append(candidate)
        configs.extend(extra)
    return configs


def _hygon_mm_default_base_prune(configs, named_args, **kwargs):
    """Shortlist Default's base tiles by output coverage and problem size."""
    M = int(named_args.get("M", 0))
    N = int(named_args.get("N", 0))
    K = int(named_args.get("K", 0))
    masked_family = kwargs.get("MASK_EDGES", named_args.get("MASK_EDGES", False)) and kwargs.get(
        "LARGE_TILE_CONFIGS", named_args.get("LARGE_TILE_CONFIGS", False)
    )
    valid = [
        config
        for config in configs
        if _hygon_mm_output_tile_fits(
            M,
            N,
            int(config.kwargs.get("BLOCK_M", 1)),
            int(config.kwargs.get("BLOCK_N", 1)),
            allow_tail=masked_family,
        )
    ]
    candidates = valid or list(configs)
    if M < 16:
        floor_m = 1 << max(0, M.bit_length() - 1)
        ceil_m = 1 << max(0, (M - 1).bit_length())
        allowed_m = {floor_m, min(16, ceil_m)}
    else:
        allowed_m = {16, 32, 64}
    if N < 32:
        allowed_n = {16, 32}
    elif N < 128:
        allowed_n = {32, 64, 128}
    else:
        allowed_n = {64, 128}
    if K <= 1024:
        allowed_k = {32, 64, 128}
    elif K <= 4096:
        allowed_k = {64, 128}
    else:
        allowed_k = {128}
    available_m = {int(c.kwargs.get("BLOCK_M", 0)) for c in candidates}
    if not allowed_m.intersection(available_m):
        # Ragged dimensions such as M=100 have no 16/32/64 divisor. Keep the
        # largest available smaller tile instead of reopening the full space.
        allowed_m = {max(available_m)} if available_m else allowed_m
    narrowed = [
        config
        for config in candidates
        if int(config.kwargs.get("BLOCK_M", 0)) in allowed_m
        and int(config.kwargs.get("BLOCK_N", 0)) in allowed_n
        and int(config.kwargs.get("BLOCK_K", 0)) in allowed_k
        and int(getattr(config, "num_stages", 1)) == 1
        and int(getattr(config, "num_warps", 1))
        == (2 if M * N < 4096 else 4)
    ]
    return narrowed or candidates


def _hygon_mm_default_prune(configs, named_args, **kwargs):
    """Apply the Default shortlist and admit eligible dense and single-wave tiles."""
    base_configs = [
        config
        for config in configs
        if not _hygon_mm_is_dense_config(config)
        and not _hygon_mm_is_single_wave_config(config)
    ]
    candidates = _hygon_mm_default_base_prune(base_configs, named_args, **kwargs)
    if kwargs.get("LARGE_TILE_CONFIGS", named_args.get("LARGE_TILE_CONFIGS", False)):
        M, N = int(named_args["M"]), int(named_args["N"])
        masked = kwargs.get("MASK_EDGES", named_args.get("MASK_EDGES", False))
        load_mask = kwargs.get("ASYNC_LOAD_MASK", named_args.get("ASYNC_LOAD_MASK", 0))
        candidates.extend(
            config
            for config in configs
            if _hygon_mm_is_dense_config(config)
            and (not config.kwargs.get("ASYNC_LOAD", False) or (not masked and load_mask == 3))
            and (masked or config.kwargs.get("waves_per_eu", 1) == 1)
            and (
                (
                    masked
                    and config.kwargs.get("mmac_transposed", False)
                    and config.kwargs["BLOCK_M"] <= 2 * M
                    and config.kwargs["BLOCK_N"] <= 2 * N
                )
                or (
                    not masked
                    and _hygon_mm_output_tile_fits(
                        M, N, config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]
                    )
                )
            )
        )
        # Additional choices for a broad small dense family; retain all old
        # candidates and let the normal tuner decide between them.
        K = int(named_args["K"])
        if (
            64 <= M <= 1024
            and 128 <= N <= 1024
            and 65536 <= M * N <= 262144
            and K >= 128
        ):
            candidates.extend(
                config
                for config in configs
                if _hygon_mm_is_single_wave_config(config)
                and (
                    masked
                    or (
                        M % config.kwargs["BLOCK_M"] == 0
                        and N % config.kwargs["BLOCK_N"] == 0
                    )
                )
            )
    return candidates


def _hygon_mm_edge_prune(configs, named_args, **kwargs):
    """Select the pruning policy for the effective autotune config table."""
    if configs and "LOAD_MODE" in configs[0].kwargs:
        return _hygon_mm_expanded_prune(configs, named_args, **kwargs)
    return _hygon_mm_default_prune(configs, named_args, **kwargs)


_HYGON_MM_DEFAULT_CONFIGS = _hygon_mm_configs()


@libentry(cache_miss_fallback=capture_gemm_fallback)
@libtuner(
    configs=_HYGON_MM_DEFAULT_CONFIGS,
    key=["M", "N", "K", "stride_am", "stride_bk", "LARGE_TILE_CONFIGS",
         "ASYNC_LOAD_MASK", "K_DIVISIBLE_BY_16", "B_K_CONTIGUOUS"],
    strategy=[
        _hygon_mm_tile_bucket,
        _hygon_mm_tile_bucket,
        _hygon_mm_bucket,
        "align32",
        "align32",
        "default",
        "default",
        "default",
        "default",
    ],
    warmup=5,
    rep=10,
    benchmark_mode="replay",
    replay_refinement=(25, 100),
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_hygon",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    prune_configs_by={"early_config_prune": _hygon_mm_edge_prune},
)
@triton.jit(do_not_specialize=["K"])
def mm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    IS_FP64: tl.constexpr = False,
    MASK_EDGES: tl.constexpr = False,
    USE_TLE: tl.constexpr = False,
    LARGE_TILE_CONFIGS: tl.constexpr = False,
    ASYNC_LOAD: tl.constexpr = False,
    LOAD_MODE: tl.constexpr = -1,
    ASYNC_LOAD_MASK: tl.constexpr = 0,
    K_DIVISIBLE_BY_16: tl.constexpr = False,
    LOOP_UNROLL_FACTOR: tl.constexpr = 1,
    B_K_CONTIGUOUS: tl.constexpr = False,
    USE_I32_INDEX: tl.constexpr = False,
    WIDE_INDEX: tl.constexpr = False,
):
    # Preserve the wrapper's K alignment proof independently of async-load
    # capability so synchronous boundary-checked loads retain vector accesses.
    tl.static_assert(LOAD_MODE == -1 or (LOAD_MODE >= 0 and LOAD_MODE <= 3))
    tl.static_assert(LOAD_MODE == -1 or (LOAD_MODE & ~ASYNC_LOAD_MASK) == 0)
    async_a: tl.constexpr = ASYNC_LOAD if LOAD_MODE == -1 else (LOAD_MODE & 1) != 0
    async_b: tl.constexpr = ASYNC_LOAD if LOAD_MODE == -1 else (LOAD_MODE & 2) != 0
    if K_DIVISIBLE_BY_16:
        k_bound = tl.multiple_of(tl.cast(K, tl.int64), 16)
    else:
        k_bound = tl.cast(K, tl.int64)
    pid = ext.program_id(0)
    if WIDE_INDEX:
        M = M.to(tl.int64)
        N = N.to(tl.int64)
    elif USE_I32_INDEX:
        pid = pid.to(tl.int32)

    # --------------------------
    # match naming: num_pid_m, num_pid_n
    # --------------------------
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # reorder for L2
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    group_size_m = min(num_pid_m - group_id * GROUP_M, GROUP_M)

    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # --------------------------
    # match naming: offs_am, offs_bn, offs_k
    # --------------------------
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # contiguous aligned offsets (ram/rbn → offs_am/offs_bn)
    if MASK_EDGES:
        # Keep raw offsets for partial tiles. The unmasked path retains the
        # alignment hints used by the full-tile HCU MMAC path.
        offs_am_cont = offs_am
        offs_bn_cont = offs_bn
    else:
        offs_am_cont = tl.max_contiguous(tl.multiple_of(offs_am % M, BLOCK_M), BLOCK_M)
        offs_bn_cont = tl.max_contiguous(tl.multiple_of(offs_bn % N, BLOCK_N), BLOCK_N)

    # Promote before ceil-div and padding so rk and its stride products are
    # wide on the explicit-pointer path. Block pointers already use i64 strides.
    prev_k_mult = tl.cdiv(k_bound if WIDE_INDEX else K, BLOCK_K) * BLOCK_K
    if not USE_TLE:
        prev_k_mult = tl.maximum(prev_k_mult - BLOCK_K, 0)

    # accumulator
    if IS_FP64:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Keep A in block-pointer form: DTK Clang18 can lose byte-offset scaling
    # for explicit masked A pointers in pipelined gfx936 MM. The boundary
    # checks retain tails for both descriptors.
    if USE_TLE:
        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=(M, k_bound),
            strides=(stride_am, stride_ak),
            offsets=((pid_m * BLOCK_M).to(tl.int32), 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        b_block_ptr = tl.make_block_ptr(
            base=b_ptr,
            shape=(k_bound, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, (pid_n * BLOCK_N).to(tl.int32)),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(0, 1) if B_K_CONTIGUOUS else (1, 0),
        )

    # --------------------------
    # main K loop
    # --------------------------
    for start_k in tl.range(
        0, prev_k_mult, BLOCK_K, loop_unroll_factor=LOOP_UNROLL_FACTOR
    ):
        rk = start_k + offs_k

        if USE_TLE:
            # Keep async input lifetimes and K-tail padding in one loop.
            a = tle.load(
                a_block_ptr, boundary_check=(0, 1), padding_option="zero",
                is_async=async_a,
            )
            b = tle.load(
                b_block_ptr, boundary_check=(0, 1), padding_option="zero",
                is_async=async_b,
            )
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
        elif MASK_EDGES:
            a = tl.load(
                a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
                mask=(offs_am_cont < M)[:, None],
                other=0.0,
            )
            b = tl.load(
                b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
                mask=(offs_bn_cont < N)[None, :],
                other=0.0,
            )
        else:
            a = tl.load(
                a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak)
            )
            b = tl.load(
                b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn)
            )

        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)

        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        elif USE_TLE:
            accumulator = tl.dot(
                a, b, accumulator, out_dtype=tl.float32, allow_tf32=False
            )
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # --------------------------
    # loop peel
    # --------------------------
    if not USE_TLE:
        rk = prev_k_mult + offs_k
        mask_k = rk < K

        a = tl.load(
            a_ptr + (offs_am_cont[:, None] * stride_am + rk[None, :] * stride_ak),
            mask=mask_k[None, :] & ((offs_am_cont < M)[:, None] if MASK_EDGES else True),
            other=0.0,
        )
        b = tl.load(
            b_ptr + (rk[:, None] * stride_bk + offs_bn_cont[None, :] * stride_bn),
            mask=mask_k[:, None] & ((offs_bn_cont < N)[None, :] if MASK_EDGES else True),
            other=0.0,
        )

        if a.dtype != b.dtype:
            a = a.to(c_ptr.dtype.element_ty)
            b = b.to(c_ptr.dtype.element_ty)

        if IS_FP64:
            accumulator += tl.dot(a, b, out_dtype=tl.float64, allow_tf32=False)
        else:
            accumulator += tl.dot(a, b, out_dtype=tl.float32, allow_tf32=False)

    # cast to output dtype
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # --------------------------
    # rematerialize offsets for store
    # (match naming: offs_cm, offs_cn)
    # --------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    c_ptr = c_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    mask_store = (offs_cm < M)[:, None] & (offs_cn < N)[None, :]

    if LARGE_TILE_CONFIGS and not MASK_EDGES and BLOCK_M == 256 and BLOCK_N == 256:
        mask_store = None
    tl.store(c_ptr, accumulator, mask=mask_store)


# Cooperative producer/consumer GEMM.

@triton.jit
def grouped_program_coordinates(
    pid, count_m: tl.constexpr, count_n: tl.constexpr, GROUP_M: tl.constexpr
):
    # The caller launches count_m * count_n consecutive program IDs.
    tl.static_assert(count_m > 0 and count_n > 0 and GROUP_M > 0)
    # Program IDs in the supported launch grid are nonnegative.
    pid = pid.to(tl.uint32)
    group_width: tl.constexpr = GROUP_M * count_n
    if count_n == 1:
        # Keep rotated tail-row ordering; the single column is always zero.
        if count_m > GROUP_M and count_m % GROUP_M != 0:
            first_tail: tl.constexpr = (count_m // GROUP_M) * GROUP_M
            tail_rows: tl.constexpr = count_m % GROUP_M
            pm = tl.where(pid >= first_tail, first_tail + pid % tail_rows, pid)
            return pm.to(tl.int32), tl.cast(0, tl.int32)
        return pid.to(tl.int32), tl.cast(0, tl.int32)
    elif count_m <= GROUP_M:
        return (pid % count_m).to(tl.int32), (pid // count_m).to(tl.int32)
    elif count_m % GROUP_M == 0:
        return ((pid // group_width) * GROUP_M + pid % GROUP_M).to(tl.int32), (
            (pid % group_width) // GROUP_M
        ).to(tl.int32)
    else:
        full_groups: tl.constexpr = count_m // GROUP_M
        full_span: tl.constexpr = full_groups * group_width
        tail_m: tl.constexpr = count_m % GROUP_M
        in_tail = pid >= full_span
        full_m = (pid // group_width) * GROUP_M + pid % GROUP_M
        full_n = (pid % group_width) // GROUP_M
        # Preserve the original absolute modulo, including rotated tail rows.
        tail_m_index = full_groups * GROUP_M + pid % tail_m
        tail_n = (pid - full_span) // tail_m
        return tl.where(in_tail, tail_m_index, full_m).to(tl.int32), tl.where(
            in_tail, tail_n, full_n
        ).to(tl.int32)


@triton.jit
def _coordinates(M: tl.constexpr, N: tl.constexpr, GROUP_M: tl.constexpr):
    if GROUP_M == 0:
        return tl.program_id(0), tl.program_id(1)
    else:
        cm: tl.constexpr = (M + 255) // 256
        cn: tl.constexpr = (N + 255) // 256
        return grouped_program_coordinates(
            tl.program_id(0) + tl.program_id(1) * cm, cm, cn, GROUP_M
        )


@triton.jit
def _produce_tile(
    B,
    owner,
    n,
    k,
    t,
    N: tl.constexpr,
    K: tl.constexpr,
    BK: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
):
    column = t * BK + k
    mask = n[None, :] < N
    if B_K_CONTIGUOUS:
        pointer = B + n[None, :].to(tl.int64) * K + column[:, None]
    else:
        pointer = B + column[:, None].to(tl.int64) * N + n[None, :]
    value = tle.load(pointer, mask, other=0, is_async=True)
    owner.slot(t % 2).store(value)
    cta_phase()
    cta_phase()


@triton.jit
def _producer(
    B,
    owner,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GROUP_M: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
):
    pm, pn = _coordinates(M, N, GROUP_M)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    for pair in range(K // (2 * BK)):
        _produce_tile(B, owner, n, k, pair * 2, N, K, BK, B_K_CONTIGUOUS)
        _produce_tile(B, owner, n, k, pair * 2 + 1, N, K, BK, B_K_CONTIGUOUS)
    if (K // BK) % 2:
        _produce_tile(B, owner, n, k, K // BK - 1, N, K, BK, B_K_CONTIGUOUS)
    cta_phase()
    cta_phase()


@triton.jit
def _load_parts(
    A, row, K: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, M: tl.constexpr, t
):
    pointer = tl.make_block_ptr(A, (M, K), (K, 1), (row, t * BK), (BM, BK), (1, 0))
    full = tl.load(pointer, boundary_check=(0, 1), padding_option="zero")
    lo, hi = tl.split(full.reshape((BM, 2, BK // 2)).permute((0, 2, 1)))
    return lo, hi


@triton.jit
def _read_cooperative_b(
    owner, slot, start: tl.constexpr, half_k: tl.constexpr, BN: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
):
    values = owner.slot(slot).subslice(start, half_k, 0).load()
    if not B_K_CONTIGUOUS:
        # Present the N-contiguous producer's shared tile in MMAC fragment order.
        values = values.reshape((half_k, BN // 32, 16, 2)).permute((0, 1, 3, 2))
        values = values.reshape((half_k, BN))
    return values


@triton.jit
def _compute_parts(
    A,
    owner,
    acc,
    av0,
    av1,
    row,
    t,
    M: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    HIGH_PHASE: tl.constexpr,
    BN: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
):
    bv0 = _read_cooperative_b(owner, t % 2, 0, BK // 2, BN, B_K_CONTIGUOUS)
    if HIGH_PHASE:
        cta_phase()
    acc = accumulator_fence(acc, 1)
    acc = tl.dot(av0, bv0, acc, allow_tf32=False)
    acc = accumulator_fence(acc, 0)
    if not HIGH_PHASE:
        cta_phase()
    next0, next1 = _load_parts(A, row, K, BM, BK, M, t + 1)
    bv1 = _read_cooperative_b(owner, t % 2, BK // 2, BK // 2, BN, B_K_CONTIGUOUS)
    if HIGH_PHASE:
        cta_phase()
    acc = accumulator_fence(acc, 1)
    acc = tl.dot(av1, bv1, acc, allow_tf32=False)
    acc = accumulator_fence(acc, 0)
    if not HIGH_PHASE:
        cta_phase()
    return acc, next0, next1


@triton.jit
def _consumer(
    A,
    C,
    owner,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ROW_START: tl.constexpr,
    HIGH_PHASE: tl.constexpr,
    GROUP_M: tl.constexpr,
    B_K_CONTIGUOUS: tl.constexpr,
):
    pm, pn = _coordinates(M, N, GROUP_M)
    row = pm * (2 * BM) + ROW_START
    acc = tl.zeros((BM, BN), tl.float32)
    av0, av1 = _load_parts(A, row, K, BM, BK, M, 0)
    cta_phase()
    cta_phase()
    for pair in range(tl.cdiv(K, BK) // 2):
        acc, av0, av1 = _compute_parts(
            A, owner, acc, av0, av1, row, pair * 2, M, K, BM, BK, HIGH_PHASE, BN, B_K_CONTIGUOUS
        )
        acc, av0, av1 = _compute_parts(
            A, owner, acc, av0, av1, row, pair * 2 + 1, M, K, BM, BK, HIGH_PHASE, BN, B_K_CONTIGUOUS
        )
    if tl.cdiv(K, BK) % 2:
        acc, av0, av1 = _compute_parts(
            A, owner, acc, av0, av1, row, tl.cdiv(K, BK) - 1, M, K, BM, BK, HIGH_PHASE, BN, B_K_CONTIGUOUS
        )
    if B_K_CONTIGUOUS:
        output = tl.make_block_ptr(C, (M, N), (N, 1), (row, pn * BN), (BM, BN), (1, 0))
        tl.store(output, acc.to(C.dtype.element_ty), boundary_check=(0, 1))
    else:
        i = tl.arange(0, BM)
        j = tl.arange(0, BN)
        # Undo the column permutation from the shared read on the destination
        # coordinates while keeping the accumulator in its original lane.
        j = (j & ~31) | ((j & 15) << 1) | ((j >> 4) & 1)
        col = pn.to(tl.int64) * BN
        origin = C + row.to(tl.int64) * N + col
        tl.store(
            origin + i[:, None] * N + j[None, :],
            acc.to(C.dtype.element_ty),
            (row.to(tl.int64) + i[:, None] < M) & (col + j[None, :] < N),
        )


@libentry()
@triton.jit
def _mm_cooperative_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr = 256,
    BN: tl.constexpr = 256,
    BK: tl.constexpr = 64,
    GROUP_M: tl.constexpr = 0,
    B_K_CONTIGUOUS: tl.constexpr = True,
):
    tl.static_assert(BM == 256 and BN == 256 and BK == 64)
    tl.static_assert(K >= 64 and K < 8192 and K % 64 == 0)
    tl.static_assert(M > 0 and N > 0 and M < 2147483648 and N < 2147483648)
    tl.static_assert(512 * N < 2147483648 and GROUP_M >= 0)
    # Describe logical storage; the backend jointly selects its physical chart
    # from the global producer and actual MMAC consumers.
    if B_K_CONTIGUOUS:
        owner = tle.gpu.alloc([2, BK, BN], A.dtype.element_ty)
    else:
        layout: tl.constexpr = tle.gpu.swizzled_shared_layout(
            32, 1, 4, [2, 1, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 1, 0]
        )
        owner = tle.gpu.alloc([2, BK, BN], A.dtype.element_ty, layout=layout)
    tle.gpu.warp_specialize(
        [
            (_consumer, (A, C, owner, M, N, K, BM // 2, BN, BK, 0, False, GROUP_M, B_K_CONTIGUOUS)),
            (
                _consumer,
                (A, C, owner, M, N, K, BM // 2, BN, BK, BM // 2, True, GROUP_M, B_K_CONTIGUOUS),
            ),
            (
                _producer,
                (B, owner, M, N, K, BN, BK, GROUP_M, B_K_CONTIGUOUS),
            ),
        ],
        [4, 4],
        [128, 32],
    )


# Independent CTA-local Split-K GEMM.

@triton.jit
def _write_partial_sum(
    accumulator,
    C,
    mi,
    ni,
    M: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    pairs = tl.reshape(tl.permute(accumulator, (1, 2, 0)), (BM * BN, 2, 2))
    even, odd = tl.split(pairs)
    low, high = tl.split(even + odd)
    value = tl.reshape(low + high, (BM, BN))
    tl.store(
        C + mi[:, None].to(tl.int64) * N + ni[None, :],
        value.to(C.dtype.element_ty),
        (mi[:, None] < M) & (ni[None, :] < N),
    )


@libentry()
@triton.jit
def _mm_local_splitk_kernel(
    A,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    K_STEP: tl.constexpr = 256,
    B_ROW_MAJOR: tl.constexpr = False,
    M_TILES: tl.constexpr = 1,
):
    tl.static_assert(K_STEP == 128 or K_STEP == 256)
    tl.static_assert(M_TILES > 0 and BM >= 16 and BN >= 16)
    row = tl.program_id(0) * (M_TILES * BM)
    col = tl.program_id(1) * BN
    part = tl.arange(0, 4)
    offsets = tl.arange(0, BM)
    ni = col + tl.arange(0, BN)
    ki = part[:, None] * (K_STEP // 4) + tl.arange(0, K_STEP // 4)[None, :]
    accumulators = ()
    for i in tl.static_range(M_TILES):
        accumulators = accumulators + (tl.zeros((4, BM, BN), tl.float32),)
    for start in range(tl.cdiv(K, K_STEP)):
        kk = start * K_STEP + ki
        if B_ROW_MAJOR:
            bv = tl.load(
                B + kk[:, :, None].to(tl.int64) * N + ni[None, None, :],
                (kk[:, :, None] < K) & (ni[None, None, :] < N),
                other=0,
            )
        else:
            bv = tl.load(
                B + ni[None, None, :].to(tl.int64) * K + kk[:, :, None],
                (kk[:, :, None] < K) & (ni[None, None, :] < N),
                other=0,
            )
        next_accumulators = ()
        for i in tl.static_range(M_TILES):
            mi = row + i * BM + offsets
            av = tl.load(
                A + mi[None, :, None].to(tl.int64) * K + kk[:, None, :],
                (mi[None, :, None] < M) & (kk[:, None, :] < K),
                other=0,
            )
            next_accumulators = next_accumulators + (
                tl.dot(av, bv, accumulators[i], allow_tf32=False),
            )
        accumulators = next_accumulators
    for i in tl.static_range(M_TILES):
        _write_partial_sum(accumulators[i], C, row + i * BM + offsets, ni, M, N, BM, BN)


# Tuned Split-K partials and their optional global reducer.

try:
    from triton.backends.hcu.compiler import HIPOptions
except ImportError:
    _SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED = False
else:
    _SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED = (
        "wave_private_sync" in HIPOptions.__dataclass_fields__
    )

# Split-K partials and the reducer use sections in the same Expanded YAML
# as the ordinary MM kernel. Separate section names keep the tuning keys
# and launch parameters specific to each kernel.


@libentry(cache_miss_fallback=capture_gemm_fallback)
@libtuner(
    configs=[triton.Config(dict(BLOCK=256), num_warps=4, num_stages=1)],
    key=["MN", "SPLITS"],
    strategy=["default"] * 2,
    warmup=5,
    rep=10,
    benchmark_mode="replay",
    replay_refinement=(25, 100),
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_hygon_splitk_post",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
)
@triton.jit
def _mm_splitk_post(W, C, MN: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), tl.float32)
    for split in tl.static_range(SPLITS):
        acc += tl.load(W + split * MN + i, i < MN, other=0.)
    tl.store(C + i, acc, i < MN)


def _splitk_post_hook(args, exception):
    # Triton invokes this inside its timed kernel_call. The final launch and
    # LibEntry cache hits are completed by the public wrapper below.
    if exception is None and args["KIND"] == 0 and args["SPLITS"] > 1:
        mn = args["M"] * args["N"]
        _mm_splitk_post[lambda meta: (triton.cdiv(mn, meta["BLOCK"]),)](
            args["W"], args["C"], mn, args["SPLITS"],
        )


def _splitk_default_configs():
    # Retain the two existing synchronous global-split families for Default.
    # Expanded replaces this list with the complete declared YAML space.
    return [
        triton.Config(
            dict(BM=32, BN=bn, BK=bk, SPLITS=splits, KIND=0,
                 LOAD_MODE=0, optimize_epilogue=False,
                 schedule_hint="local-prefetch"),
            num_warps=2, num_stages=2,
        )
        for bn, bk, splits in [(16, 64, 32), (32, 32, 16)]
    ]


def _splitk_prune(configs, named_args, **kwargs):
    mask = int(kwargs.get("ASYNC_LOAD_MASK", named_args.get("ASYNC_LOAD_MASK", 0)))
    # Tail tiles and shallow per-split loops remain eligible. Resource limits
    # are checked by the compiler/launcher, not an estimated performance model.
    supports_private = bool(kwargs.get(
        "WAVE_PRIVATE_SYNC_SUPPORTED",
        named_args.get("WAVE_PRIVATE_SYNC_SUPPORTED", False),
    ))
    selected = []
    for config in configs:
        if config.kwargs["LOAD_MODE"] & ~mask:
            continue
        if not supports_private and config.kwargs.get("wave_private_sync", False):
            continue
        if not supports_private and "wave_private_sync" in config.kwargs:
            # Older HCU backends must not receive an unknown backend option.
            config = copy.deepcopy(config)
            config.kwargs.pop("wave_private_sync")
        selected.append(config)
    return selected


@libentry(cache_miss_fallback=capture_gemm_fallback)
@libtuner(
    configs=_splitk_default_configs(),
    key=["M", "N", "K", "stride_bk", "stride_bn", "B_K_CONTIGUOUS",
         "ASYNC_LOAD_MASK", "WAVE_PRIVATE_SYNC_SUPPORTED", "M_ALIGNMENT", "N_ALIGNMENT"],
    strategy=["default"] * 10,
    warmup=5,
    rep=10,
    benchmark_mode="replay",
    replay_refinement=(25, 100),
    post_hook=_splitk_post_hook,
    flagtune_op_name="mm",
    flagtune_expand_op_name="mm_hygon_splitk",
    flagtune_yaml_path=EXPAND_CONFIG_FILENAME,
    prune_configs_by={"early_config_prune": _splitk_prune},
)
@triton.jit
def _mm_splitk_partial(A, B, W, C, M, N, K, stride_bk, stride_bn,
            B_K_CONTIGUOUS: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
            SPLITS: tl.constexpr, KIND: tl.constexpr,
            LOAD_MODE: tl.constexpr, ASYNC_LOAD_MASK: tl.constexpr,
            WAVE_PRIVATE_SYNC_SUPPORTED: tl.constexpr,
            M_ALIGNMENT: tl.constexpr = 1, N_ALIGNMENT: tl.constexpr = 1):
    tl.static_assert(KIND == 0 or KIND == 1)
    tl.static_assert(LOAD_MODE >= 0 and LOAD_MODE <= 3)
    tl.static_assert((LOAD_MODE & ~ASYNC_LOAD_MASK) == 0)
    # Host launch contract: exact ceil-grid, positive M/N, facts divide M/N.
    tl.static_assert(M_ALIGNMENT > 0 and (M_ALIGNMENT & (M_ALIGNMENT - 1)) == 0)
    tl.static_assert(N_ALIGNMENT > 0 and (N_ALIGNMENT & (N_ALIGNMENT - 1)) == 0)
    FULL_M: tl.constexpr = M_ALIGNMENT % BM == 0
    FULL_N: tl.constexpr = N_ALIGNMENT % BN == 0
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    kr = tl.arange(0, BK)
    if KIND == 0:
        split = tl.program_id(2)
        # Match the existing production split-K interleaving exactly. Keep
        # both loads as block pointers to avoid the known DTK masked-pointer
        # byte-offset problem.
        ap = tl.make_block_ptr(A, (M, K), (K, 1),
                              (tl.program_id(0) * BM, split * BK),
                              (BM, BK), (1, 0))
        if B_K_CONTIGUOUS:
            bp = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn),
                                  (split * BK, tl.program_id(1) * BN),
                                  (BK, BN), (0, 1))
        else:
            bp = tl.make_block_ptr(B, (K, N), (stride_bk, stride_bn),
                                  (split * BK, tl.program_id(1) * BN),
                                  (BK, BN), (1, 0))
        acc = tl.zeros((BM, BN), tl.float32)
        for q in range(tl.cdiv(K, BK * SPLITS)):
            if FULL_M:
                a = tle.load(ap, boundary_check=(1,), padding_option='zero',
                             is_async=(LOAD_MODE & 1) != 0)
            else:
                a = tle.load(ap, boundary_check=(0, 1), padding_option='zero',
                             is_async=(LOAD_MODE & 1) != 0)
            if FULL_N:
                b = tle.load(bp, boundary_check=(0,), padding_option='zero',
                             is_async=(LOAD_MODE & 2) != 0)
            else:
                b = tle.load(bp, boundary_check=(0, 1), padding_option='zero',
                             is_async=(LOAD_MODE & 2) != 0)
            acc = tl.dot(a, b, acc)
            ap = tl.advance(ap, (0, BK * SPLITS))
            bp = tl.advance(bp, (BK * SPLITS, 0))
        if SPLITS == 1:
            out = C + rm[:, None] * N + rn[None, :]
        else:
            out = W + split * M * N + rm[:, None] * N + rn[None, :]
        if FULL_M and FULL_N:
            tl.store(out, acc)
        elif FULL_M:
            tl.store(out, acc, rn[None, :] < N)
        elif FULL_N:
            tl.store(out, acc, rm[:, None] < M)
        else:
            tl.store(out, acc, (rm[:, None] < M) & (rn[None, :] < N))
    else:
        # Express CTA-internal K partitioning with existing batched dot and
        # reduction primitives. Inspect actual warp distribution before
        # calling this the same parallel algorithm as Torch.
        sp = tl.arange(0, SPLITS)
        acc = tl.zeros((SPLITS, BM, BN), tl.float32)
        for q in range(tl.cdiv(K, BK * SPLITS)):
            kk = (sp[:, None] + q * SPLITS) * BK + kr[None, :]
            if FULL_M:
                a = tle.load(A + rm[None, :, None] * K + kk[:, None, :],
                             kk[:, None, :] < K,
                             other=0., is_async=(LOAD_MODE & 1) != 0)
            else:
                a = tle.load(A + rm[None, :, None] * K + kk[:, None, :],
                             (rm[None, :, None] < M) & (kk[:, None, :] < K),
                             other=0., is_async=(LOAD_MODE & 1) != 0)
            if FULL_N:
                b = tle.load(B + kk[:, :, None] * stride_bk
                             + rn[None, None, :] * stride_bn,
                             kk[:, :, None] < K,
                             other=0., is_async=(LOAD_MODE & 2) != 0)
            else:
                b = tle.load(B + kk[:, :, None] * stride_bk
                             + rn[None, None, :] * stride_bn,
                             (kk[:, :, None] < K) & (rn[None, None, :] < N),
                             other=0., is_async=(LOAD_MODE & 2) != 0)
            acc = tl.dot(a, b, acc)
        result = tl.sum(acc, 0)
        if FULL_M and FULL_N:
            tl.store(C + rm[:, None] * N + rn[None, :], result)
        elif FULL_M:
            tl.store(C + rm[:, None] * N + rn[None, :], result, rn[None, :] < N)
        elif FULL_N:
            tl.store(C + rm[:, None] * N + rn[None, :], result, rm[:, None] < M)
        else:
            tl.store(C + rm[:, None] * N + rn[None, :], result,
                     (rm[:, None] < M) & (rn[None, :] < N))


def _splitk_eligible(a, b, c):
    if not (HAS_HYGON_TLE and _HYGON_SPLITK_OPTIONS_SUPPORTED):
        return False
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        return False
    M, K = a.shape
    N = b.shape[1]
    if not (
        M >= 2 and N >= 2
        # K partitions supply parallel work for small output grids. Limit
        # the grid to 64 tiles and the tuning workspace to 16MiB.
        and triton.cdiv(M, 32) * triton.cdiv(N, 32) <= 64
        and K >= (8192 if N < 16 else 2048)
        and b.shape[0] == K
        and c.shape == (M, N)
        and a.dtype == b.dtype == c.dtype
        and a.dtype in (torch.bfloat16, torch.float16)
        and a.device == b.device == c.device
        and a.device.type == "cuda"
        and not c.requires_grad
        and (not c.is_inference() or torch.is_inference_mode_enabled())
        and c.untyped_storage().data_ptr()
        not in (a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        and a.is_contiguous() and c.is_contiguous()
        and (b.is_contiguous() or (
            b.stride(0) == 1 and b.stride(1) == K and b.t().is_contiguous()
        ))
        and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
        and all(t.untyped_storage().nbytes() <= 2**31 - 1 for t in (a, b, c))
    ):
        return False
    from triton import knobs

    if not (
        knobs.hcu.use_buffer_ops
        and not knobs.hcu.use_async_copy
        and not knobs.hcu.use_block_pingpong
        and not knobs.hcu.use_in_thread_transpose
    ):
        return False
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx936"


def _splitk_alignment_facts(M, N):
    """Divisibility facts for the exact ceil-grid used by try_mm_splitk."""
    if any(type(x) is not int or not 0 < x <= 2**31 - 1 for x in (M, N)):
        return 1, 1
    return M & -M, N & -N


def try_mm_splitk(a, b, c, *, compare_local_split=False):
    """Write c using the small-output family; return False outside its domain."""
    if not _splitk_eligible(a, b, c):
        return False
    if compare_local_split:
        _mm_splitk_partial._apply_flagtune()
        tuner = _mm_splitk_partial.fn
        if tuner._run_mode == "normal":
            key = gemm_tuning_key(tuner)
            if try_mm_family(
                a,
                b,
                c,
                lambda: _launch_mm_splitk(a, b, c),
                tuning_key=key,
                ordinary_with_output=lambda output: _launch_mm_splitk(a, b, output),
                splitk_pool=True,
            ):
                return True
    _launch_mm_splitk(a, b, c)
    return True


def _launch_mm_splitk(a, b, c):
    M, K = a.shape
    N = b.shape[1]
    ma, na = _splitk_alignment_facts(M, N)
    # Facts below are valid only for this exact ceil-grid and its original
    # pid axes. Arbitrary caller grids must retain the original checks.
    # All Expanded configurations share one workspace. CTA/GSU1 leave it
    # unused; every global split writes its complete disjoint FP32 partial.
    workspace = torch.empty((64, M, N), device=a.device, dtype=torch.float32)
    with torch_device_fn.device(a.device):
        # Preserve the qualified Split-K load modes while ordinary MM is
        # validated with the per-operand layout contract.
        mask = _hygon_mm_async_load_mask(a, b, c, M, N, K) if b.is_contiguous() else 0
        grid = lambda meta: (
            triton.cdiv(M, meta["BM"]), triton.cdiv(N, meta["BN"]),
            meta["SPLITS"] if meta["KIND"] == 0 else 1,
        )
        _, meta = _mm_splitk_partial[grid](
            a, b, workspace, c, M, N, K, *b.stride(),
            B_K_CONTIGUOUS=b.stride(0) == 1, ASYNC_LOAD_MASK=mask,
            WAVE_PRIVATE_SYNC_SUPPORTED=_SPLITK_WAVE_PRIVATE_SYNC_SUPPORTED,
            M_ALIGNMENT=ma, N_ALIGNMENT=na,
        )
        # LibEntry returns the metadata for this exact launch, including on
        # cache hits; do not read a mutable global tuner's best_config.
        _splitk_post_hook(dict(meta, W=workspace, C=c, M=M, N=N, K=K), None)


# Fixed Shared and short-K profiles for the ordinary kernel.


@dataclass(frozen=True)
class SharedMMConfig:
    tile: tuple
    warps: tuple = (2, 2)
    group_m: int = 8
    address_mode: str = "direct"

    def __post_init__(self):
        if len(self.tile) != 3 or any(x < 8 or x & (x - 1) for x in self.tile):
            raise ValueError("tile dimensions must be powers of two >= 8")
        bm, bn, bk = self.tile
        if max(bm * bk, bn * bk) > 4096:
            raise ValueError("operand tiles must fit the shared slot")
        if (
            len(self.warps) != 2
            or any(x < 1 or x & (x - 1) for x in self.warps)
            or prod(self.warps) > 8
        ):
            raise ValueError("requires power-of-two wave axes and at most 8 waves")
        if self.group_m < 1:
            raise ValueError("traversal group must be positive")
        if self.address_mode not in ("direct", "fold", "scalar"):
            raise ValueError("unknown buffer-address strategy")

    def supports(self, a, b, c, *, async_load=True):
        """Check vector alignment, storage disjointness and address bounds."""
        if any(t.ndim != 2 for t in (a, b, c)):
            return False
        m, k = a.shape
        n = b.shape[1]
        bm, bn, bk = self.tile
        nt = b.stride() == (1, k)
        local_span = max(
            (bm - 1) * k + bk, (bn - 1) * k + bk if nt else (bk - 1) * n + bn
        )
        output_begin, output_end = c.data_ptr(), c.data_ptr() + 2 * m * n
        disjoint = all(
            output_end <= t.data_ptr() or t.data_ptr() + 2 * count <= output_begin
            for t, count in ((a, m * k), (b, k * n))
        )
        return (
            m > 0
            and n > 0
            and k >= 8
            and k % 8 == 0
            and (not async_load or nt or n % 8 == 0)
            and b.shape[0] == k
            and tuple(c.shape) == (m, n)
            and a.stride() == (k, 1)
            and (nt or b.stride() == (n, 1))
            and c.stride() == (n, 1)
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and disjoint
            and max(m, n, k) < 2**31
            and 2 * m * n < 2**31
            and (2 * local_span < 2**31 or max(2 * m * k, 2 * k * n) < 2**31)
        )


_SHARED_HCU_OPTIONS = dict(
    waves_per_eu=1,
    loop_unroll_stage="post-pipeline",
    schedule_hint="local-prefetch",
    lds_dma_alias_workaround=True,
    prefer_wide_async_copy=True,
    kpack=2,
    mmac_transposed=True,
    optimize_epilogue=True,
    rebase_buffer_store=False,
    terminal_readonly_publication=False,
)
_SHARED_ADDRESS_HCU_OPTIONS = dict(
    coalesce_lgkm_waits=True,
    combine_shifted_disjoint_or=True,
    buffer_store_split_immediate=True,
    buffer_store_preserve_masked_base=True,
)


@cache
def _shared_address_supported():
    return supports_hcu_options(
        (*_SHARED_ADDRESS_HCU_OPTIONS, "buffer_store_scalar_offset")
    )


@cache
def _shared_compile_parameters(config, transposed_b):
    options = {
        **_SHARED_HCU_OPTIONS,
        "num_warps": prod(config.warps),
        "num_stages": 5,
        "mmac_warps_m": config.warps[0],
        "GROUP_M": config.group_m,
        "BLOCK_M": config.tile[0],
        "BLOCK_N": config.tile[1],
        "BLOCK_K": config.tile[2],
        "USE_TLE": True,
        "ASYNC_LOAD": True,
        "ASYNC_LOAD_MASK": 3,
        "LOAD_MODE": 3,
        "LARGE_TILE_CONFIGS": True,
        "LOOP_UNROLL_FACTOR": 4,
        "B_K_CONTIGUOUS": transposed_b,
    }
    if config.address_mode != "direct":
        options.update(
            **_SHARED_ADDRESS_HCU_OPTIONS,
            buffer_store_scalar_offset=config.address_mode == "scalar",
        )
    # Cache immutable items; callers never share a mutable options dictionary.
    return tuple(options.items())


@lru_cache(maxsize=1)
def _ordinary_mm_entry():
    # Fixed family candidates launch the same JIT function without the tuner.
    kernel = mm_kernel
    while not isinstance(kernel, triton.runtime.JITFunction):
        kernel = kernel.fn
    return libentry()(kernel)


def launch_shared_mm(a, b, c, config):
    """Launch one explicit ordinary GEMM candidate through the real LibEntry."""
    if not config.supports(a, b, c):
        raise ValueError("unsupported matrix storage, vector extent or address range")
    m, k = a.shape
    n = b.shape[1]
    bm, bn, _ = config.tile
    tiles_m, tiles_n = triton.cdiv(m, bm), triton.cdiv(n, bn)
    padded_m, padded_n = tiles_m * bm, tiles_n * bn
    options = dict(_shared_compile_parameters(config, b.stride() == (1, k)))
    options.update(
        MASK_EDGES=m % bm != 0 or n % bn != 0,
        K_DIVISIBLE_BY_16=k % 16 == 0,
        # Only tile coordinates and output offsets are narrowed. Block-pointer
        # shapes/strides and global input origins retain their i64 arithmetic.
        USE_I32_INDEX=max(
            tiles_m * tiles_n,
            config.group_m * tiles_n,
            padded_m,
            padded_n,
            2 * ((padded_m - 1) * n + padded_n),
        )
        < 2**31,
    )
    with torch_device_fn.device(a.device):
        return _ordinary_mm_entry()[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
            a,
            b,
            c,
            tl.constexpr(m),
            tl.constexpr(n),
            tl.constexpr(k),
            *(tl.constexpr(s) for s in a.stride()),
            *(tl.constexpr(s) for s in b.stride()),
            *(tl.constexpr(s) for s in c.stride()),
            **options,
        )


@lru_cache(maxsize=1)
def shared_mm_configs():
    """Logical tiles and traversal choices, shared by NN and NT.

    Lane ownership, shared layout, copy instruction and synchronization are
    selected by the compiler. Do not duplicate candidates for those details.
    """
    return (
        SharedMMConfig((64, 128, 32)),
        SharedMMConfig((32, 64, 64)),
        SharedMMConfig((128, 128, 32)),
        SharedMMConfig((128, 256, 16)),
        SharedMMConfig((64, 64, 64)),
        SharedMMConfig((256, 256, 16), warps=(4, 2), group_m=1, address_mode="scalar"),
        SharedMMConfig((256, 256, 16), warps=(4, 2), address_mode="fold"),
        SharedMMConfig((256, 256, 16), warps=(4, 2), group_m=1, address_mode="fold"),
        SharedMMConfig((128, 128, 16), address_mode="scalar"),
        SharedMMConfig((256, 256, 16), warps=(4, 2)),
        SharedMMConfig((128, 256, 16), address_mode="scalar"),
        SharedMMConfig((64, 64, 64), group_m=1),
    )


@dataclass(frozen=True)
class ShortKMMConfig:
    block_k: int


SHORT_K_MM_CONFIGS = (ShortKMMConfig(32), ShortKMMConfig(64))
_SHORT_K_STORAGE = SharedMMConfig((128, 128, 16))
_SHORT_K_HCU_OPTIONS = dict(
    mmac_warps_m=2,
    mmac_transposed=True,
    optimize_epilogue=True,
    schedule_hint="local-prefetch",
    kpack=2,
)


@cache
def _short_k_supported():
    return callable(getattr(tle, "load", None)) and supports_hcu_options(_SHORT_K_HCU_OPTIONS)


def short_k_mm_candidates(a, b, c):
    m, k = a.shape
    n = b.shape[1]
    if (
        min(m, n) < 128 or not 64 <= k <= 256
        or not _short_k_supported()
        or not _SHORT_K_STORAGE.supports(a, b, c, async_load=False)
    ):
        return ()
    compute_units = torch_device_fn.get_device_properties(a.device).multi_processor_count
    if triton.cdiv(m, 128) * triton.cdiv(n, 128) < compute_units:
        return ()
    # Two to four whole K panels can be folded into a straight-line
    # synchronous schedule. The regular tuner remains in the competition.
    return tuple(
        config for config in SHORT_K_MM_CONFIGS
        if k % config.block_k == 0 and 2 <= k // config.block_k <= 4
    )


def launch_short_k_mm(a, b, c, config):
    m, k = a.shape
    n = b.shape[1]
    with torch_device_fn.device(a.device):
        return _ordinary_mm_entry()[(triton.cdiv(m, 128) * triton.cdiv(n, 128),)](
            a, b, c, m, n, tl.constexpr(k), *a.stride(), *b.stride(), *c.stride(),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=config.block_k, GROUP_M=4,
            MASK_EDGES=m % 128 != 0 or n % 128 != 0,
            USE_TLE=True, LOAD_MODE=0, K_DIVISIBLE_BY_16=True,
            B_K_CONTIGUOUS=b.stride() == (1, k), LOOP_UNROLL_FACTOR=4,
            num_warps=4, num_stages=2, **_SHORT_K_HCU_OPTIONS,
        )


# Cooperative kernel capability, configurations and launch.

_COOPERATIVE_HCU_OPTIONS = dict(
    prefer_wide_async_copy=True,
    kpack=2,
    mmac_warps_m=4,
    mmac_transposed=True,
    disable_llvm_loop_unroll=True,
    structured_register_prepack_address=True,
    structured_register_unclamped_column=True,
    priority_after_first_mmac=True,
    buffer_store_scalar_offset=True,
    interleave_global_lds_reads=True,
    interleave_global_lds_reads_fences=True,
    tupled_register_fences=True,
)


@lru_cache(maxsize=1)
def _cooperative_supported():
    try:
        from triton._C.libtriton import hcu
    except ImportError:
        return False
    gpu = getattr(tle, "gpu", None)
    return (
        supports_hcu_options(_COOPERATIVE_HCU_OPTIONS)
        and all(
            callable(op)
            for op in (
                cta_phase,
                accumulator_fence,
                getattr(gpu, "alloc", None),
                getattr(gpu, "warp_specialize", None),
                getattr(tle, "load", None),
            )
        )
        and all(
            callable(getattr(hcu, name, None))
            for name in (
                "create_cta_barrier",
                "has_warp_specialize",
                "create_accumulator_fence",
            )
        )
    )


@dataclass(frozen=True)
class CooperativeMMConfig:
    group_m: int = 0

    def __post_init__(self):
        if type(self.group_m) is not int or self.group_m < 0:
            raise ValueError("CTA grouping must be a nonnegative integer")

    def supports(self, a, b, c):
        if not _cooperative_supported() or any(t.ndim != 2 for t in (a, b, c)):
            return False
        m, k = a.shape
        n = b.shape[1]
        begin, end = c.data_ptr(), c.data_ptr() + 2 * m * n
        disjoint = all(
            end <= t.data_ptr() or t.data_ptr() + 2 * count <= begin
            for t, count in ((a, m * k), (b, k * n))
        )
        return (
            m > 0
            and n > 0
            and 64 <= k < 8192
            and k % 64 == 0
            and b.shape[0] == k
            and c.shape == (m, n)
            and a.stride() == (k, 1)
            and (
                b.stride() == (1, k)
                or (
                    b.stride() == (n, 1)
                    and n % 8 == 0
                    and 2 * m * n < 2**31
                    and 2 * k * n < 2**31
                    and callable(getattr(tle.gpu, "swizzled_shared_layout", None))
                )
            )
            and c.stride() == (n, 1)
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and disjoint
            and max(m, n) < 2**31
            # Each CTA rebases C in i64; only its local byte offsets are i32.
            and 512 * n < 2**31
            and (not self.group_m or triton.cdiv(m, 256) * triton.cdiv(n, 256) < 2**31)
        )


def launch_cooperative_mm(a, b, c, config):
    if not config.supports(a, b, c):
        raise ValueError("unsupported cooperative matrix storage or pipeline extent")
    m, k = a.shape
    n = b.shape[1]
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx936":
            raise ValueError("cooperative GEMM currently requires gfx936")
        return _mm_cooperative_kernel[(triton.cdiv(m, 256), triton.cdiv(n, 256))](
            a,
            b,
            c,
            m,
            n,
            k,
            GROUP_M=config.group_m,
            B_K_CONTIGUOUS=b.stride() == (1, k),
            num_warps=4,
            num_stages=1,
            **_COOPERATIVE_HCU_OPTIONS,
        )


@lru_cache(maxsize=1)
def cooperative_mm_configs():
    """Two CTA traversal choices for one logical producer/consumer algorithm."""
    return (CooperativeMMConfig(), CooperativeMMConfig(4))


def _cooperative_mm_candidates(a, b, c, *, allow_nn):
    configs = cooperative_mm_configs()
    m, k = a.shape
    n = b.shape[1]
    if b.stride() == (1, k):
        return tuple(config for config in configs if config.supports(a, b, c))
    if not allow_nn or b.stride() != (n, 1):
        return ()
    # One CTA uses the full LDS allocation. Keep tiles with at least 75%
    # useful output and enough CTAs to use at least 75% of the first CU wave.
    tile = 256
    tiles = triton.cdiv(m, tile) * triton.cdiv(n, tile)
    if min(m, n) < tile or 4 * m * n < 3 * tiles * tile**2:
        return ()
    config = CooperativeMMConfig(group_m=4)
    if not config.supports(a, b, c):
        return ()
    compute_units = torch_device_fn.get_device_properties(a.device).multi_processor_count
    return (config,) if 4 * tiles >= 3 * compute_units else ()


# CTA-local Split-K configurations and launch shared with MV.


def _matches_strides(tensor, expected):
    return all(
        size == 1 or actual == stride
        for size, actual, stride in zip(tensor.shape, tensor.stride(), expected)
    )


@dataclass(frozen=True)
class LocalSplitMMConfig:
    tile: tuple = (32, 32)
    k_step: int = 256
    m_tiles: int = 1
    num_stages: int = 1

    def __post_init__(self):
        object.__setattr__(self, "tile", tuple(self.tile))
        if len(self.tile) != 2 or any(
            type(x) is not int or x < 16 or x & (x - 1) for x in self.tile
        ):
            raise ValueError("local split tile dimensions must be powers of two >=16")
        if type(self.k_step) is not int or self.k_step not in (128, 256):
            raise ValueError("unsupported local split K step")
        if type(self.num_stages) is not int or self.num_stages not in (1, 2):
            raise ValueError("local split uses one or two pipeline stages")
        if type(self.m_tiles) is not int or self.m_tiles < 1:
            raise ValueError("m_tiles must be a positive integer")
        bm, bn = self.tile
        # Bound the simultaneous input planes and cross-wave FP32 output
        # scratch independently. Their lifetimes do not overlap.
        if max(2 * self.k_step * (self.m_tiles * bm + bn), 16 * bm * bn) > 65536:
            raise ValueError("local split tiles exceed the 64KiB shared allocation")

    def supports(self, a, b, c):
        if any(t.ndim != 2 for t in (a, b, c)):
            return False
        m, k = a.shape
        n = b.shape[1]
        bm, bn = self.tile
        begin, end = c.data_ptr(), c.data_ptr() + 2 * m * n
        disjoint = all(
            end <= t.data_ptr() or t.data_ptr() + 2 * count <= begin
            for t, count in ((a, m * k), (b, k * n))
        )
        return (
            min(m, n, k) > 0
            and b.shape[0] == k
            and c.shape == (m, n)
            and _matches_strides(a, (k, 1))
            and (_matches_strides(b, (1, k)) or _matches_strides(b, (n, 1)))
            and _matches_strides(c, (n, 1))
            and a.dtype == b.dtype == c.dtype
            and a.dtype in (torch.bfloat16, torch.float16)
            and a.device == b.device == c.device
            and a.device.type == "cuda"
            and not c.requires_grad
            and (not c.is_inference() or torch.is_inference_mode_enabled())
            and all(t.data_ptr() % 2 == 0 for t in (a, b, c))
            and disjoint
            # Coordinates and masks use signed i32, including inactive lanes.
            # All global pointer origins and element offsets use i64.
            and triton.cdiv(m, self.m_tiles * bm) * (self.m_tiles * bm) <= 2**31
            and triton.cdiv(n, bn) * bn <= 2**31
            and triton.cdiv(k, self.k_step) * self.k_step <= 2**31
        )


_LOCAL_SPLIT_CONFIGS = tuple(
    LocalSplitMMConfig(tile, k_step=step, m_tiles=rows, num_stages=stages)
    for stages in (1, 2)
    for tile, step, rows in (
        ((16, 32), 128, 1),
        ((32, 32), 256, 1),
        ((16, 16), 256, 1),
        ((16, 32), 256, 5),
    )
)


_LOCAL_SPLIT_HCU_OPTIONS = dict(
    waves_per_eu=1,
    wave_private_sync=True,
    disable_llvm_loop_unroll=True,
    kpack=2,
    mmac_transposed=False,
)


@cache
def local_split_mm_configs():
    if supports_hcu_options((*_LOCAL_SPLIT_HCU_OPTIONS, "schedule_hint")):
        return _LOCAL_SPLIT_CONFIGS
    return ()


def launch_local_split_mm(a, b, c, config):
    if not isinstance(config, LocalSplitMMConfig) or not config.supports(a, b, c):
        raise ValueError("unsupported local split matrix storage or coordinate range")
    m, k = a.shape
    n = b.shape[1]
    bm, bn = config.tile
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
        if target.backend != "hip" or target.arch != "gfx936":
            raise ValueError("local split GEMM currently requires gfx936")
        return _mm_local_splitk_kernel[
            (triton.cdiv(m, bm * config.m_tiles), triton.cdiv(n, bn), 1)
        ](
            a,
            b,
            c,
            m,
            n,
            k,
            bm,
            bn,
            K_STEP=config.k_step,
            B_ROW_MAJOR=_matches_strides(b, (n, 1)),
            M_TILES=config.m_tiles,
            num_warps=4,
            num_stages=config.num_stages,
            schedule_hint="local-prefetch" if config.num_stages == 2 else "none",
            **_LOCAL_SPLIT_HCU_OPTIONS,
        )


# MV reduction, grouped MV and CTA-local MMAC.


@dataclass(frozen=True)
class GroupedMVConfig:
    block_n: int
    block_k: int


_GROUPED_MV_CONFIGS = tuple(
    GroupedMVConfig(block_n, block_k)
    for block_n, block_k in ((1, 1024), (2, 1024), (2, 2048), (4, 1024))
)


def grouped_mv_candidates(a, b, compute_units):
    """Offer vector reduction for underfilled MMAC rows and contiguous K."""
    m, k = a.shape
    n = b.shape[1]
    if not (1 < m < 8 and b.stride() == (1, k)):
        return ()
    return tuple(
        config
        for config in _GROUPED_MV_CONFIGS
        if config.block_n <= n
        and config.block_k <= k
        and m * triton.cdiv(n, config.block_n) <= 4 * compute_units
    )


@libentry(cache_miss_fallback=capture_gemm_fallback)
@libtuner(
    # Keep the default config fallback for callers that do not opt into
    # FlagTune.  The real-model acceptance path sets FLAGTUNE_INCLUDE=mv,
    # which selects the MV section of mm_hygon_expand.yaml at first launch.
    configs=runtime.get_tuned_config("mv"),
    key=["M", "N", "stride_an", "stride_am", "stride_bm"],
    strategy=runtime.common.DEFAULT_STRATEGIES["mv_hygon"],
    benchmark_mode="replay",
    warmup=5,
    rep=10,
    replay_refinement=(25, 100),
    flagtune_op_name="mv",
    flagtune_expand_op_name="mv_hygon",
)
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M: tl.constexpr,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
    GROUP_VEC_STRIDE: tl.constexpr = 0,
    GROUP_OUT_STRIDE: tl.constexpr = 0,
    WIDE_INDEX: tl.constexpr = False,
):
    """Accumulate K tiles before reducing each output row.

    ``A`` has shape ``[N, M]`` and ``B`` has shape ``[M]``.  Each program
    computes ``BLOCK_N`` output rows and walks the reduction dimension in
    ``BLOCK_M`` element tiles.
    """

    if GROUP_VEC_STRIDE:
        row = ext.program_id(1)
        B += row * GROUP_VEC_STRIDE
        C += row * GROUP_OUT_STRIDE
    pid = ext.program_id(0)
    offset_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_m = tl.arange(0, BLOCK_M)
    n_mask = offset_n < N
    k_bound = M
    if WIDE_INDEX:
        k_bound = tl.cast(M, tl.int64)
        stride_am = tl.cast(stride_am, tl.int64)
        stride_bm = tl.cast(stride_bm, tl.int64)

    # Accumulate corresponding elements across K tiles before reducing.  This
    # avoids a cross-warp reduction on every iteration of a multi-tile row.
    a_ptrs = A + offset_n[:, None] * stride_an + offset_m[None, :] * stride_am
    b_ptrs = B + offset_m * stride_bm
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # Use Triton's loop iterator so LOOP_STAGES is emitted as a loop
    # pipeline attribute.  The JIT parser accepts keyword arguments on a
    # builtin ``range`` call but ignores them when lowering that iterator.
    for m in tl.range(0, k_bound, BLOCK_M, num_stages=LOOP_STAGES):
        m_mask = m + offset_m < k_bound
        a = tl.load(
            a_ptrs,
            mask=n_mask[:, None] & m_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        b = tl.load(b_ptrs, mask=m_mask, other=0.0).to(tl.float32)
        acc += a * b[None, :]
        a_ptrs += BLOCK_M * stride_am
        b_ptrs += BLOCK_M * stride_bm

    acc = tl.sum(acc, axis=1)
    c_ptrs = C + offset_n * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=n_mask)


@lru_cache(maxsize=1)
def _fixed_mv_entry():
    return libentry()(mv_kernel.fn.fn)


def launch_grouped_mv(a, b, c, config):
    """Run one GEMV per A row against the shared matrix B transpose."""
    m, k = a.shape
    n = b.shape[1]
    with torch_device_fn.device(a.device):
        return _fixed_mv_entry()[(triton.cdiv(n, config.block_n), m)](
            b,
            a,
            c,
            n,
            k,
            b.stride(1),
            b.stride(0),
            a.stride(1),
            c.stride(1),
            BLOCK_N=config.block_n,
            BLOCK_M=config.block_k,
            LOOP_STAGES=1,
            GROUP_VEC_STRIDE=a.stride(0),
            GROUP_OUT_STRIDE=c.stride(0),
            WIDE_INDEX=max(m, n, k) >= I32_GEMM_COORD_LIMIT or gemm_has_wide_storage(a, b, c),
            num_warps=4,
        )


def mv(inp, vec):
    """Hygon ``mv`` entry point matching the ATen signature."""

    logger.debug("GEMS_HYGON MV")
    assert inp.dim() == 2 and vec.dim() == 1, "mv expects a matrix and a vector"
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"

    out = torch.empty((inp.shape[0],), device=inp.device, dtype=inp.dtype)
    return _mv_out(inp, vec, out)


def _mv_out(inp, vec, out):
    """Launch the same reduction into an existing, possibly strided vector."""
    assert inp.dim() == 2 and vec.dim() == out.dim() == 1
    N, M = inp.shape
    assert vec.shape[0] == M and out.shape[0] == N
    _dispatch_hygon_mv(inp, vec, out)
    return out


def _dispatch_hygon_mv(inp, vec, out):
    """Choose the vector reduction or the MMAC candidate."""
    mv_kernel._apply_flagtune()
    tuner = mv_kernel.fn
    if tuner._run_mode == "normal":
        tuning_key = gemm_tuning_key(tuner)
        if try_mv_mmac(inp, vec, out,
                       lambda output: _launch_ordinary_mv(inp, vec, output),
                       tuning_key=tuning_key):
            return
    _launch_ordinary_mv(inp, vec, out)


def _launch_ordinary_mv(inp, vec, out):
    N, M = inp.shape
    grid = lambda META: (triton.cdiv(N, META["BLOCK_N"]),)

    with torch_device_fn.device(inp.device):
        mv_kernel[grid](
            inp,
            vec,
            out,
            N,
            M,
            inp.stride(0),
            inp.stride(1),
            vec.stride(0),
            out.stride(0),
            WIDE_INDEX=max(M, N) >= I32_GEMM_COORD_LIMIT or gemm_has_wide_storage(inp, vec, out),
        )

    return out


# MV matrix-instruction candidate and replay selection.


_mv_mmac_choices = GemmChoiceCache()


@cache
def _mmac_config():
    return LocalSplitMMConfig((16, 16), num_stages=2)


def _mv_mmac_eligible(inp, vec, out):
    n, k = inp.shape
    config = _mmac_config()
    if (
        vec.shape != (k,)
        or out.shape != (n,)
        or vec.stride() != (1,)
        or out.stride() != (1,)
        or config not in local_split_mm_configs()
    ):
        return False
    if not config.supports(vec.view(1, k), inp.t(), out.view(1, n)):
        return False
    with torch_device_fn.device(inp.device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx936"


def _launch_mmac(inp, vec, out):
    n, k = inp.shape
    return launch_local_split_mm(
        vec.view(1, k), inp.t(), out.view(1, n), _mmac_config()
    )


def _measure_mv_family(inp, vec, out, ordinary_with_output):
    ordinary_with_output(out)
    succeeded, _ = try_gemm_candidate(_launch_mmac, inp, vec, out)
    if not succeeded:
        return GemmChoice(0)

    def compare(bench):
        samples = ([], [])
        for repeat in range(3):
            for index in (0, 1) if repeat % 2 == 0 else (1, 0):

                def fresh_output():
                    output = torch.empty_like(out)
                    if index:
                        _launch_mmac(inp, vec, output)
                    else:
                        ordinary_with_output(output)
                    return output

                samples[index].append(
                    benchmark_gemm_candidate(
                        bench, fresh_output, optional=index != 0,
                    )
                )
        return tuple(
            statistics.median(x) if all(map(isfinite, x)) else inf
            for x in samples
        )

    screened = compare(gemm_replay_benchmarker(5, 10, 3))
    medians = screened
    if all(map(isfinite, screened)) and max(screened) <= 1.05 * min(screened):
        medians = compare(gemm_replay_benchmarker(25, 100, 10))
    choice = min(range(2), key=medians.__getitem__)
    logger.debug("Hygon MV family=%s screening=%s allocating=%s ms", choice, screened, medians)
    return GemmChoice(choice)


def try_mv_mmac(inp, vec, out, ordinary_with_output, *, tuning_key):
    if not _mv_mmac_eligible(inp, vec, out):
        return False
    key = (
        inp.device,
        inp.dtype,
        vec.dtype,
        out.dtype,
        _mmac_config(),
        tuple(inp.shape),
        tuple(inp.stride()),
        tuple(vec.stride()),
        tuple(out.stride()),
        tuning_key,
        tuple(map(gemm_storage_is_wide, (inp, vec, out))),
    )
    with torch_device_fn.device(inp.device):
        choice = _mv_mmac_choices.get(key)
        if choice is None:
            choice = _mv_mmac_choices.get_or_measure(
                key, _measure_mv_family, inp, vec, out, ordinary_with_output
            )
        if choice is None:
            return False
        if choice.index:
            succeeded, _ = try_gemm_candidate(_launch_mmac, inp, vec, out)
            if succeeded:
                return True
            _mv_mmac_choices.fallback(key, choice)
        ordinary_with_output(out)
    return True


# MM family shortlist, replay selection and choice cache.


_mm_family_choices = GemmChoiceCache()
_SPLITK_LOCAL = (
    LocalSplitMMConfig((16, 16)),
    LocalSplitMMConfig((16, 16), num_stages=2),
)
_SPLITK_LOCAL_WIDE = LocalSplitMMConfig((32, 32), num_stages=2)
_SPLITK_SHARED = SharedMMConfig((32, 64, 64))


@lru_cache(maxsize=1)
def _shared_supported():
    return (
        callable(getattr(tle, "load", None))
        and supports_hcu_options((*_SHARED_HCU_OPTIONS, "mmac_warps_m"))
    )


def mm_family_candidates(a, b, c, *, splitk_pool=False, allow_cooperative_nn=False):
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2 or a.device.type != "cuda":
        return ()
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
    if target.backend != "hip" or target.arch != "gfx936":
        return ()
    local_configs = local_split_mm_configs()
    shared_configs = shared_mm_configs() if _shared_supported() else ()
    if shared_configs and not _shared_address_supported():
        shared_configs = tuple(c for c in shared_configs if c.address_mode == "direct")
    if splitk_pool:
        m, n = c.shape
        compute_units = torch_device_fn.get_device_properties(a.device).multi_processor_count
        bm, bn = _SPLITK_LOCAL_WIDE.tile
        local_tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        local_shortlist = _SPLITK_LOCAL
        if (m >= bm and n >= bn) or local_tiles * 2 >= compute_units:
            local_shortlist += (_SPLITK_LOCAL_WIDE,)
        local_configs = tuple(config for config in local_shortlist if config in local_configs)
        # Consider the smallest Shared tile only when its CTA grid can
        # supply at least one wave per compute unit in aggregate.
        bm, bn, _ = _SPLITK_SHARED.tile
        shared_tiles = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        shared_configs = (
            (_SPLITK_SHARED,)
            if _SPLITK_SHARED in shared_configs
            and shared_tiles * prod(_SPLITK_SHARED.warps) >= compute_units
            else ()
        )
    local = tuple(config for config in local_configs if config.supports(a, b, c))
    shared = tuple(config for config in shared_configs if config.supports(a, b, c))
    if splitk_pool:
        return local + shared + grouped_mv_candidates(a, b, compute_units)
    cooperative = _cooperative_mm_candidates(a, b, c, allow_nn=allow_cooperative_nn)
    short_k = short_k_mm_candidates(a, b, c)
    return shared + cooperative + local + short_k


def _mm_candidate_label(candidate):
    if isinstance(candidate, CooperativeMMConfig):
        return dict(family="cooperative", **asdict(candidate))
    if isinstance(candidate, LocalSplitMMConfig):
        return dict(family="local_split", **asdict(candidate))
    if isinstance(candidate, GroupedMVConfig):
        return dict(family="grouped_mv", **asdict(candidate))
    if isinstance(candidate, ShortKMMConfig):
        return dict(family="short_k", **asdict(candidate))
    return asdict(candidate) if isinstance(candidate, SharedMMConfig) else candidate


def clear_mm_family_choices():
    _mm_family_choices.clear()


def _measure_mm_family(
    a, b, c, ordinary, ordinary_with_output, splitk_pool, allow_cooperative_nn,
):
    candidates = mm_family_candidates(
        a, b, c, splitk_pool=splitk_pool,
        allow_cooperative_nn=allow_cooperative_nn,
    )
    if not candidates:
        return None
    screen = gemm_replay_benchmarker(5, 10, 3)
    ordinary()
    calls = (ordinary,) + tuple(
        lambda candidate=candidate: _dispatch_hygon_mm_candidate(
            a, b, c, candidate
        )
        for candidate in candidates
    )
    available = [0]
    for index, call in enumerate(calls[1:], 1):
        succeeded, _ = try_gemm_candidate(call)
        if succeeded:
            available.append(index)
    samples = tuple([] for _ in calls)
    for repeat in range(3):
        order = available
        if repeat % 2:
            order = reversed(order)
        for index in order:
            samples[index].append(
                benchmark_gemm_candidate(
                    screen, calls[index],
                    optional=index != 0,
                )
            )
    medians = tuple(
        statistics.median(x) if x and all(map(isfinite, x)) else inf
        for x in samples
    )
    choice = min(available, key=medians.__getitem__)
    finalists = tuple(
        i
        for i in sorted(range(len(calls)), key=medians.__getitem__)
        if isfinite(medians[i]) and medians[i] <= medians[choice] * 1.05
    )[:3]
    refined = {}
    if ordinary_with_output is not None and len(finalists) > 1:
        refine = gemm_replay_benchmarker(25, 100, 10)
        refined_samples = {i: [] for i in finalists}
        for repeat in range(3):
            order = (
                finalists if repeat % 2 == 0 else reversed(finalists)
            )
            for index in order:

                def fresh_output():
                    output = torch.empty_like(c)
                    if index:
                        _dispatch_hygon_mm_candidate(
                            a, b, output, candidates[index - 1]
                        )
                    else:
                        ordinary_with_output(output)
                    return output

                refined_samples[index].append(
                    benchmark_gemm_candidate(
                        refine, fresh_output,
                        optional=index != 0,
                    )
                )
        refined = {
            i: statistics.median(x)
            if all(map(isfinite, x)) else inf
            for i, x in refined_samples.items()
        }
        choice = min(finalists, key=refined.__getitem__)
        if not isfinite(refined[choice]):
            choice = 0
    label = _mm_candidate_label(candidates[choice - 1]) if choice else "incumbent"
    logger.debug("Hygon MM family=%s screening=%s allocating=%s ms", label, medians, refined)
    return GemmChoice(choice, candidates)


def try_mm_family(
    a, b, c, ordinary, *, tuning_key, ordinary_with_output=None, splitk_pool=False,
    allow_cooperative_nn=False,
):
    """Compare real candidates with the existing Default/Expanded TL tuner.

    Cache the winner and its candidate descriptors. A cold graph capture
    defers family selection to the incumbent path. In the Split-K region, the
    incumbent is the Split-K tuner and only a fixed shortlist challenges it.
    """
    if any(t.ndim != 2 for t in (a, b, c)) or a.device.type != "cuda":
        return False
    m, k = a.shape
    n = b.shape[1]
    begin, end = c.data_ptr(), c.data_ptr() + 2 * m * n
    storage = (
        tuple(t.data_ptr() % 16 == 0 for t in (a, b, c)),
        all(
            end <= t.data_ptr() or t.data_ptr() + 2 * count <= begin
            for t, count in ((a, m * k), (b, k * n))
        ),
        c.untyped_storage().data_ptr()
        not in (a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr()),
        c.requires_grad,
        c.is_inference(),
        torch.is_inference_mode_enabled(),
        tuple(map(gemm_storage_is_wide, (a, b, c))),
    )
    key = (
        a.device,
        b.device,
        c.device,
        a.dtype,
        b.dtype,
        c.dtype,
        tuple(a.shape),
        tuple(b.shape),
        tuple(c.shape),
        tuple(a.stride()),
        tuple(b.stride()),
        tuple(c.stride()),
        tuning_key,
        storage,
        splitk_pool,
        allow_cooperative_nn,
        ordinary_with_output is not None,
    )
    with torch_device_fn.device(a.device):
        choice = _mm_family_choices.get(key)
        if choice is None:
            choice = _mm_family_choices.get_or_measure(
                key, _measure_mm_family, a, b, c, ordinary, ordinary_with_output,
                splitk_pool, allow_cooperative_nn,
            )
        if choice is None:
            return False
        if choice.index:
            succeeded, _ = try_gemm_candidate(
                _dispatch_hygon_mm_candidate, a, b, c, choice.candidates[choice.index - 1]
            )
            if succeeded:
                return True
            _mm_family_choices.fallback(key, choice)
        ordinary()
    return True


_ordered_datatypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]


def get_higher_dtype(a, b):
    if a is b:
        return a

    assert a in _ordered_datatypes
    assert b in _ordered_datatypes

    for d in _ordered_datatypes:
        if a is d:
            return b
        if b is d:
            return a


def mm(a, b):
    logger.debug("GEMS_HYGON MM")
    device = a.device
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c_dtype = get_higher_dtype(a.dtype, b.dtype)
    if M == 0 or N == 0 or K == 0:
        return torch.zeros((M, N), device=device, dtype=c_dtype)

    # Matrix-vector boundaries use the Hygon-tuned mv family. This is a
    # generic dimension rule (not a model-shape special case) and avoids
    # launching a 2-D tile kernel with one entirely wasted output axis.
    if N == 1 and c_dtype != torch.float64:
        return mv(a, b[:, 0]).reshape(M, 1).to(c_dtype)
    if M == 1 and c_dtype != torch.float64:
        return mv(b.transpose(0, 1), a[0]).reshape(1, N).to(c_dtype)

    c = torch.empty((M, N), device=device, dtype=c_dtype)
    _dispatch_hygon_mm(a, b, c)
    return c


def _dispatch_hygon_mm_candidate(a, b, c, candidate):
    """Launch a fixed configuration selected by the family tuner."""
    if isinstance(candidate, SharedMMConfig):
        return launch_shared_mm(a, b, c, candidate)
    if isinstance(candidate, CooperativeMMConfig):
        return launch_cooperative_mm(a, b, c, candidate)
    if isinstance(candidate, GroupedMVConfig):
        return launch_grouped_mv(a, b, c, candidate)
    if isinstance(candidate, ShortKMMConfig):
        return launch_short_k_mm(a, b, c, candidate)
    return launch_local_split_mm(a, b, c, candidate)


def _dispatch_hygon_mm(a, b, c):
    """Choose the regular or small-output MM pool and launch its winner."""
    tuner = mm_kernel.fn
    # Offline fixed-config collection must reach the requested ordinary tuner.
    if tuner._run_mode != "normal":
        _launch_ordinary_mm(a, b, c)
        return
    if try_mm_splitk(a, b, c, compare_local_split=True):
        return
    mm_kernel._apply_flagtune()
    tuning_key = gemm_tuning_key(tuner)
    if not try_mm_family(
        a, b, c, lambda: _launch_ordinary_mm(a, b, c), tuning_key=tuning_key,
        ordinary_with_output=lambda output: _launch_ordinary_mm(a, b, output),
        allow_cooperative_nn=tuner._flagtune_mode == runtime.TuningMode.EXPANDED,
    ):
        _launch_ordinary_mm(a, b, c)


def _hygon_mm_requires_edge_mask(M, N, large_tile_configs=False):
    if large_tile_configs and (M % 128 or N % 128):
        return True
    # Divisors that occur only in Expanded (for example BLOCK_M=1) cannot
    # prove the default shortlist safe. If any default tile divides the
    # output, pruning retains only dividing tiles; otherwise mask edges.
    return not any(
        _hygon_mm_output_tile_fits(
            M, N, config.kwargs["BLOCK_M"], config.kwargs["BLOCK_N"]
        )
        for config in _HYGON_MM_DEFAULT_CONFIGS
        if not _hygon_mm_is_single_wave_config(config)
    )


def _launch_ordinary_mm(a, b, c):
    M, K = a.shape
    N = b.shape[1]
    bounded_coords = max(M, N, K) < I32_GEMM_COORD_LIMIT
    wide_index = not bounded_coords or gemm_has_wide_storage(a, b, c)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    with torch_device_fn.device(a.device):
        load_mask = 0 if not bounded_coords else _hygon_mm_async_load_mask(a, b, c, M, N, K)
        large_tile_configs = _hygon_mm_large_tile_eligible(a, M, N, K, load_mask)
        mm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            IS_FP64=a.dtype == torch.float64,
            MASK_EDGES=_hygon_mm_requires_edge_mask(M, N, large_tile_configs),
            # Triton block-pointer offsets are i32 even with i64 strides.
            USE_TLE=HAS_HYGON_TLE and bounded_coords,
            WIDE_INDEX=wide_index,
            LARGE_TILE_CONFIGS=large_tile_configs,
            ASYNC_LOAD_MASK=load_mask,
            K_DIVISIBLE_BY_16=K % 16 == 0,
            B_K_CONTIGUOUS=b.stride(0) == 1,
        )


def _hygon_mm_direct_mv_out(a, b, c, M, N):
    # Preserve conversion and overlapping-storage behavior of the temporary
    # route. A vector view retains c's stride without allocating another output.
    return (
        c.shape == (M, N)
        and a.dtype == b.dtype == c.dtype
        and a.device == b.device == c.device
        and not c.requires_grad
        and (not c.is_inference() or torch.is_inference_mode_enabled())
        and (c.stride(0) if N == 1 else c.stride(1)) > 0
        and c.untyped_storage().data_ptr()
        not in (a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
    )


def mm_out(a, b, *, out):
    logger.debug("GEMS_HYGON MM_OUT")
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = out
    if M == 0 or N == 0:
        return c
    if K == 0:
        return c.zero_()
    if N == 1 and a.dtype != torch.float64 and b.dtype != torch.float64:
        if _hygon_mm_direct_mv_out(a, b, c, M, N):
            _mv_out(a, b[:, 0], c[:, 0])
            torch.autograd.graph.increment_version(c)
            return c
        c.copy_(mv(a, b[:, 0]).reshape(M, 1).to(c.dtype))
        return c
    if M == 1 and a.dtype != torch.float64 and b.dtype != torch.float64:
        if _hygon_mm_direct_mv_out(a, b, c, M, N):
            _mv_out(b.transpose(0, 1), a[0], c[0])
            torch.autograd.graph.increment_version(c)
            return c
        c.copy_(mv(b.transpose(0, 1), a[0]).reshape(1, N).to(c.dtype))
        return c

    _dispatch_hygon_mm(a, b, c)
    return c
