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

import copy
import logging
import os

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

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner
from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)
EXPAND_CONFIG_FILENAME = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "mm_hygon_expand.yaml")
)


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
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return False
    required = {"mmac_warps_m", "optimize_epilogue", "schedule_hint", *additional_options}
    return required.issubset(getattr(HIPOptions, "__dataclass_fields__", {}))


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
        and all(t.is_contiguous() and t.data_ptr() % 16 == 0 for t in (a, b, c))
        and all(t.untyped_storage().nbytes() <= 2**31 - 1 for t in (a, b, c))
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
    # A's contiguous reduction dimension is aligned above. B additionally
    # needs an aligned leading dimension; irregular columns retain sync B.
    return 3 if N % 16 == 0 else 1


def _hygon_mm_large_tile_eligible(a, b, c, M, N, K):
    # Preserve the production shortlist's domain. Expanded uses the separate
    # operand capability mask and does not inherit these size preferences.
    masked_family = (
        _HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED
        and M >= 64 and N >= 128 and M * N >= 65536 and K >= 128
    )
    minimum_tile = 128 if _HYGON_MM_TRANSPOSE_OPTIONS_SUPPORTED else 256
    return (
        a.dtype == torch.bfloat16
        and bool(_hygon_mm_async_load_mask(a, b, c, M, N, K))
        and (masked_family or (
            M >= minimum_tile and N >= minimum_tile
            and M % minimum_tile == N % minimum_tile == 0
        ))
    )


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
        bm, bn, bk = (int(meta[k]) for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        mode, stages = int(meta["LOAD_MODE"]), int(config.num_stages)
        if mode not in (0, 1, 2, 3) or mode & ~load_mask:
            continue
        if not masked and (M < bm or N < bn or M % bm or N % bn):
            continue
        # Three-stage async pipelines are legal too. Particular mixed-load
        # shaders can miscompile in DTK; validate compiled candidates instead
        # of excluding an entire stage count from the tuning space.
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


def _hygon_mm_legacy_edge_prune(configs, named_args, **kwargs):
    """Keep a generic, shape-compatible HCU tile family.

    Full tiles must divide the output grid. The gated masked family may
    cover partial output tiles; narrow the search by dimension scale.  The rules use only dimension ranges and preserve
    several neighboring candidates around each boundary; they are not model
    shape dispatches.
    """
    M = int(named_args.get("M", 0))
    N = int(named_args.get("N", 0))
    K = int(named_args.get("K", 0))
    masked_family = kwargs.get("MASK_EDGES", named_args.get("MASK_EDGES", False)) and kwargs.get(
        "LARGE_TILE_CONFIGS", named_args.get("LARGE_TILE_CONFIGS", False)
    )
    valid = [
        config
        for config in configs
        if M >= int(config.kwargs.get("BLOCK_M", 1))
        and N >= int(config.kwargs.get("BLOCK_N", 1))
        and (
            masked_family
            or (
                M % int(config.kwargs.get("BLOCK_M", 1)) == 0
                and N % int(config.kwargs.get("BLOCK_N", 1)) == 0
            )
        )
    ]
    candidates = valid or list(configs)
    if M < 16:
        floor_m = 1 << max(0, M.bit_length() - 1)
        ceil_m = 1 << max(0, (M - 1).bit_length())
        allowed_m = {floor_m, min(16, ceil_m)}
    elif M < 64:
        allowed_m = {16, 32, 64}
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


def _hygon_mm_edge_prune(configs, named_args, **kwargs):
    if configs and "LOAD_MODE" in configs[0].kwargs:
        return _hygon_mm_expanded_prune(configs, named_args, **kwargs)
    legacy = [
        config
        for config in configs
        if not _hygon_mm_is_dense_config(config)
        and not _hygon_mm_is_single_wave_config(config)
    ]
    candidates = _hygon_mm_legacy_edge_prune(legacy, named_args, **kwargs)
    if kwargs.get("LARGE_TILE_CONFIGS", named_args.get("LARGE_TILE_CONFIGS", False)):
        M, N = int(named_args["M"]), int(named_args["N"])
        masked = kwargs.get("MASK_EDGES", named_args.get("MASK_EDGES", False))
        candidates.extend(
            config
            for config in configs
            if _hygon_mm_is_dense_config(config)
            and (not config.kwargs.get("ASYNC_LOAD", False) or not masked)
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
                    and M >= config.kwargs["BLOCK_M"]
                    and N >= config.kwargs["BLOCK_N"]
                    and M % config.kwargs["BLOCK_M"] == 0
                    and N % config.kwargs["BLOCK_N"] == 0
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


_HYGON_MM_DEFAULT_CONFIGS = _hygon_mm_configs()


def _hygon_mm_requires_edge_mask(M, N, large_tile_configs=False):
    if large_tile_configs and (M % 128 or N % 128):
        return True
    # Divisors that occur only in Expanded (for example BLOCK_M=1) cannot
    # prove the default shortlist safe. If any default tile divides the
    # output, pruning retains only dividing tiles; otherwise mask edges.
    return not any(
        M >= config.kwargs["BLOCK_M"]
        and N >= config.kwargs["BLOCK_N"]
        and M % config.kwargs["BLOCK_M"] == 0
        and N % config.kwargs["BLOCK_N"] == 0
        for config in _HYGON_MM_DEFAULT_CONFIGS
        if not _hygon_mm_is_single_wave_config(config)
    )


@triton.jit
def prev_multiple_of(a, b):
    # the largest x<a that x%b ==0
    return tl.cdiv(a, b) * b - b


@libentry()
@libtuner(
    configs=_HYGON_MM_DEFAULT_CONFIGS,
    key=["M", "N", "K", "stride_am", "stride_bk", "LARGE_TILE_CONFIGS",
         "ASYNC_LOAD_MASK", "K_DIVISIBLE_BY_16"],
    strategy=[
        _hygon_mm_tile_bucket,
        _hygon_mm_tile_bucket,
        _hygon_mm_bucket,
        "align32",
        "align32",
        "default",
        "default",
        "default",
    ],
    warmup=25,
    rep=100,
    benchmark_mode="replay",
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
    FULL_K_LOOP: tl.constexpr = False,
    B_K_CONTIGUOUS: tl.constexpr = False,
    USE_I32_INDEX: tl.constexpr = False,
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
    if USE_I32_INDEX:
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

    # previous K multiple
    # prev_k_mult = prev_multiple_of(K, BLOCK_K)
    full_tiles_only: tl.constexpr = (
        FULL_K_LOOP and K_DIVISIBLE_BY_16 and (16 % BLOCK_K == 0)
    )
    if USE_TLE:
        prev_k_mult = tl.cdiv(K, BLOCK_K) * BLOCK_K
    elif full_tiles_only:
        prev_k_mult = K
    elif FULL_K_LOOP:
        prev_k_mult = (K // BLOCK_K) * BLOCK_K
    else:
        prev_k_mult = tl.maximum(tl.cdiv(K, BLOCK_K) * BLOCK_K - BLOCK_K, 0)

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
    if not USE_TLE and not full_tiles_only:
        if not FULL_K_LOOP or prev_k_mult < K:
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
        from .mv import mv

        return mv(a, b[:, 0]).reshape(M, 1).to(c_dtype)
    if M == 1 and c_dtype != torch.float64:
        from .mv import mv

        return mv(b.transpose(0, 1), a[0]).reshape(1, N).to(c_dtype)

    c = torch.empty((M, N), device=device, dtype=c_dtype)
    _launch_matrix_mm(a, b, c)
    return c


def _launch_matrix_mm(a, b, c):
    from .mm_splitk import try_mm_splitk

    if not try_mm_splitk(a, b, c, compare_local_split=True):
        _launch_mm_family(a, b, c)


def _launch_mm_family(a, b, c):
    from .mm_family import try_mm_family

    mm_kernel._apply_flagtune()
    tuner = mm_kernel.fn
    # Offline fixed-config collection must reach the requested ordinary tuner.
    if tuner._run_mode != "normal":
        _launch_ordinary_mm(a, b, c)
        return
    tuning_key = (
        tuner.config_table_name, id(tuner.configs), tuple(tuner._benchmark_protocol),
    )
    if not try_mm_family(
        a, b, c, lambda: _launch_ordinary_mm(a, b, c), tuning_key=tuning_key,
        ordinary_with_output=lambda output: _launch_ordinary_mm(a, b, output),
    ):
        _launch_ordinary_mm(a, b, c)


def _launch_ordinary_mm(a, b, c):
    M, K = a.shape
    N = b.shape[1]
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    large_tile_configs = _hygon_mm_large_tile_eligible(a, b, c, M, N, K)
    with torch_device_fn.device(a.device):
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
            USE_TLE=HAS_HYGON_TLE,
            LARGE_TILE_CONFIGS=large_tile_configs,
            ASYNC_LOAD_MASK=_hygon_mm_async_load_mask(a, b, c, M, N, K),
            K_DIVISIBLE_BY_16=K % 16 == 0,
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
        from .mv import _mv_out, mv

        if _hygon_mm_direct_mv_out(a, b, c, M, N):
            _mv_out(a, b[:, 0], c[:, 0])
            torch.autograd.graph.increment_version(c)
            return c
        c.copy_(mv(a, b[:, 0]).reshape(M, 1).to(c.dtype))
        return c
    if M == 1 and a.dtype != torch.float64 and b.dtype != torch.float64:
        from .mv import _mv_out, mv

        if _hygon_mm_direct_mv_out(a, b, c, M, N):
            _mv_out(b.transpose(0, 1), a[0], c[0])
            torch.autograd.graph.increment_version(c)
            return c
        c.copy_(mv(b.transpose(0, 1), a[0]).reshape(1, N).to(c.dtype))
        return c

    _launch_matrix_mm(a, b, c)
    return c
