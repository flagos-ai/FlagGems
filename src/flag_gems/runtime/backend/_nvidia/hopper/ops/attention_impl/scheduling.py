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

"""Complete host scheduling and kernel-local launch policy for Hopper TLE FA3.

``FA3Scheduler`` owns both kernel routing and every route-dependent heuristic,
and returns one closed ``FA3ExecutionPlan``.  The remaining scheduling classes
only select Triton configs, tile sizes, and launch shapes inside that plan.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import lru_cache
from typing import Any, TypeVar

import torch
import triton

from .validation import PreparedFA3Inputs


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


class KernelFamily(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    LONG = "long"


class MetadataMode(str, Enum):
    PREFILL = "prefill"
    DIRECT_DECODE = "direct_decode"
    MULTI_TOKEN_DECODE = "multi_token_decode"


class PagedGatherMode(IntEnum):
    LEGACY = 0
    BLOCKWISE = 1
    AUTO = 2


class PagedKVLoadMode(str, Enum):
    NONE = "none"
    TMA = "tma"
    NON_TMA = "non_tma"


class TileWave(IntEnum):
    """Candidate tile count relative to one device wave."""

    UNDER = -1
    EXACT = 0
    OVER = 1


class FA3Workload(str, Enum):
    """Coarse serving workload selected before kernel-specific routing."""

    TOKEN_QUERY = "token_query"
    SPECULATIVE_QUERY = "speculative_query"
    SHORT_QUERY = "short_query"
    SERVING_PREFILL = "serving_prefill"
    PREFILL = "prefill"


class QueryPopulation(str, Enum):
    """Mutually exclusive query-population layout classes."""

    SINGLE = "single"
    UNIFORM = "uniform"
    RAGGED = "ragged"
    MIXED = "mixed"
    OVERSIZED = "oversized"

    @property
    def ragged_supported(self) -> bool:
        return self not in {QueryPopulation.SINGLE, QueryPopulation.OVERSIZED}

    @property
    def split_supported(self) -> bool:
        return self is not QueryPopulation.OVERSIZED


class FA3InputProfile(str, Enum):
    """Measured input cohort that refines one coarse workload."""

    GENERAL = "general"
    THROUGHPUT_DECODE = "throughput_decode"
    HEAVY = "heavy"
    LARGE_SINGLE_PREFILL = "large_single_prefill"


class TMAAlignment(str, Enum):
    """TMA legality exposed by volatile Q/O and K/V layouts."""

    NONE = "none"
    QO_ONLY = "qo_only"
    FULL = "full"


_DIRECT_NONE = 1 << 0
_LONG_NONE = 1 << 1
_DIRECT_NON_TMA = 1 << 2
_LONG_NON_TMA = 1 << 3
_DIRECT_TMA = 1 << 4
_LONG_TMA = 1 << 5

# Page-size literals below describe measured profiles or the smallest legal
# TMA tile.  They are deliberately centralized and are not address-generation
# constraints; Blockwise pointer generation supports every power-of-two page.
_DIRECT_TMA_TILE_ROWS = 16
_PERSISTENT_TMA_TILE_ROWS = 64
_COMPACT_PAGED_PROFILE_SIZES = frozenset({16})
_TMA_PAGED_PROFILE_SIZES = frozenset({128})
_DIRECT_WIDE_DECODE_LEGACY_GATHER_PAGE_SIZES = frozenset({32, 256})
_PERSISTENT_LEGACY_GATHER_PAGE_SIZES = frozenset({16, 32, 256})


class PersistentRouteProfile(str, Enum):
    """Persistent tile class selected by the cross-family route model."""

    GENERAL = "general"
    WIDE_DECODE = "wide_decode"
    SHORT_SPEC = "short_spec"


class FA3KernelProfile(str, Enum):
    """Static kernel/hardware profile combined with an input bucket."""

    GENERAL = "general"
    DENSE_MHA = "h100_dense_mha_work"
    PAGED_TMA_D128 = "h100_paged_tma_gqa4_q"
    PAGED_TMA_WIDE = "h100_paged_tma_wide_waves"
    PAGED_COMPACT_MHA = "h100_paged_compact_mha"
    PAGED_COMPACT_D128 = "h100_paged_compact_d128"
    PAGED_COMPACT_WIDE = "h100_paged_compact_wide"
    PAGED_D256_PREFILL = "h100_paged_d256_prefill"


class Toggle(str, Enum):
    AUTO = "auto"
    ON = "on"
    OFF = "off"


class HeadsInL2Mode(str, Enum):
    AUTO = "auto"
    L2_AUTO = "l2_auto"
    EXPLICIT = "explicit"


@dataclass(frozen=True, slots=True)
class HeadsInL2Policy:
    """Host policy for the unified reverse-M and L2-head schedule."""

    mode: HeadsInL2Mode
    value: int = 0


class PagedPrefillRoute(str, Enum):
    AUTO = "auto"
    DIRECT = "direct"
    LONG = "long"


_ParsedArg = TypeVar("_ParsedArg")


@dataclass(frozen=True)
class RouteConfig:
    decode_strategy: KernelFamily
    pack_gqa: Toggle
    paged_prefill_route: PagedPrefillRoute
    paged_prefill_min_q: int | None
    paged_prefill_min_avg_q: int
    paged_gather: PagedGatherMode
    wide_pack_gqa: bool
    force_paged_kv_tma: bool
    ragged_scheduler: str
    heads_in_l2: HeadsInL2Policy
    dynamic_scheduler: str
    dynamic_split: bool
    log_plan: bool


@dataclass(frozen=True, slots=True)
class FA3ExecutionPlan:
    """Cached, shape-independent algorithm plan for one feature region."""

    kernel: KernelFamily
    kernel_name: str
    metadata_mode: MetadataMode
    workload: str
    paged_kv_load: PagedKVLoadMode
    paged_prefill_candidate: bool
    paged_prefill_long: bool
    reason: str
    pack_gqa: bool
    pack_factor: int
    paged_gather_mode: int
    ragged_scheduler: bool
    heads_in_l2: HeadsInL2Policy
    dynamic_scheduler: bool
    persistent_split_kv: bool
    persistent_num_splits: int
    requires_tma_alignment: bool
    log_plan: bool

    @property
    def paged_kv_non_tma(self) -> bool:
        return self.paged_kv_load is PagedKVLoadMode.NON_TMA


@dataclass(frozen=True)
class FA3RouteDecision:
    """Final legalized family/transport preference."""

    family: KernelFamily
    paged_kv_load: PagedKVLoadMode
    persistent_profile: PersistentRouteProfile
    kernel_profile: FA3KernelProfile


@dataclass(frozen=True, slots=True)
class FA3InputBucket:
    """Coarse, hash-stable summary extracted from serving-volatile inputs."""

    workload: FA3Workload
    population: QueryPopulation
    profile: FA3InputProfile
    alignment: TMAAlignment
    padded_unpacked_wave: TileWave
    padded_packed_wave: TileWave
    ragged_unpacked_wave: TileWave
    ragged_packed_wave: TileWave

    def select_wave(self, *, packed: bool, ragged: bool) -> TileWave:
        if ragged:
            return self.ragged_packed_wave if packed else self.ragged_unpacked_wave
        return self.padded_packed_wave if packed else self.padded_unpacked_wave


@dataclass(frozen=True, slots=True)
class FA3RouteFeatures:
    """Canonical family/transport facts derived inside ``route``."""

    persistent_profile: PersistentRouteProfile
    kernel_profile: FA3KernelProfile
    default_legacy_tma: bool
    auto_family_without_tma: KernelFamily
    auto_family_with_tma: KernelFamily
    measured_family: KernelFamily
    measured_load: PagedKVLoadMode | None
    is_paged: bool
    persistent_tma_tile: bool
    capabilities: int


class FA3RouteCostModel:
    """Measured route rules plus a compatibility fallback for unseen shapes."""

    VERSION = "h100_coarse_buckets_v2"
    DENSE_WORK_CROSSOVER = 1 << 26

    @staticmethod
    def _candidate_bit(family: KernelFamily, load: PagedKVLoadMode) -> int:
        if load is PagedKVLoadMode.NONE:
            return _DIRECT_NONE if family is KernelFamily.DIRECT else _LONG_NONE
        if load is PagedKVLoadMode.NON_TMA:
            return _DIRECT_NON_TMA if family is KernelFamily.DIRECT else _LONG_NON_TMA
        return _DIRECT_TMA if family is KernelFamily.DIRECT else _LONG_TMA

    @staticmethod
    def _capabilities(
        is_paged: bool,
        alignment: TMAAlignment,
        arch: int,
        block_size: int,
    ) -> int:
        qo_tma_aligned = alignment is not TMAAlignment.NONE
        kv_tma_aligned = alignment is TMAAlignment.FULL
        if not is_paged:
            return _DIRECT_NONE | (_LONG_NONE if alignment is TMAAlignment.FULL else 0)
        capabilities = _DIRECT_NON_TMA | (_LONG_NON_TMA if qo_tma_aligned else 0)
        if qo_tma_aligned and kv_tma_aligned and arch >= 90:
            if block_size % _DIRECT_TMA_TILE_ROWS == 0:
                capabilities |= _DIRECT_TMA
            if block_size % _PERSISTENT_TMA_TILE_ROWS == 0:
                capabilities |= _LONG_TMA
        return capabilities

    @classmethod
    def analyze(
        cls,
        input_bucket: FA3InputBucket,
        *,
        arch: int,
        dtype: Any,
        element_size: int,
        head_size: int,
        num_heads: int,
        num_heads_k: int,
        pack_factor: int,
        has_cache_kv: bool,
        is_paged: bool,
        is_alibi: bool,
        is_local: bool,
        is_causal: bool,
        is_softcap: bool,
        block_size: int,
        paged_prefill_candidate: bool,
        tile_wave: TileWave,
    ) -> FA3RouteFeatures:
        """Combine coarse input buckets with a static kernel profile."""

        gqa_ratio = num_heads // num_heads_k
        uses_kv_head_grid = num_heads > num_heads_k and pack_factor > 1
        dense_mha_profile = (
            not is_paged
            and num_heads == num_heads_k
            and not (is_causal or is_local or is_alibi or is_softcap)
            and head_size in (64, 128)
        )
        tma_profile_gqa4 = (
            arch == 90
            and element_size == 2
            and is_paged
            and block_size in _TMA_PAGED_PROFILE_SIZES
            and uses_kv_head_grid
            and gqa_ratio == 4
            and not (is_local or is_alibi or is_softcap)
        )
        compact_paged = is_paged and block_size in _COMPACT_PAGED_PROFILE_SIZES
        measured_d256_prefill = (
            paged_prefill_candidate
            and arch == 90
            and dtype == torch.float16
            and input_bucket.population is QueryPopulation.SINGLE
            and input_bucket.profile
            in {FA3InputProfile.HEAVY, FA3InputProfile.LARGE_SINGLE_PREFILL}
            and head_size == 256
            and (block_size, gqa_ratio) in {(16, 4), (32, 8)}
            and pack_factor == 1
            and is_causal
            and not (is_local or is_alibi or is_softcap)
        )
        if dense_mha_profile:
            kernel_profile = FA3KernelProfile.DENSE_MHA
        elif tma_profile_gqa4 and head_size == 128 and is_causal:
            kernel_profile = FA3KernelProfile.PAGED_TMA_D128
        elif tma_profile_gqa4 and head_size == 192:
            kernel_profile = FA3KernelProfile.PAGED_TMA_WIDE
        elif measured_d256_prefill:
            kernel_profile = FA3KernelProfile.PAGED_D256_PREFILL
        elif (
            compact_paged
            and uses_kv_head_grid
            and gqa_ratio == 4
            and head_size in (192, 256)
        ):
            kernel_profile = FA3KernelProfile.PAGED_COMPACT_WIDE
        elif compact_paged and uses_kv_head_grid and head_size == 128:
            kernel_profile = FA3KernelProfile.PAGED_COMPACT_D128
        elif (
            compact_paged
            and num_heads == num_heads_k
            and head_size == 128
            and not (is_alibi or is_local)
        ):
            kernel_profile = FA3KernelProfile.PAGED_COMPACT_MHA
        else:
            kernel_profile = FA3KernelProfile.GENERAL

        if (
            kernel_profile is FA3KernelProfile.PAGED_COMPACT_WIDE
            and input_bucket.workload is FA3Workload.TOKEN_QUERY
        ):
            persistent_profile = PersistentRouteProfile.WIDE_DECODE
        elif (
            kernel_profile is FA3KernelProfile.PAGED_COMPACT_D128
            and gqa_ratio == 4
            and input_bucket.workload is FA3Workload.SPECULATIVE_QUERY
        ):
            persistent_profile = PersistentRouteProfile.SHORT_SPEC
        else:
            persistent_profile = PersistentRouteProfile.GENERAL

        default_legacy_tma = (
            input_bucket.profile is FA3InputProfile.LARGE_SINGLE_PREFILL
            and arch >= 90
            and element_size == 2
            and is_paged
            and block_size in _TMA_PAGED_PROFILE_SIZES
            and head_size == 128
            and gqa_ratio == 4
            and pack_factor > 1
            and is_causal
            and not (is_local or is_alibi or is_softcap)
        )
        compact_paged_ws = kernel_profile in {
            FA3KernelProfile.PAGED_COMPACT_MHA,
            FA3KernelProfile.PAGED_COMPACT_D128,
        }
        compact_serving_prefill = (
            compact_paged_ws
            and uses_kv_head_grid
            and input_bucket.workload is FA3Workload.SERVING_PREFILL
        )
        if has_cache_kv:
            common_paged_long = is_paged and (
                persistent_profile is not PersistentRouteProfile.GENERAL
                or compact_serving_prefill
                or (paged_prefill_candidate and compact_paged_ws)
            )
            auto_family_without_tma = (
                KernelFamily.LONG if common_paged_long else KernelFamily.DIRECT
            )
            auto_family_with_tma = (
                KernelFamily.LONG
                if common_paged_long or (is_paged and paged_prefill_candidate)
                else KernelFamily.DIRECT
            )
        else:
            auto_family_without_tma = (
                KernelFamily.DIRECT
                if input_bucket.workload
                in {
                    FA3Workload.TOKEN_QUERY,
                    FA3Workload.SPECULATIVE_QUERY,
                    FA3Workload.SHORT_QUERY,
                }
                else KernelFamily.LONG
            )
            auto_family_with_tma = auto_family_without_tma

        measured_family = KernelFamily.AUTO
        if (
            kernel_profile is FA3KernelProfile.DENSE_MHA
            and input_bucket.population
            in {QueryPopulation.SINGLE, QueryPopulation.UNIFORM}
        ):
            measured_family = (
                KernelFamily.LONG
                if input_bucket.profile
                in {
                    FA3InputProfile.HEAVY,
                    FA3InputProfile.LARGE_SINGLE_PREFILL,
                }
                else KernelFamily.DIRECT
            )
        elif kernel_profile is FA3KernelProfile.PAGED_TMA_D128:
            if input_bucket.workload is FA3Workload.TOKEN_QUERY:
                measured_family = KernelFamily.DIRECT
            elif input_bucket.workload is FA3Workload.SPECULATIVE_QUERY:
                measured_family = (
                    KernelFamily.LONG
                    if tile_wave is TileWave.UNDER
                    else KernelFamily.DIRECT
                )
            else:
                measured_family = KernelFamily.LONG
        elif kernel_profile is FA3KernelProfile.PAGED_D256_PREFILL:
            measured_family = KernelFamily.LONG
        elif (
            kernel_profile is FA3KernelProfile.PAGED_TMA_WIDE
            and input_bucket.workload is FA3Workload.TOKEN_QUERY
        ) or persistent_profile is not PersistentRouteProfile.GENERAL:
            measured_family = (
                KernelFamily.LONG
                if tile_wave is TileWave.UNDER
                else KernelFamily.DIRECT
            )

        measured_load = PagedKVLoadMode.NONE if not is_paged else None
        measured_tma_profile = kernel_profile is FA3KernelProfile.PAGED_TMA_D128 or (
            kernel_profile is FA3KernelProfile.PAGED_TMA_WIDE
            and input_bucket.workload is FA3Workload.TOKEN_QUERY
        )
        if (
            measured_tma_profile
            and input_bucket.profile is not FA3InputProfile.THROUGHPUT_DECODE
        ):
            measured_load = PagedKVLoadMode.TMA
        return FA3RouteFeatures(
            persistent_profile=persistent_profile,
            kernel_profile=kernel_profile,
            default_legacy_tma=default_legacy_tma,
            auto_family_without_tma=auto_family_without_tma,
            auto_family_with_tma=auto_family_with_tma,
            measured_family=measured_family,
            measured_load=measured_load,
            is_paged=is_paged,
            persistent_tma_tile=(block_size % _PERSISTENT_TMA_TILE_ROWS == 0),
            capabilities=cls._capabilities(
                is_paged,
                input_bucket.alignment,
                arch,
                block_size,
            ),
        )

    @classmethod
    def choose(
        cls,
        features: FA3RouteFeatures,
        family_override: KernelFamily,
        transport_override: bool | None,
    ) -> FA3RouteDecision:
        legacy_tma = (
            features.default_legacy_tma
            if transport_override is None
            else transport_override
        )
        if family_override is KernelFamily.AUTO:
            incumbent_family = (
                features.auto_family_with_tma
                if legacy_tma
                else features.auto_family_without_tma
            )
        else:
            incumbent_family = family_override
        if not features.is_paged:
            incumbent_load = PagedKVLoadMode.NONE
        elif legacy_tma and (
            incumbent_family is KernelFamily.DIRECT or features.persistent_tma_tile
        ):
            incumbent_load = PagedKVLoadMode.TMA
        else:
            incumbent_load = PagedKVLoadMode.NON_TMA

        preferred_family = (
            incumbent_family
            if features.measured_family is KernelFamily.AUTO
            else features.measured_family
        )
        preferred_load = (
            incumbent_load if features.measured_load is None else features.measured_load
        )
        other_family = (
            KernelFamily.LONG
            if preferred_family is KernelFamily.DIRECT
            else KernelFamily.DIRECT
        )
        family_order = (
            (family_override,)
            if family_override is not KernelFamily.AUTO
            else (preferred_family, other_family)
        )
        load_order = (
            (
                preferred_load,
                (
                    PagedKVLoadMode.TMA
                    if preferred_load is PagedKVLoadMode.NON_TMA
                    else PagedKVLoadMode.NON_TMA
                ),
            )
            if features.is_paged
            else (PagedKVLoadMode.NONE,)
        )

        if transport_override is not None and features.is_paged:
            forced_load = (
                PagedKVLoadMode.TMA if transport_override else PagedKVLoadMode.NON_TMA
            )
            for family in family_order:
                if features.capabilities & cls._candidate_bit(family, forced_load):
                    return FA3RouteDecision(
                        family,
                        forced_load,
                        features.persistent_profile,
                        features.kernel_profile,
                    )
        for family in family_order:
            for load in load_order:
                if features.capabilities & cls._candidate_bit(family, load):
                    return FA3RouteDecision(
                        family,
                        load,
                        features.persistent_profile,
                        features.kernel_profile,
                    )
        raise RuntimeError(
            "FA3 route model found no feasible family/transport candidate"
        )


class FA3Scheduler:
    """Build one concrete FA3 route and every heuristic derived from it."""

    RAGGED_SEQUENCE_GROUP_SIZE = 31
    RAGGED_MAX_GROUPS = 32
    RAGGED_MAX_BATCH_SIZE = RAGGED_SEQUENCE_GROUP_SIZE * RAGGED_MAX_GROUPS
    RAGGED_MAX_Q_FILL_RATIO = 0.75
    DEFAULT_PAGED_PREFILL_MIN_Q = 1024
    MAX_DYNAMIC_SPLITS = 3
    MIN_EXPLICIT_SPLIT_K = 12 * 128

    @staticmethod
    def _parse_arg(
        name: str,
        value_map: dict[str, Any],
        arg_class: type[_ParsedArg],
    ) -> _ParsedArg:
        value = os.getenv(name, "auto").strip().lower()
        try:
            mapped_value = value_map[value]
        except KeyError as exc:
            choices = ", ".join(sorted(value_map))
            raise RuntimeError(
                f"invalid {name}={value!r}; expected one of {choices}"
            ) from exc
        return arg_class(mapped_value)

    @staticmethod
    def _optional_env_int(name: str) -> int | None:
        value = os.getenv(name)
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise RuntimeError(
                f"invalid {name}={value!r}; expected an integer"
            ) from exc

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = FA3Scheduler._optional_env_int(name)
        return default if value is None else value

    @staticmethod
    def _parse_choice(name: str, default: str, allowed: set[str]) -> str:
        value = os.getenv(name, default).strip().lower()
        if value not in allowed:
            choices = ", ".join(sorted(allowed))
            raise RuntimeError(f"invalid {name}={value!r}; expected one of {choices}")
        return value

    @staticmethod
    def _parse_bool(name: str, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        value = value.strip().lower()
        if value not in {"0", "1", "false", "true", "off", "on"}:
            raise RuntimeError(f"invalid {name}={value!r}; expected a boolean")
        return value in {"1", "true", "on"}

    @staticmethod
    def _parse_heads_in_l2() -> HeadsInL2Policy:
        name = "FLAG_GEMS_FA3_TLE_HEADS_IN_L2"
        choices = "auto, l2_auto, 0, 1, 2, 4, 8, or 16"
        value = os.getenv(name, HeadsInL2Mode.AUTO.value).strip().lower()
        if value == HeadsInL2Mode.AUTO.value:
            return HeadsInL2Policy(HeadsInL2Mode.AUTO)
        if value == HeadsInL2Mode.L2_AUTO.value:
            return HeadsInL2Policy(HeadsInL2Mode.L2_AUTO)
        try:
            explicit = int(value)
        except ValueError as exc:
            raise RuntimeError(f"invalid {name}={value!r}; expected {choices}") from exc
        if explicit not in (0, 1, 2, 4, 8, 16):
            raise RuntimeError(f"invalid {name}={value!r}; expected {choices}")
        return HeadsInL2Policy(HeadsInL2Mode.EXPLICIT, explicit)

    @staticmethod
    @lru_cache(maxsize=1)
    def load_config() -> RouteConfig:
        """Read and cache one coherent scheduler configuration snapshot."""

        paged_gather = FA3Scheduler._parse_arg(
            "FLAG_GEMS_FA3_TLE_PAGED_GATHER",
            {
                "legacy": PagedGatherMode.LEGACY.value,
                "blockwise": PagedGatherMode.BLOCKWISE.value,
                "auto": PagedGatherMode.AUTO.value,
            },
            PagedGatherMode,
        )
        paged_prefill_route = FA3Scheduler._parse_arg(
            "FLAG_GEMS_FA3_TLE_PAGED_PREFILL_ROUTE",
            {"auto": "auto", "direct": "direct", "long": "long"},
            PagedPrefillRoute,
        )
        decode_strategy = FA3Scheduler._parse_arg(
            "FLAG_GEMS_FA3_TLE_DECODE_STRATEGY",
            {"auto": "auto", "direct": "direct", "long": "long"},
            KernelFamily,
        )
        paged_prefill_min_q = FA3Scheduler._optional_env_int(
            "FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_Q"
        )
        paged_prefill_min_avg_q = FA3Scheduler._env_int(
            "FLAG_GEMS_FA3_TLE_PAGED_PREFILL_MIN_AVG_Q", 128
        )
        pack_gqa = FA3Scheduler._parse_arg(
            "FLAG_GEMS_FA3_TLE_RAGGED_GQA_PACK",
            {"auto": "auto", "on": "on", "off": "off"},
            Toggle,
        )
        ragged_scheduler = FA3Scheduler._parse_choice(
            "FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT",
            "auto",
            {"off", "auto", "ragged"},
        )
        heads_in_l2 = FA3Scheduler._parse_heads_in_l2()
        dynamic_scheduler = FA3Scheduler._parse_choice(
            "FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER",
            "auto",
            {"off", "auto", "on"},
        )
        return RouteConfig(
            decode_strategy=decode_strategy,
            pack_gqa=pack_gqa,
            paged_prefill_route=paged_prefill_route,
            paged_prefill_min_q=paged_prefill_min_q,
            paged_prefill_min_avg_q=paged_prefill_min_avg_q,
            paged_gather=paged_gather,
            wide_pack_gqa=FA3Scheduler._parse_bool(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_WIDE_PACK_GQA"
            ),
            force_paged_kv_tma=FA3Scheduler._parse_bool(
                "FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT"
            ),
            ragged_scheduler=ragged_scheduler,
            heads_in_l2=heads_in_l2,
            dynamic_scheduler=dynamic_scheduler,
            dynamic_split=FA3Scheduler._parse_bool(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DYNAMIC_SPLIT", default=True
            ),
            log_plan=FA3Scheduler._parse_bool("FLAG_GEMS_FA3_TLE_LOG_PLAN"),
        )

    @staticmethod
    def clear_config_cache() -> None:
        FA3Scheduler.load_config.cache_clear()
        FA3Scheduler.route.cache_clear()

    @staticmethod
    def paged_gather_name(mode: PagedGatherMode | int) -> str:
        try:
            return PagedGatherMode(mode).name.lower()
        except ValueError:
            return f"unknown({mode})"

    @staticmethod
    def classify_population(
        *,
        total_q: int,
        batch_size: int,
        padded_q: int,
    ) -> QueryPopulation:
        if batch_size == 1:
            return QueryPopulation.SINGLE
        if batch_size > FA3Scheduler.RAGGED_MAX_BATCH_SIZE:
            return QueryPopulation.OVERSIZED
        if total_q == padded_q:
            return QueryPopulation.UNIFORM
        if 4 * total_q <= 3 * padded_q:
            return QueryPopulation.RAGGED
        return QueryPopulation.MIXED

    @staticmethod
    def classify_workload(
        *,
        total_q: int,
        batch_size: int,
        max_seqlen_q: int,
        gqa_ratio: int,
        has_cache_kv: bool,
        prefill_ready: bool,
        population: QueryPopulation,
    ) -> FA3Workload:
        """Map volatile lengths to one serving-level routing bucket."""

        if max_seqlen_q == 1:
            return FA3Workload.TOKEN_QUERY
        if has_cache_kv and max_seqlen_q <= 8:
            return FA3Workload.SPECULATIVE_QUERY
        if prefill_ready:
            return FA3Workload.PREFILL
        if has_cache_kv:
            serving_prefill = (
                population is QueryPopulation.RAGGED and max_seqlen_q >= 256
            ) or (
                population is QueryPopulation.SINGLE
                and total_q == max_seqlen_q
                and max_seqlen_q >= 512
            )
            return (
                FA3Workload.SERVING_PREFILL
                if serving_prefill
                else FA3Workload.SHORT_QUERY
            )
        if max_seqlen_q * gqa_ratio <= 128 or total_q <= 64 * batch_size:
            return FA3Workload.SHORT_QUERY
        return FA3Workload.PREFILL

    @staticmethod
    def classify_input_profile(
        *,
        workload: FA3Workload,
        population: QueryPopulation,
        total_q: int,
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_heads: int,
        head_dim: int,
    ) -> FA3InputProfile:
        if (
            workload is FA3Workload.TOKEN_QUERY
            and batch_size >= 32
            and max_seqlen_k >= 4096
        ):
            return FA3InputProfile.THROUGHPUT_DECODE
        if (
            population is QueryPopulation.SINGLE
            and total_q == max_seqlen_q
            and max_seqlen_q >= 8192
            and max_seqlen_k >= 8192
        ):
            return FA3InputProfile.LARGE_SINGLE_PREFILL
        if (
            total_q * max_seqlen_k * num_heads * head_dim
            > FA3RouteCostModel.DENSE_WORK_CROSSOVER
        ):
            return FA3InputProfile.HEAVY
        return FA3InputProfile.GENERAL

    @staticmethod
    def metadata_mode(*, has_cache_kv: bool, max_query_len: int) -> MetadataMode:
        if not has_cache_kv:
            return MetadataMode.PREFILL
        if max_query_len <= 1:
            return MetadataMode.DIRECT_DECODE
        return MetadataMode.MULTI_TOKEN_DECODE

    @staticmethod
    def _tile_wave(tile_count: int, num_sms: int) -> TileWave:
        if tile_count < num_sms:
            return TileWave.UNDER
        if tile_count == num_sms:
            return TileWave.EXACT
        return TileWave.OVER

    @classmethod
    def analyze_inputs(
        cls,
        inputs: PreparedFA3Inputs,
        config: RouteConfig,
    ) -> FA3InputBucket:
        """Extract host-only, discrete facts from serving-volatile inputs."""

        batch_size = inputs.batch_size
        max_q = inputs.max_seqlen_q
        max_k = inputs.max_seqlen_k
        total_q = inputs.total_q
        num_heads = inputs.num_heads
        num_heads_k = inputs.num_heads_k
        head_dim = inputs.head_dim
        gqa_ratio = num_heads // num_heads_k if num_heads > num_heads_k else 1
        packed_heads = num_heads_k if gqa_ratio > 1 else num_heads
        padded_q = batch_size * max_q
        population = cls.classify_population(
            total_q=total_q,
            batch_size=batch_size,
            padded_q=padded_q,
        )

        padded_unpacked_tiles = ((max_q + 127) // 128) * batch_size * num_heads
        padded_packed_tiles = (
            ((max_q * gqa_ratio + 127) // 128) * batch_size * packed_heads
        )
        ragged_unpacked_tiles = ((total_q + 127) // 128 + batch_size - 1) * num_heads
        ragged_packed_tiles = (
            (total_q * gqa_ratio + 127) // 128 + batch_size - 1
        ) * packed_heads

        min_prefill_q = config.paged_prefill_min_q
        if min_prefill_q is None:
            min_prefill_q = cls.DEFAULT_PAGED_PREFILL_MIN_Q
        prefill_ready = (
            max_q >= min_prefill_q
            and total_q >= config.paged_prefill_min_avg_q * batch_size
        )
        workload = cls.classify_workload(
            total_q=total_q,
            batch_size=batch_size,
            max_seqlen_q=max_q,
            gqa_ratio=gqa_ratio,
            has_cache_kv=inputs.has_cache_kv,
            prefill_ready=prefill_ready,
            population=population,
        )
        profile = cls.classify_input_profile(
            workload=workload,
            population=population,
            total_q=total_q,
            batch_size=batch_size,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            num_heads=num_heads,
            head_dim=head_dim,
        )
        tile_wave = cls._tile_wave
        num_sms = inputs.num_sms
        return FA3InputBucket(
            workload=workload,
            population=population,
            profile=profile,
            alignment=(
                TMAAlignment.FULL
                if inputs.qo_tma_aligned and inputs.kv_tma_aligned
                else (
                    TMAAlignment.QO_ONLY if inputs.qo_tma_aligned else TMAAlignment.NONE
                )
            ),
            padded_unpacked_wave=tile_wave(padded_unpacked_tiles, num_sms),
            padded_packed_wave=tile_wave(padded_packed_tiles, num_sms),
            ragged_unpacked_wave=tile_wave(ragged_unpacked_tiles, num_sms),
            ragged_packed_wave=tile_wave(ragged_packed_tiles, num_sms),
        )

    @classmethod
    @lru_cache(maxsize=1024)
    def route(
        cls,
        *,
        config: RouteConfig,
        bucket: FA3InputBucket,
        arch: int,
        dtype: Any,
        element_size: int,
        head_size: int,
        num_heads: int,
        num_heads_k: int,
        has_cache_kv: bool,
        is_paged: bool,
        is_alibi: bool,
        is_local: bool,
        block_size: int,
        is_causal: bool,
        is_softcap: bool,
        has_seqused_k: bool,
        requested_family: KernelFamily = KernelFamily.AUTO,
        pagedkv_tma: bool | None = None,
        max_num_splits: int = 0,
        explicit_split_has_work: bool = False,
    ) -> FA3ExecutionPlan:
        """Return a complete plan keyed only by discrete/stable route facts."""

        # ``dtype`` intentionally partitions the LRU key even though the current
        # legality rules only need ``element_size``.
        if config.force_paged_kv_tma and not is_paged:
            raise RuntimeError(
                "FLAG_GEMS_FA3_TLE_PAGED_KV_TMA_EXPERIMENT requires paged KV"
            )

        gqa_ratio = num_heads // num_heads_k
        # These measured single-request D256 cohorts otherwise launch only one
        # or two direct CTAs and serialize long K despite an explicit split cap.
        # Keep the exception separate from the default page/workload profiles.
        explicit_wide_split = (
            max_num_splits > 1
            and explicit_split_has_work
            and has_cache_kv
            and has_seqused_k
            and is_paged
            and config.dynamic_split
            and config.pack_gqa is not Toggle.OFF
            and bucket.population is QueryPopulation.SINGLE
            and bucket.profile is FA3InputProfile.HEAVY
            and bucket.workload in {FA3Workload.TOKEN_QUERY, FA3Workload.SHORT_QUERY}
            and head_size == 256
            and (block_size, gqa_ratio) in {(16, 4), (32, 8)}
            and (bucket.workload is FA3Workload.TOKEN_QUERY or is_causal)
            and not (is_local or is_alibi or is_softcap)
        )
        pack_gqa = (
            num_heads > num_heads_k
            and gqa_ratio <= 16
            and (gqa_ratio & (gqa_ratio - 1)) == 0
            and (
                head_size <= 128
                or (
                    bucket.workload is FA3Workload.TOKEN_QUERY
                    and head_size in (192, 256)
                )
                or explicit_wide_split
                or config.wide_pack_gqa
            )
            and config.pack_gqa is not Toggle.OFF
        )
        pack_factor = gqa_ratio if pack_gqa else 1

        ragged_scheduler = cls.select_ragged_scheduler(
            config.ragged_scheduler,
            supported=bucket.population.ragged_supported,
            auto_candidate=bucket.population is QueryPopulation.RAGGED,
        )

        tile_wave = bucket.select_wave(packed=pack_gqa, ragged=ragged_scheduler)
        paged_prefill_candidate = (
            has_cache_kv and is_paged and bucket.workload is FA3Workload.PREFILL
        )
        route_features = FA3RouteCostModel.analyze(
            bucket,
            arch=arch,
            dtype=dtype,
            element_size=element_size,
            head_size=head_size,
            num_heads=num_heads,
            num_heads_k=num_heads_k,
            pack_factor=pack_factor,
            has_cache_kv=has_cache_kv,
            is_paged=is_paged,
            is_alibi=is_alibi,
            is_local=is_local,
            is_causal=is_causal,
            is_softcap=is_softcap,
            block_size=block_size,
            paged_prefill_candidate=paged_prefill_candidate,
            tile_wave=tile_wave,
        )
        selected_family = (
            config.decode_strategy
            if config.decode_strategy is not KernelFamily.AUTO
            else requested_family
        )
        family_override = selected_family
        if (
            family_override is KernelFamily.AUTO
            and has_cache_kv
            and is_paged
            and config.paged_prefill_route is not PagedPrefillRoute.AUTO
        ):
            family_override = KernelFamily(config.paged_prefill_route.value)
        if family_override is KernelFamily.AUTO and explicit_wide_split:
            family_override = KernelFamily.LONG
        transport_override = True if config.force_paged_kv_tma else pagedkv_tma
        decision = FA3RouteCostModel.choose(
            route_features,
            family_override,
            transport_override,
        )

        paged_kv_non_tma = decision.paged_kv_load is PagedKVLoadMode.NON_TMA
        split_profile_supported = (
            max_num_splits == 0
            and decision.persistent_profile is PersistentRouteProfile.SHORT_SPEC
        ) or (
            max_num_splits > 1
            and explicit_split_has_work
            and (
                explicit_wide_split
                or decision.persistent_profile
                in {
                    PersistentRouteProfile.SHORT_SPEC,
                    PersistentRouteProfile.WIDE_DECODE,
                }
            )
        )
        persistent_split_kv = (
            config.dynamic_split
            and is_paged
            and paged_kv_non_tma
            and has_seqused_k
            and bucket.population.split_supported
            and decision.family is KernelFamily.LONG
            and pack_gqa
            and split_profile_supported
        )
        persistent_num_splits = (
            max_num_splits
            if persistent_split_kv and max_num_splits > 1
            else cls.MAX_DYNAMIC_SPLITS if persistent_split_kv else 0
        )

        heads_in_l2 = cls.select_heads_in_l2(
            config.heads_in_l2,
            causal=is_causal,
            local=is_local,
        )
        dynamic_scheduler = cls.select_dynamic_scheduler(
            config.dynamic_scheduler,
            causal=is_causal,
            local=is_local,
            tile_wave=tile_wave,
        )
        if persistent_split_kv:
            # A single compact query population has uniform split work and the
            # static program-id stride covers every item without scheduler
            # atomics or producer/consumer handshakes.  Keep dynamic claiming
            # for auto-s3, ragged/multi-request work, or an explicit force-on.
            static_explicit_split = (
                max_num_splits > 1
                and bucket.population is QueryPopulation.SINGLE
                and config.dynamic_scheduler != "on"
            )
            dynamic_scheduler = not static_explicit_split

        paged_gather_mode = int(config.paged_gather)
        if is_paged and (block_size & (block_size - 1)) != 0:
            # Blockwise relies on power-of-two page/tile divisibility.  Resolve
            # unsupported page sizes on the host so no validity check reaches
            # the Triton kernel.
            paged_gather_mode = int(PagedGatherMode.LEGACY)
        elif (
            paged_gather_mode == int(PagedGatherMode.AUTO)
            and decision.family is KernelFamily.DIRECT
            and is_paged
            and paged_kv_non_tma
            and pack_gqa
            and bucket.workload is FA3Workload.TOKEN_QUERY
            and head_size == 192
            and block_size in _DIRECT_WIDE_DECODE_LEGACY_GATHER_PAGE_SIZES
        ):
            # H100 D192 decode: three isolated CUDA-Graph rounds and NCU both
            # favor Legacy here; the compact-page profile remains Blockwise.
            paged_gather_mode = int(PagedGatherMode.LEGACY)
        elif (
            paged_gather_mode == int(PagedGatherMode.AUTO)
            and persistent_split_kv
            and max_num_splits > 1
            and (
                explicit_wide_split
                or decision.persistent_profile is PersistentRouteProfile.WIDE_DECODE
            )
            and block_size in _COMPACT_PAGED_PROFILE_SIZES
        ):
            # Once wide decode is split into short K ranges, measured H100
            # CUDA-Graph latency favors blockwise address generation.  Keep the
            # existing one-pass and auto-s3 gather choices unchanged.
            paged_gather_mode = int(PagedGatherMode.BLOCKWISE)
        elif persistent_split_kv or decision.family is KernelFamily.LONG:
            paged_gather_mode = cls.select_persistent_paged_gather(
                paged_gather_mode,
                is_paged=is_paged,
                paged_kv_non_tma=paged_kv_non_tma,
                pack_gqa=pack_gqa,
                block_size=block_size,
            )

        paged_prefill_long = (
            has_cache_kv and is_paged and decision.family is KernelFamily.LONG
        )
        if paged_prefill_long:
            kernel_name = "long_paged_prefill"
        else:
            kernel_name = decision.family.value
        if decision.family is KernelFamily.DIRECT and pack_gqa:
            kernel_name = "direct_packed_gqa"
        if ragged_scheduler:
            kernel_name = f"{kernel_name}_ragged"
        if persistent_split_kv:
            kernel_name = f"persistent_splitkv_s{persistent_num_splits}"

        return FA3ExecutionPlan(
            kernel=decision.family,
            kernel_name=kernel_name,
            metadata_mode=cls.metadata_mode(
                has_cache_kv=has_cache_kv,
                max_query_len=(1 if bucket.workload is FA3Workload.TOKEN_QUERY else 2),
            ),
            workload=bucket.workload.value,
            paged_kv_load=decision.paged_kv_load,
            paged_prefill_candidate=paged_prefill_candidate,
            paged_prefill_long=paged_prefill_long,
            reason=(
                f"model={FA3RouteCostModel.VERSION} "
                f"kernel_profile={decision.kernel_profile.value} "
                f"input_profile={bucket.profile.value} "
                f"profile={decision.persistent_profile.value} "
                f"family={decision.family.value} "
                f"load={decision.paged_kv_load.value} "
                f"wave={tile_wave.name.lower()}"
            ),
            pack_gqa=pack_gqa,
            pack_factor=pack_factor,
            paged_gather_mode=paged_gather_mode,
            ragged_scheduler=ragged_scheduler,
            heads_in_l2=heads_in_l2,
            dynamic_scheduler=dynamic_scheduler,
            persistent_split_kv=persistent_split_kv,
            persistent_num_splits=persistent_num_splits,
            requires_tma_alignment=(
                (is_paged and not paged_kv_non_tma)
                or (decision.family is KernelFamily.LONG and not persistent_split_kv)
            ),
            log_plan=config.log_plan,
        )

    @staticmethod
    def select_ragged_scheduler(
        mode: str,
        *,
        supported: bool,
        auto_candidate: bool,
    ) -> bool:
        if mode not in {"off", "auto", "ragged"}:
            raise RuntimeError(
                "FLAG_GEMS_FA3_TLE_MIXED_EXPERIMENT must be off, auto, or ragged"
            )
        return supported and (mode == "ragged" or mode == "auto" and auto_candidate)

    @staticmethod
    def select_heads_in_l2(
        policy: HeadsInL2Policy, *, causal: bool, local: bool
    ) -> HeadsInL2Policy:
        """Resolve mask-dependent modes without using volatile shape data."""

        enabled = causal or local
        if policy.mode is HeadsInL2Mode.AUTO:
            return HeadsInL2Policy(HeadsInL2Mode.EXPLICIT, int(enabled))
        if policy.mode is HeadsInL2Mode.L2_AUTO and not enabled:
            return HeadsInL2Policy(HeadsInL2Mode.EXPLICIT, 0)
        return policy

    @staticmethod
    def select_dynamic_scheduler(
        mode: str,
        *,
        causal: bool,
        local: bool,
        tile_wave: TileWave,
    ) -> bool:
        if mode not in {"off", "auto", "on"}:
            raise RuntimeError(
                "FLAG_GEMS_FA3_TLE_DYNAMIC_SCHEDULER must be off, auto, or on"
            )
        if mode == "off":
            return False
        if mode == "on":
            return tile_wave is TileWave.OVER
        return (causal or local) and tile_wave is TileWave.OVER

    @staticmethod
    def select_persistent_paged_gather(
        mode: int,
        *,
        is_paged: bool,
        paged_kv_non_tma: bool,
        pack_gqa: bool,
        block_size: int,
    ) -> int:
        if (
            mode == int(PagedGatherMode.AUTO)
            and is_paged
            and paged_kv_non_tma
            and pack_gqa
            and block_size in _PERSISTENT_LEGACY_GATHER_PAGE_SIZES
        ):
            return int(PagedGatherMode.LEGACY)
        return mode

    @classmethod
    def build(
        cls,
        inputs: PreparedFA3Inputs,
        config: RouteConfig | None = None,
    ) -> FA3ExecutionPlan:
        """Analyze volatile inputs and fetch a cached execution plan."""

        if config is None:
            config = cls.load_config()
        bucket = cls.analyze_inputs(inputs, config)
        plan = cls.route(
            config=config,
            bucket=bucket,
            arch=inputs.arch,
            dtype=inputs.q.dtype,
            element_size=inputs.q.element_size(),
            head_size=inputs.head_dim,
            num_heads=inputs.num_heads,
            num_heads_k=inputs.num_heads_k,
            has_cache_kv=inputs.has_cache_kv,
            is_paged=inputs.is_paged,
            is_alibi=inputs.alibi_slopes is not None,
            is_local=inputs.window.local,
            block_size=inputs.block_size,
            is_causal=inputs.window.causal,
            is_softcap=inputs.is_softcap,
            has_seqused_k=inputs.seqused_k is not None,
            requested_family=KernelFamily.AUTO,
            max_num_splits=inputs.max_num_splits,
            explicit_split_has_work=(
                inputs.max_num_splits > 1
                and inputs.max_seqlen_k > cls.MIN_EXPLICIT_SPLIT_K
            ),
        )
        return plan


@dataclass(frozen=True)
class DirectLaunchPlan:
    effective_max_q: int
    effective_num_heads: int
    pack_factor: int
    batch_first_grid: bool
    single_kv_tile: bool
    shape_bucket: int


@dataclass(frozen=True)
class PersistentLaunchPlan:
    num_mma_groups: int
    block_m: int
    heads_in_l2: int
    dynamic_scheduler: bool


@dataclass(frozen=True)
class SplitCombineLaunchPlan:
    block_m: int
    block_k: int
    compact_mblocks: int
    compact_ragged: bool


class CommonSchedulingHeuristics:
    """Tile math shared by direct and persistent kernel launchers."""

    @staticmethod
    def padded_head_dim(head_dim: int) -> int:
        return _next_power_of_2(head_dim)

    @staticmethod
    def binary_heads_in_l2(policy: HeadsInL2Policy) -> int:
        """Collapse an enabled policy for families without head swizzling."""

        if policy.mode is HeadsInL2Mode.EXPLICIT and policy.value == 0:
            return 0
        return 1

    @classmethod
    def block_k(cls, args):
        """Triton heuristic shared by direct and persistent kernels."""

        return cls.padded_head_dim(args["d"])

    @staticmethod
    def compact_m_upper(
        *, total_q: int, pack_factor: int, batch_size: int, block_m: int
    ) -> int:
        return _ceil_div(total_q * pack_factor, block_m) + batch_size - 1


class DirectSchedulingHeuristics:
    """Autotune and launch-shape policy for the direct one-pass path."""

    DENSE_DECODE_BUCKET = 11
    PAGED_DECODE_BUCKET = 12
    PAGED_PREFILL_BUCKET = 25
    PAGED_PACKED_MEDIUM_BUCKET = 26

    DEFAULT_FORCED_BLOCK_N = 0
    DEFAULT_FORCED_NUM_WARPS = 0
    DEFAULT_FORCED_NUM_STAGES = 0

    @classmethod
    def autotune_configs(cls):
        configs = []
        for block_m in (16, 32, 64, 128):
            for block_n in (16, 32, 64, 128, 256):
                stage_choices = (
                    (1, 2, 3) if block_m == 16 and block_n <= 128 else (2, 3)
                )
                if block_n > 128:
                    stage_choices = (3,)
                warp_choices = (4, 8) if block_n >= 64 else (4,)
                for num_stages in stage_choices:
                    for num_warps in warp_choices:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n},
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )

        forced_block_n = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_BLOCK_N",
                str(cls.DEFAULT_FORCED_BLOCK_N),
            )
        )
        forced_num_warps = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_NUM_WARPS",
                str(cls.DEFAULT_FORCED_NUM_WARPS),
            )
        )
        forced_num_stages = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_NUM_STAGES",
                str(cls.DEFAULT_FORCED_NUM_STAGES),
            )
        )
        if forced_block_n not in (0, 16, 32, 64, 128, 256):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_BLOCK_N must be 0, 16, "
                "32, 64, 128, or 256"
            )
        if forced_num_warps not in (0, 4, 8):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_NUM_WARPS must be 0, 4, " "or 8"
            )
        if forced_num_stages not in (0, 1, 2, 3):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DIRECT_NUM_STAGES must be 0, 1, "
                "2, or 3"
            )
        configs = [
            config
            for config in configs
            if (not forced_block_n or config.kwargs["BLOCK_N"] == forced_block_n)
            and (not forced_num_warps or config.num_warps == forced_num_warps)
            and (not forced_num_stages or config.num_stages == forced_num_stages)
        ]
        if not configs:
            raise ValueError("forced FA3 direct configuration has no candidate")
        return configs

    @classmethod
    def prune_autotune_configs(cls, configs, nargs, **kwargs):
        q_ptr = kwargs.get("q_ptr", nargs.get("q_ptr"))
        batch_size = kwargs.get("b", nargs.get("b", 0))
        head_dim = kwargs.get("d", nargs.get("d"))
        is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
        max_seqlen_q = kwargs.get("seqlen_q", nargs.get("seqlen_q", 0))
        seqlen_k = kwargs.get("seqlen_k", nargs.get("seqlen_k", 0))
        total_q = kwargs.get("total_q", nargs.get("total_q", 0))
        h_hk_ratio = kwargs.get("h_hk_ratio", nargs.get("h_hk_ratio", 1))
        shape_bucket = kwargs.get(
            "DIRECT_SHAPE_BUCKET",
            nargs.get("DIRECT_SHAPE_BUCKET", cls.DENSE_DECODE_BUCKET),
        )
        ragged_scheduler = kwargs.get(
            "RAGGED_SCHEDULER", nargs.get("RAGGED_SCHEDULER", False)
        )
        paged_kv_non_tma = kwargs.get(
            "PAGED_KV_NON_TMA", nargs.get("PAGED_KV_NON_TMA", True)
        )
        block_size = kwargs.get("block_size", nargs.get("block_size", 1))

        # Cold-cache autotuning favors BM64/BN64 for this serving regime, but
        # repeated CUDA Graph measurements show that BF16 is consistently
        # faster with more Q parallelism and half as many KV loop iterations.
        # Keep the rule structural so nearby ragged serving batches benefit
        # without coupling scheduling to one benchmark name.
        short_ragged_bf16 = (
            str(getattr(q_ptr, "dtype", "")) == "torch.bfloat16"
            and shape_bucket == cls.PAGED_PACKED_MEDIUM_BUCKET
            and is_paged
            and paged_kv_non_tma
            and ragged_scheduler
            and batch_size >= 32
            and max_seqlen_q <= 64
            and total_q <= batch_size * 8
            and 256 <= seqlen_k <= 1024
            and head_dim == 128
            and h_hk_ratio == 2
        )
        if short_ragged_bf16:
            measured = [
                config
                for config in configs
                if config.kwargs == {"BLOCK_M": 32, "BLOCK_N": 128}
                and config.num_warps == 4
                and config.num_stages == 2
            ]
            if measured:
                return measured

        kept = []
        for config in configs:
            block_m = config.kwargs["BLOCK_M"]
            block_n = config.kwargs["BLOCK_N"]
            if block_n == 16 and (
                not is_paged or paged_kv_non_tma or block_size % block_n != 0
            ):
                continue
            if is_paged and not paged_kv_non_tma and block_size % block_n != 0:
                continue
            if shape_bucket in (cls.DENSE_DECODE_BUCKET, cls.PAGED_DECODE_BUCKET):
                page_tma_tile = (
                    is_paged and not paged_kv_non_tma and block_n == block_size
                )
                if block_m != 16 or (block_n < 64 and not page_tma_tile):
                    continue
                if head_dim >= 192 and block_n > 128:
                    continue
                if is_paged and block_n > 128:
                    continue
            elif shape_bucket == cls.PAGED_PACKED_MEDIUM_BUCKET:
                if not is_paged:
                    continue
                if block_m < 32 or block_n > 128:
                    continue
                if seqlen_k <= 128 and block_n != 128:
                    continue
                if head_dim > 128 and block_n > 64:
                    continue
            elif shape_bucket == cls.PAGED_PREFILL_BUCKET:
                if not is_paged:
                    continue
                if block_m < 64 or block_n > 128:
                    continue
                if head_dim > 128 and block_n > 64:
                    continue
            else:
                if block_n > 128:
                    continue
                if block_m < 64:
                    continue
                if seqlen_k <= 128 and block_n < 64:
                    continue
                if head_dim > 128 and block_n > 64:
                    continue
            if head_dim > 192 and block_n > 128:
                continue
            kept.append(config)
        return kept or [configs[0]]

    @classmethod
    def launch_plan(
        cls,
        *,
        max_seqlen_k: int,
        batch_size: int,
        effective_max_q: int,
        effective_num_heads: int,
        pack_factor: int,
        is_paged: bool,
        paged_prefill: bool,
        pack_gqa: bool,
    ) -> DirectLaunchPlan:
        batch_first_grid = (
            is_paged and pack_gqa and 1 < batch_size <= 8 and effective_max_q <= 256
        )
        medium_packed_q = is_paged and pack_gqa and effective_max_q > 64
        single_kv_tile = medium_packed_q and max_seqlen_k <= 128
        if paged_prefill:
            shape_bucket = cls.PAGED_PREFILL_BUCKET
        elif medium_packed_q:
            shape_bucket = cls.PAGED_PACKED_MEDIUM_BUCKET
        elif is_paged:
            shape_bucket = cls.PAGED_DECODE_BUCKET
        else:
            shape_bucket = cls.DENSE_DECODE_BUCKET
        return DirectLaunchPlan(
            effective_max_q=effective_max_q,
            effective_num_heads=effective_num_heads,
            pack_factor=pack_factor,
            batch_first_grid=batch_first_grid,
            single_kv_tile=single_kv_tile,
            shape_bucket=shape_bucket,
        )


class PersistentSchedulingHeuristics:
    """Autotune, pipeline, and launch-shape policy for the persistent path."""

    DEFAULT_BLOCK_M = 128
    DEFAULT_NUM_MMA_GROUPS = 2
    DEFAULT_NUM_Q_BUFFERS = 1
    DEFAULT_USE_TMA_QO = True
    DEFAULT_REUSE_Q_SMEM_O = True
    DEFAULT_Q_PIPE_ASYNC = True
    DEFAULT_PAGED_PIPE_ASYNC = True
    DEFAULT_DECODE_USE_TMA_QO = False
    DEFAULT_WARP_MMA = False
    DEFAULT_FORCED_BLOCK_M = 0
    DEFAULT_FORCED_KV_BUFFERS = 0
    DEFAULT_FORCED_BLOCK_N = 0
    LPT_HEURISTIC_BLOCK_N = 128
    LPT_L2_BYTES = 32 * 1024 * 1024
    DENSE_POINTER_MAX_K = 128
    DENSE_POINTER_MAX_D = 64
    DENSE_POINTER_MIN_TILES = 768

    @staticmethod
    def combine_launch_plan(
        *, max_seqlen_q: int, head_dim: int, total_q: int, batch_size: int
    ) -> SplitCombineLaunchPlan:
        """Choose the Split-KV reduction tile and compact work mapping."""

        block_m = min(64, _next_power_of_2(max_seqlen_q))
        block_k = CommonSchedulingHeuristics.padded_head_dim(head_dim)
        rectangular_mblocks = _ceil_div(max_seqlen_q, block_m) * batch_size
        compact_mblocks = _ceil_div(total_q, block_m) + batch_size - 1
        return SplitCombineLaunchPlan(
            block_m=block_m,
            block_k=block_k,
            compact_mblocks=compact_mblocks,
            compact_ragged=compact_mblocks < rectangular_mblocks,
        )

    @classmethod
    def num_mma_groups(cls) -> int:
        value = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_MMA_GROUPS",
                str(cls.DEFAULT_NUM_MMA_GROUPS),
            )
        )
        if value not in (1, 2):
            raise ValueError("FLAG_GEMS_FA3_TLE_EXPERIMENT_MMA_GROUPS must be 1 or 2")
        return value

    @classmethod
    def num_q_buffers(cls) -> int:
        value = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_Q_BUFFERS",
                str(cls.DEFAULT_NUM_Q_BUFFERS),
            )
        )
        if value not in (1, 2):
            raise ValueError("FLAG_GEMS_FA3_TLE_EXPERIMENT_Q_BUFFERS must be 1 or 2")
        return value

    @classmethod
    def make_config(
        cls,
        *,
        block_n: int,
        num_buffers_kv: int,
        block_m: int = DEFAULT_BLOCK_M,
        num_mma_groups: int | None = None,
        use_tma_qo: bool | None = None,
        use_tma_kv: bool = True,
    ):
        if num_mma_groups is None:
            num_mma_groups = cls.num_mma_groups()
        elif num_mma_groups not in (1, 2):
            raise ValueError("FLAG_GEMS_FA3_TLE_EXPERIMENT_MMA_GROUPS must be 1 or 2")
        num_buffers_q = cls.num_q_buffers()
        if use_tma_qo is None:
            default = "1" if cls.DEFAULT_USE_TMA_QO else "0"
            use_tma_qo = (
                os.getenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_TMA_QO", default) != "0"
            )
        reuse_default = "1" if cls.DEFAULT_REUSE_Q_SMEM_O else "0"
        q_pipe_default = "1" if cls.DEFAULT_Q_PIPE_ASYNC else "0"
        pipe_default = "1" if cls.DEFAULT_PAGED_PIPE_ASYNC else "0"
        return triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
                "NUM_BUFFERS_Q": num_buffers_q,
                "NUM_BUFFERS_KV": num_buffers_kv,
                "NUM_MMA_WARPS": 4 * num_mma_groups,
                "NUM_MMA_GROUPS": num_mma_groups,
                "Q_STAGE_CAPACITY": num_mma_groups * num_buffers_q,
                "USE_TMA_QO": use_tma_qo,
                "Q_PIPE_ASYNC": os.getenv(
                    "FLAG_GEMS_FA3_TLE_EXPERIMENT_Q_PIPE_ASYNC",
                    q_pipe_default,
                )
                != "0",
                "REUSE_Q_SMEM_O": os.getenv(
                    "FLAG_GEMS_FA3_TLE_EXPERIMENT_REUSE_Q_SMEM_O", reuse_default
                )
                == "1",
                "USE_TMA_KV": use_tma_kv,
                "PAGED_PIPE_ASYNC": os.getenv(
                    "FLAG_GEMS_FA3_TLE_EXPERIMENT_PIPE_ASYNC", pipe_default
                )
                != "0",
            },
            num_warps=4,
        )

    @classmethod
    def autotune_configs(cls):
        configs = []

        def add_transport_pair(**config_kwargs):
            configs.extend(
                cls.make_config(use_tma_kv=use_tma_kv, **config_kwargs)
                for use_tma_kv in (True, False)
            )

        add_transport_pair(block_n=128, num_buffers_kv=2)
        add_transport_pair(block_n=64, num_buffers_kv=2)
        forced_mma_groups = os.getenv("FLAG_GEMS_FA3_TLE_EXPERIMENT_MMA_GROUPS")
        decode_tma_default = "1" if cls.DEFAULT_DECODE_USE_TMA_QO else "0"
        decode_tma_qo = (
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_DECODE_TMA_QO",
                decode_tma_default,
            )
            == "1"
        )
        if forced_mma_groups in (None, "1"):
            add_transport_pair(
                block_m=64,
                block_n=64,
                num_buffers_kv=2,
                num_mma_groups=1,
                use_tma_qo=decode_tma_qo,
            )
            warp_mma_default = "1" if cls.DEFAULT_WARP_MMA else "0"
            if (
                os.getenv(
                    "FLAG_GEMS_FA3_TLE_EXPERIMENT_WARP_MMA",
                    warp_mma_default,
                )
                == "1"
            ):
                warp_mma_kv_buffers = (
                    int(
                        os.getenv(
                            "FLAG_GEMS_FA3_TLE_EXPERIMENT_KV_BUFFERS",
                            str(cls.DEFAULT_FORCED_KV_BUFFERS),
                        )
                    )
                    or 2
                )
                add_transport_pair(
                    block_m=16,
                    block_n=64,
                    num_buffers_kv=warp_mma_kv_buffers,
                    num_mma_groups=1,
                    use_tma_qo=False,
                )
                add_transport_pair(
                    block_m=16,
                    block_n=128,
                    num_buffers_kv=warp_mma_kv_buffers,
                    num_mma_groups=1,
                    use_tma_qo=False,
                )
            add_transport_pair(
                block_m=64,
                block_n=128,
                num_buffers_kv=2,
                num_mma_groups=1,
                use_tma_qo=decode_tma_qo,
            )

        forced_kv_buffers = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_KV_BUFFERS",
                str(cls.DEFAULT_FORCED_KV_BUFFERS),
            )
        )
        forced_block_m = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M",
                str(cls.DEFAULT_FORCED_BLOCK_M),
            )
        )
        forced_block_n = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_N",
                str(cls.DEFAULT_FORCED_BLOCK_N),
            )
        )
        if forced_kv_buffers not in (0, 2):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_KV_BUFFERS must be 0 or 2; "
                "capacity 1 is disabled because it produces invalid Hopper "
                "shared-memory operands"
            )
        if forced_kv_buffers == 2:
            add_transport_pair(block_n=64, num_buffers_kv=2)
        if forced_block_m not in (0, 16, 64, 128):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M must be 0, 16, 64, or " "128"
            )
        if forced_block_n not in (0, 64, 128):
            raise ValueError(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_N must be 0, 64, or 128"
            )
        if forced_block_n:
            configs = [
                config
                for config in configs
                if config.kwargs["BLOCK_N"] == forced_block_n
            ]
        if forced_kv_buffers:
            configs = [
                config
                for config in configs
                if config.kwargs["NUM_BUFFERS_KV"] == forced_kv_buffers
            ]
        if forced_block_m:
            configs = [
                config
                for config in configs
                if config.kwargs["BLOCK_M"] == forced_block_m
            ]
        return configs

    @staticmethod
    def config_smem_bytes(config, head_dim: int) -> int:
        block_k = _next_power_of_2(head_dim)
        block_m = config.kwargs["BLOCK_M"]
        block_n = config.kwargs["BLOCK_N"]
        num_groups = config.kwargs["NUM_MMA_GROUPS"]
        bm_split = block_m // num_groups
        q_elems = config.kwargs["Q_STAGE_CAPACITY"] * bm_split * block_k
        kv_elems = 2 * config.kwargs["NUM_BUFFERS_KV"] * block_n * block_k
        return (q_elems + kv_elems) * 2

    @classmethod
    def use_dense_kv_tma(
        cls,
        *,
        batch_size,
        num_heads,
        num_heads_k,
        head_dim,
        max_seqlen_q,
        max_seqlen_k,
        total_q,
        is_paged,
        is_causal,
        is_local,
        is_alibi,
        is_softcap,
        pack_gqa,
        ragged_scheduler,
        split_kv,
    ) -> bool:
        """Select dense K/V transport; True keeps the descriptor/TMA path."""

        if is_paged:
            return True
        required_shape = (
            batch_size,
            num_heads,
            num_heads_k,
            head_dim,
            max_seqlen_q,
            max_seqlen_k,
            total_q,
        )
        if any(value is None for value in required_shape):
            return True
        if (
            is_causal
            or is_local
            or is_alibi
            or is_softcap
            or ragged_scheduler
            or split_kv
            or total_q != batch_size * max_seqlen_q
            or max_seqlen_q % cls.DEFAULT_BLOCK_M != 0
            or head_dim > cls.DENSE_POINTER_MAX_D
            or max_seqlen_k > cls.DENSE_POINTER_MAX_K
        ):
            return True

        if pack_gqa:
            gqa_ratio = num_heads // num_heads_k
            effective_q = max_seqlen_q * gqa_ratio
            effective_heads = num_heads_k
        else:
            effective_q = max_seqlen_q
            effective_heads = num_heads
        persistent_tiles = (
            batch_size * effective_heads * _ceil_div(effective_q, cls.DEFAULT_BLOCK_M)
        )
        return persistent_tiles < cls.DENSE_POINTER_MIN_TILES

    @classmethod
    def prune_autotune_configs(cls, configs, nargs, **kwargs):
        head_dim = kwargs.get("d", nargs.get("d"))
        is_paged = kwargs.get("is_paged", nargs.get("is_paged"))
        paged_kv_non_tma = kwargs.get(
            "PAGED_KV_NON_TMA", nargs.get("PAGED_KV_NON_TMA", True)
        )
        pack_gqa = kwargs.get("PACK_GQA", nargs.get("PACK_GQA", False))
        split_kv = kwargs.get("SPLIT_KV", nargs.get("SPLIT_KV", False))
        block_size = kwargs.get("block_size", nargs.get("block_size", 1))
        seqlen_q = kwargs.get("seqlen_q", nargs.get("seqlen_q"))
        use_tma_kv = cls.use_dense_kv_tma(
            batch_size=kwargs.get("b", nargs.get("b")),
            num_heads=kwargs.get("h", nargs.get("h")),
            num_heads_k=kwargs.get("hk", nargs.get("hk")),
            head_dim=head_dim,
            max_seqlen_q=seqlen_q,
            max_seqlen_k=kwargs.get("seqlen_k", nargs.get("seqlen_k")),
            total_q=kwargs.get("total_q", nargs.get("total_q")),
            is_paged=is_paged,
            is_causal=kwargs.get("is_causal", nargs.get("is_causal", False)),
            is_local=kwargs.get("is_local", nargs.get("is_local", False)),
            is_alibi=kwargs.get("is_alibi", nargs.get("is_alibi", False)),
            is_softcap=kwargs.get("is_softcap", nargs.get("is_softcap", False)),
            pack_gqa=pack_gqa,
            ragged_scheduler=kwargs.get(
                "RAGGED_SCHEDULER", nargs.get("RAGGED_SCHEDULER", False)
            ),
            split_kv=split_kv,
        )
        decode_packgqa_ws = (
            is_paged
            and paged_kv_non_tma
            and pack_gqa
            and block_size in _COMPACT_PAGED_PROFILE_SIZES
            and seqlen_q == 1
            and head_dim in (192, 256)
        )
        spec_packgqa_ws = (
            is_paged
            and paged_kv_non_tma
            and pack_gqa
            and block_size in _COMPACT_PAGED_PROFILE_SIZES
            and 1 < seqlen_q <= 8
            and head_dim == 128
        )
        explicit_wide_split_ws = (
            split_kv
            and is_paged
            and paged_kv_non_tma
            and pack_gqa
            and block_size in (16, 32)
            and head_dim == 256
        )
        wide_paged_nontma = (
            is_paged and paged_kv_non_tma and not pack_gqa and head_dim > 128
        )

        def selected_shape_config(config):
            if decode_packgqa_ws or explicit_wide_split_ws or wide_paged_nontma:
                return (
                    config.kwargs["BLOCK_M"] == 64
                    and config.kwargs["BLOCK_N"] == 64
                    and config.kwargs["NUM_BUFFERS_KV"] == 2
                    and config.kwargs["NUM_MMA_GROUPS"] == 1
                    and not config.kwargs["USE_TMA_QO"]
                )
            if spec_packgqa_ws:
                return (
                    config.kwargs["BLOCK_M"] == 64
                    and config.kwargs["BLOCK_N"] == 128
                    and config.kwargs["NUM_BUFFERS_KV"] == 2
                    and config.kwargs["NUM_MMA_GROUPS"] == 1
                    and not config.kwargs["USE_TMA_QO"]
                )
            return config.kwargs["BLOCK_M"] == cls.DEFAULT_BLOCK_M

        kept = [
            config
            for config in configs
            if selected_shape_config(config)
            and config.kwargs["USE_TMA_KV"] == use_tma_kv
            and not (head_dim > 128 and config.kwargs["BLOCK_N"] > 64)
            and not (
                is_paged
                and paged_kv_non_tma
                and pack_gqa
                and config.kwargs["BLOCK_N"] < 128
                and not (decode_packgqa_ws or explicit_wide_split_ws)
            )
            and (
                not is_paged
                or paged_kv_non_tma
                or block_size % config.kwargs["BLOCK_N"] == 0
            )
            and cls.config_smem_bytes(config, head_dim) <= 220 * 1024
        ]
        if kept:
            return kept
        fallback = [
            config
            for config in reversed(configs)
            if config.kwargs["USE_TMA_KV"] == use_tma_kv
        ]
        return fallback[:1] or [configs[-1]]

    @classmethod
    def launch_plan(
        cls,
        *,
        heads_in_l2: HeadsInL2Policy,
        allow_head_swizzle: bool,
        pack_gqa: bool,
        gqa_ratio: int,
        effective_num_heads: int,
        max_seqlen_k: int,
        head_size: int,
        element_size: int,
        dynamic_scheduler: bool,
    ) -> PersistentLaunchPlan:
        num_mma_groups = cls.num_mma_groups()
        forced_block_m = int(
            os.getenv(
                "FLAG_GEMS_FA3_TLE_EXPERIMENT_BLOCK_M",
                str(cls.DEFAULT_FORCED_BLOCK_M),
            )
        )
        block_m = forced_block_m or cls.DEFAULT_BLOCK_M
        if not allow_head_swizzle:
            resolved_heads_in_l2 = CommonSchedulingHeuristics.binary_heads_in_l2(
                heads_in_l2
            )
        elif heads_in_l2.mode is HeadsInL2Mode.L2_AUTO:
            resolved_heads_in_l2 = 1
            l2_divisor = min(16, _next_power_of_2(gqa_ratio))
            l2_budget = cls.LPT_L2_BYTES // l2_divisor
            kv_block_bytes = (
                cls.LPT_HEURISTIC_BLOCK_N * (head_size + head_size) * element_size
            )
            max_kv_blocks_in_l2 = l2_budget // kv_block_bytes
            num_kv_blocks = _ceil_div(max_seqlen_k, cls.LPT_HEURISTIC_BLOCK_N)
            for candidate in (16, 8, 4, 2):
                if num_kv_blocks * candidate <= max_kv_blocks_in_l2:
                    resolved_heads_in_l2 = candidate
                    break
            if not pack_gqa:
                resolved_heads_in_l2 *= gqa_ratio
        else:
            resolved_heads_in_l2 = heads_in_l2.value
        resolved_heads_in_l2 = min(resolved_heads_in_l2, effective_num_heads)
        return PersistentLaunchPlan(
            num_mma_groups=num_mma_groups,
            block_m=block_m,
            heads_in_l2=resolved_heads_in_l2,
            dynamic_scheduler=dynamic_scheduler,
        )
