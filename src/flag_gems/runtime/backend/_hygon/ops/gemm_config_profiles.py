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

"""Hardware tile domains with named scheduling profiles for gfx936 GEMM.

The YAML declares geometry and allowed launch dimensions. Profiles couple the
layout, copy policy and pipeline settings that implement one scheduling choice.
There are no runtime tensor dimensions, model names or benchmark winners here.
Only profile-specific allocation bounds and aliases with retained controls are
removed here. Compilation checks all remaining resource limits; mixed loads
must not be assigned a uniform buffer count.
"""

from functools import lru_cache
from itertools import product
from pathlib import Path

import triton
import yaml

from .gemm_utils import supports_hcu_options

# name, load mode, stages, K packing, transposed MMAC, epilogue, base waves, LDS slots
MM_PROFILES = (
    ("generic_min", 0, 1, 1, False, False, 1, 1),
    ("generic_pipe", 0, 2, 1, False, True, 4, 1),
    ("sync", 0, 2, 1, True, True, 4, 1),
    ("packed", 0, 2, 2, True, True, 4, 1),
    ("register_prefetch", 0, 3, 2, True, True, 4, 1),
    ("shared_prefetch", 0, 2, 1, True, True, 4, 2),
    ("async_a", 1, 2, 1, True, True, 4, 1),
    ("async_b", 2, 2, 1, True, True, 4, 1),
    ("async_pair", 3, 2, 1, True, True, 4, 1),
    ("async_deep", 3, 4, 2, True, True, 8, 1),
)

# name, load mode, stages, waves. Global partials write FP32 workspace;
# keep their epilogue policy separate from local reductions writing FP16/BF16.
GLOBAL_SPLIT_PROFILES = (
    ("global_latency", 0, 1, 2),
    ("global_bandwidth", 0, 2, 2),
    ("global_throughput", 0, 2, 4),
    ("global_async_b", 2, 3, 2),
)
LOCAL_SPLIT_PROFILES = (("local_sync_s1", 1), ("local_sync_s2", 2))


def _profiles(ranges, available):
    names = ranges["PROFILE"]
    unknown = set(names).difference(available)
    if unknown:
        raise ValueError(f"Unknown Hygon GEMM profiles: {sorted(unknown)}")
    return set(names)


def _waves(block_m, block_n, base):
    # Bound accumulator ownership per wave while retaining the profile's
    # producer/throughput tier on smaller tiles.
    minimum = max(1, block_m * block_n // (64 * 128))
    return min(8, max(minimum, min(base, block_m * block_n // 256)))


def _warps_m(block_m, block_n, waves):
    value = 1
    while value * 2 <= waves and (value * 2) ** 2 * block_n <= 2 * waves * block_m:
        value *= 2
    return max(max(1, waves // (block_n // 16)), min(value, waves, block_m // 16))


def _over_lds_capacity(name, bm, bn, bk, waves, stages, selected):
    a_bytes, b_bytes = 2 * bm * bk, 2 * bn * bk
    if name == "shared_prefetch":
        # An active two-load ring keeps both two-plane allocations live.
        # Otherwise its option is inactive and aliases the sync profile.
        return "sync" in selected and 2 * (a_bytes + b_bytes) > 65536
    if name not in ("async_a", "async_b", "async_pair", "async_deep"):
        return False
    # Both coalesced loads must own at least a dword per thread. Narrow
    # operands and weak alignment can change the number of admitted streams.
    if min(bm * bk, bn * bk) < 128 * waves:
        return False
    # The public async domain has contiguous, aligned half inputs. B-only
    # and paired profiles admit both streams and retain stages full planes.
    # A-only retains two A planes plus at least one B plane, either streamed
    # or allocated for the dot-layout conversion. Their loop lifetimes overlap.
    footprint = (
        2 * a_bytes + b_bytes if name == "async_a" else stages * (a_bytes + b_bytes)
    )
    return footprint > 65536


def _append(configs, seen, meta, waves, stages, ranges, pre_hook):
    if waves not in ranges["w"] or stages not in ranges["s"]:
        return
    key = (tuple(sorted(meta.items())), waves, stages)
    if key in seen:
        return
    seen.add(key)
    configs.append(
        triton.Config(meta, num_warps=waves, num_stages=stages, pre_hook=pre_hook)
    )


@lru_cache(maxsize=1)
def _default_mm_launches():
    """Reuse the backend's default geometry and launch coverage."""
    path = Path(__file__).parents[1] / "tune_configs.yaml"
    recipes = yaml.safe_load(path.read_text())["mm"]
    return tuple(
        sorted(
            {
                (
                    recipe["META"]["BLOCK_M"],
                    recipe["META"]["BLOCK_N"],
                    recipe["META"]["BLOCK_K"],
                    recipe["META"].get("GROUP_M", 8),
                    recipe.get("num_warps", 4),
                    recipe.get("num_stages", 1),
                )
                for recipe in recipes
            }
        )
    )


def _append_default_mm_baselines(configs, seen, ranges, supports_ring, pre_hook):
    # A minimal-wave profile does not replace the default launch choices:
    # operand registers can spill even when the accumulator fits one wave.
    # Keep the existing default geometry/launch recipes, with an ordinary
    # untransposed pipeline, as independent controls for the optimized profiles.
    for bm, bn, bk, group, waves, stages in _default_mm_launches():
        if any(
            value not in ranges[key]
            for key, value in (
                ("BLOCK_M", bm),
                ("BLOCK_N", bn),
                ("BLOCK_K", bk),
                ("GROUP_M", group),
            )
        ):
            continue
        meta = {
            "BLOCK_M": bm,
            "BLOCK_N": bn,
            "BLOCK_K": bk,
            "GROUP_M": group,
            "LOAD_MODE": 0,
            "ASYNC_LOAD": False,
            "mmac_warps_m": 0,
            "mmac_transposed": False,
            "optimize_epilogue": False,
            "schedule_hint": "none",
            "kpack": 1,
            "waves_per_eu": 1,
            "sched_latency": "none",
        }
        if supports_ring:
            meta["sync_lds_buffers"] = 1
        _append(configs, seen, meta, waves, stages, ranges, pre_hook)


def build_mm_configs(ranges, pre_hook=None):
    selected = _profiles(
        ranges, tuple(p[0] for p in MM_PROFILES) + ("default_baselines",)
    )
    if not supports_hcu_options((
        "mmac_warps_m", "mmac_transposed", "optimize_epilogue", "schedule_hint",
        "kpack", "waves_per_eu", "sched_latency",
    )):
        return []
    supports_ring = supports_hcu_options(("sync_lds_buffers",))
    configs, seen = [], set()
    for bm, bn, bk, group in product(
        ranges["BLOCK_M"], ranges["BLOCK_N"], ranges["BLOCK_K"], ranges["GROUP_M"]
    ):
        for (
            name,
            mode,
            stages,
            packing,
            transposed,
            epilogue,
            base,
            slots,
        ) in MM_PROFILES:
            if name not in selected or (slots == 2 and not supports_ring):
                continue
            waves = _waves(bm, bn, base)
            # A coalesced operand with exactly one element per thread is a
            # layout anchor and cannot form a 32-bit copy from FP16/BF16.
            # Its async request therefore aliases the retained sync/mixed
            # profile. It also prevents the two-load local-prefetch schedule
            # required by the synchronous two-slot profile. These profiles
            # share stages, packing and wave geometry with their controls.
            if stages == 2:
                scalar_operands = (bm * bk == 64 * waves) | (
                    (bn * bk == 64 * waves) << 1
                )
                if mode & scalar_operands or (slots == 2 and scalar_operands):
                    control = {0: "sync", 1: "async_a", 2: "async_b"}[
                        mode & ~scalar_operands
                    ]
                    if control in selected:
                        continue
            if _over_lds_capacity(name, bm, bn, bk, waves, stages, selected):
                continue
            meta = dict(
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                GROUP_M=group,
                LOAD_MODE=mode,
                ASYNC_LOAD=False,
                mmac_warps_m=_warps_m(bm, bn, waves) if transposed else 0,
                mmac_transposed=transposed,
                optimize_epilogue=epilogue,
                schedule_hint="none" if name == "generic_min" else "local-prefetch",
                kpack=1 if bk == 16 else packing,
                waves_per_eu=1,
                sched_latency="none",
            )
            if supports_ring:
                meta["sync_lds_buffers"] = slots
            _append(configs, seen, meta, waves, stages, ranges, pre_hook)
    if "default_baselines" in selected:
        _append_default_mm_baselines(configs, seen, ranges, supports_ring, pre_hook)
    return configs


def build_split_configs(ranges, pre_hook=None):
    selected = _profiles(
        ranges, [p[0] for p in GLOBAL_SPLIT_PROFILES + LOCAL_SPLIT_PROFILES]
    )
    if not supports_hcu_options(("optimize_epilogue", "schedule_hint")):
        return []
    configs, seen = [], set()
    for bm, bn, bk in product(ranges["BM"], ranges["BN"], ranges["BK"]):
        for splits in ranges["SPLITS"]:
            for name, mode, stages, waves in GLOBAL_SPLIT_PROFILES:
                if name not in selected:
                    continue
                if name == "global_latency":
                    # One wave per independent 16x16 MMAC output tile, capped
                    # at the latency profile's two-wave producer tier.
                    waves = _waves(bm, bn, waves)
                meta = dict(
                    BM=bm,
                    BN=bn,
                    BK=bk,
                    SPLITS=splits,
                    KIND=0,
                    LOAD_MODE=mode,
                    wave_private_sync=False,
                    optimize_epilogue=False,
                    schedule_hint="local-prefetch",
                )
                _append(configs, seen, meta, waves, stages, ranges, pre_hook)
        # This profile assigns one wave to each local K partition. Additional
        # logical partitions per wave are a different scheduling family.
        for splits in ranges["SPLITS"]:
            if splits <= 1 or splits not in ranges["w"]:
                continue
            for name, stages in LOCAL_SPLIT_PROFILES:
                if name not in selected:
                    continue
                meta = dict(
                    BM=bm,
                    BN=bn,
                    BK=bk,
                    SPLITS=splits,
                    KIND=1,
                    LOAD_MODE=0,
                    wave_private_sync=True,
                    optimize_epilogue=True,
                    schedule_hint="local-prefetch",
                )
                _append(configs, seen, meta, splits, stages, ranges, pre_hook)
    return configs
