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
Compiler resource errors remain authoritative; do not estimate shared-memory
capacity by assuming every operand always has the same buffer count.
"""

from itertools import product

import triton

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


def build_mm_configs(ranges, pre_hook=None):
    selected = _profiles(ranges, (p[0] for p in MM_PROFILES))
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        supports_ring = False
    else:
        supports_ring = "sync_lds_buffers" in HIPOptions.__dataclass_fields__
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
    return configs


def build_split_configs(ranges, pre_hook=None):
    selected = _profiles(
        ranges, [p[0] for p in GLOBAL_SPLIT_PROFILES + LOCAL_SPLIT_PROFILES]
    )
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
