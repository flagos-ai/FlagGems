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

"""AMD LDS (shared memory) aware autotune config pruning.

On AMD RDNA GPUs (e.g. gfx1201 / Radeon AI PRO R9700) the shared memory (LDS)
budget per workgroup is 64 KB, roughly half of what many data-center NVIDIA GPUs
offer. Matmul autotune configs that are fine for fp16/bf16 request twice as much
LDS when the operands are fp32, which can exceed the 64 KB limit and make the
Triton kernel fail to compile with:

    triton.runtime.errors.OutOfResources: out of resource: shared memory,
    Required: 131072, Hardware limit: 65536.

This module provides a dtype-aware ``early_config_prune`` hook that drops the
configs whose estimated LDS footprint exceeds the device limit, while leaving
fp16/bf16 configs untouched (their footprint is half and stays within budget).
"""

import functools
import logging

logger = logging.getLogger(__name__)

# Conservative fallback used when the device limit cannot be queried.
_DEFAULT_LDS_LIMIT_BYTES = 65536


@functools.lru_cache(maxsize=None)
def get_shared_memory_limit(device: int = 0) -> int:
    """Return the per-workgroup shared memory (LDS) limit in bytes."""
    try:
        import triton

        props = triton.runtime.driver.active.utils.get_device_properties(device)
        limit = int(props.get("max_shared_mem", _DEFAULT_LDS_LIMIT_BYTES))
        if limit > 0:
            return limit
    except Exception:  # pragma: no cover - defensive, keep default on any error
        pass
    return _DEFAULT_LDS_LIMIT_BYTES


def _estimate_matmul_lds_bytes(meta, num_stages: int, elem_size: int) -> int:
    """Estimate the LDS bytes a matmul config needs.

    Triton pipelines the A and B operand tiles across ``num_stages`` buffers, so
    the shared memory footprint is roughly::

        num_stages * (BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N) * elem_size

    This mirrors the quantity that triggers the OutOfResources error and is a
    safe upper bound for pruning purposes.
    """
    block_m = meta.get("BLOCK_M", 0)
    block_n = meta.get("BLOCK_N", 0)
    block_k = meta.get("BLOCK_K", 0)
    tile_elems = block_m * block_k + block_k * block_n
    return max(num_stages, 1) * tile_elems * elem_size


def _synthesize_safe_matmul_configs(configs, elem_size, limit):
    """Build small-tile configs that fit within the LDS ``limit``.

    Used as a last resort when every tuned config overflows the shared memory
    budget for the current dtype (e.g. fp32 on a 64 KB-LDS AMD GPU).
    """
    import triton

    # Candidate tiles ordered from larger (faster) to smaller (safer).
    candidate_tiles = [
        (128, 64, 32, 2, 4),
        (64, 128, 32, 2, 4),
        (64, 64, 32, 3, 4),
        (64, 64, 32, 2, 4),
        (32, 64, 32, 2, 4),
        (32, 32, 32, 2, 4),
    ]
    safe = []
    for bm, bn, bk, ns, nw in candidate_tiles:
        meta = {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk}
        if _estimate_matmul_lds_bytes(meta, ns, elem_size) <= limit:
            safe.append(triton.Config(meta, num_stages=ns, num_warps=nw))
    if not safe:
        # Extremely tight budget: fall back to the smallest tuned config.
        safe = [
            min(
                configs,
                key=lambda c: _estimate_matmul_lds_bytes(
                    getattr(c, "kwargs", {}),
                    getattr(c, "num_stages", 1),
                    elem_size,
                ),
            )
        ]
    return safe


def prune_matmul_configs_by_lds(configs, named_args, operand_keys=("A", "B"), **kwargs):
    """``early_config_prune`` hook: drop matmul configs that overflow LDS.

    Args:
        configs: candidate ``triton.Config`` objects.
        named_args: mapping from kernel arg name to the concrete argument;
            used to look up the operand tensors and their dtype.
        operand_keys: kernel arg names of the matmul operands to size against.
    """
    configs = list(configs)

    # Determine operand element size from the actual input tensors.
    elem_size = None
    for key in operand_keys:
        tensor = named_args.get(key)
        dtype = getattr(tensor, "dtype", None)
        itemsize = getattr(dtype, "itemsize", None)
        if itemsize is not None:
            elem_size = max(elem_size or 0, int(itemsize))
    if not elem_size:
        # Unknown dtype: do not prune, preserve original behavior.
        return configs

    limit = get_shared_memory_limit()
    kept = []
    for config in configs:
        meta = getattr(config, "kwargs", {})
        num_stages = getattr(config, "num_stages", 1)
        lds = _estimate_matmul_lds_bytes(meta, num_stages, elem_size)
        if lds <= limit:
            kept.append(config)

    if not kept:
        # None of the tuned configs fit the LDS budget for this dtype (this is
        # the fp32-on-64KB-LDS case on gfx1201). Synthesize small-tile configs
        # that are guaranteed to fit so the op can still run.
        kept = _synthesize_safe_matmul_configs(configs, elem_size, limit)

    pruned = len(configs) - len(kept)
    if pruned > 0:
        logger.debug(
            "AMD LDS prune: dropped %d/%d matmul configs "
            "(elem_size=%d, limit=%d bytes)",
            pruned,
            len(configs),
            elem_size,
            limit,
        )
    return kept


def matmul_lds_prune_configs_by(operand_keys=("A", "B")):
    """Return a ``prune_configs_by`` dict for AMD, or ``None`` elsewhere.

    Gated on the active FlagGems vendor so that only AMD backends receive the
    dtype-aware LDS pruning; all other backends keep their original behavior.
    """
    try:
        from flag_gems.runtime import device

        vendor = getattr(device, "vendor_name", None)
    except Exception:  # pragma: no cover - defensive
        vendor = None

    if vendor != "amd":
        return None

    def _early_config_prune(configs, named_args, **kwargs):
        return prune_matmul_configs_by_lds(
            configs, named_args, operand_keys=operand_keys, **kwargs
        )

    return {"early_config_prune": _early_config_prune}
