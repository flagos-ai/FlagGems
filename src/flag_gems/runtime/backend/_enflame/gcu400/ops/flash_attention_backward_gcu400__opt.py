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

"""Autotune config extension for scaled_dot_product_efficient_attention_backward.

The original `flash_attention_backward_gcu400.py` had only 5 groups of
BLOCK_M/BLOCK_N configs for the dq/dkv kernels (all num_warps=4, num_stages=2). On L600 (gcu400),
adding two wider tiles, (128,128) and (64,256), lets autotune pick a faster implementation for different
(seqlen_q, seqlen_k, HEAD_DIM) keys; 3 benchmark rounds measured
overall 1.159x / fp16 1.145x / per-case p10=1.109, with gradients bit-identical to the original configs.

This module only extends the autotuner's candidate config set and changes no kernel logic; the export is still
the original function, so both the `flag_gems` top level and `flag_gems.ops.*` entry points (bound at the
end of the original file) share the same extended autotuner object, and the gain applies to both call paths.
"""

import triton

from . import flash_attention_backward_gcu400 as _impl

# The original list already has (32,64)/(64,32)/(64,64)/(64,128)/(128,64); here only two wider tiles are added
_EXTRA_TILES = [(128, 128), (64, 256)]

# Low-warp configs: on GCU400 num_warps=1/2 is clearly faster than 4 warps
# for any tile on these two shapes (dq needs (64,128)+2 warp, dkv needs (128,64)+1 warp).
# Measured after the 2026-11-17 ds-split refactor (L600, (2,16,1024,64) fp16, do_bench 200ms):
# dkv's optimal configuration drifted to (128,128)+2warp=548.2µs / (128,64)+2warp=705.8µs,
# while the original (128,64)+1warp configuration was 793.8µs; all added in.
_LOW_WARP_TILES = {
    "_flash_attn_bwd_dq_fused": [(64, 128, 2)],
    "_flash_attn_bwd_dkv": [(128, 64, 1), (128, 64, 2), (128, 128, 2)],
}


def _extend_autotune_configs():
    """Append candidate configs to the two Autotuners of the non-varlen branch's dq/dkv (idempotent)."""
    for name in ("_flash_attn_bwd_dq_fused", "_flash_attn_bwd_dkv"):
        autotuner = getattr(_impl, name)
        existing = {
            (
                config.kwargs.get("BLOCK_M"),
                config.kwargs.get("BLOCK_N"),
                config.num_warps,
                config.num_stages,
            )
            for config in autotuner.configs
        }
        candidates = [(bm, bn, 4) for bm, bn in _EXTRA_TILES]
        candidates += _LOW_WARP_TILES[name]
        for block_m, block_n, num_warps in candidates:
            key = (block_m, block_n, num_warps, 2)
            if key in existing:
                continue
            autotuner.configs.append(
                triton.Config(
                    {"BLOCK_M": block_m, "BLOCK_N": block_n},
                    num_warps=num_warps,
                    num_stages=2,
                )
            )
            existing.add(key)


_extend_autotune_configs()

scaled_dot_product_efficient_attention_backward = (
    _impl.scaled_dot_product_efficient_attention_backward
)
