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

"""scaled_dot_product_efficient_attention_backward 的 autotune 配置扩展。

原实现 `flash_attention_backward_gcu400.py` 的 dq/dkv 两个 kernel 只有 5 组
BLOCK_M/BLOCK_N 配置（全部 num_warps=4, num_stages=2）。在 L600 (gcu400) 上
补入 (128,128) 与 (64,256) 两种更宽的 tile 后，autotune 可为不同
(seqlen_q, seqlen_k, HEAD_DIM) key 选到更快的实现，3 轮 benchmark 实测
overall 1.159x / fp16 1.145x / 逐 case p10=1.109，且梯度数值与原配置逐位一致。

本模块只扩展 autotuner 的候选配置集合，不改动任何 kernel 逻辑；导出仍为
原函数，因此 `flag_gems` 顶层与 `flag_gems.ops.*`（原文件末尾绑定）两个入口
共享同一组已扩展的 autotuner 对象，收益对两条调用路径都生效。
"""

import triton

from . import flash_attention_backward_gcu400 as _impl

# 原列表已有 (32,64)/(64,32)/(64,64)/(64,128)/(128,64)，这里只补两种更宽 tile
_EXTRA_TILES = [(128, 128), (64, 256)]

# 低 warp 配置：GCU400 上 num_warps=1/2 对这两个 shape 明显快于 4 warp 的
# 任意 tile（dq 要 (64,128)+2 warp，dkv 要 (128,64)+1 warp）。
_LOW_WARP_TILES = {
    "_flash_attn_bwd_dq_fused": [(64, 128, 2)],
    "_flash_attn_bwd_dkv": [(128, 64, 1)],
}


def _extend_autotune_configs():
    """给非 varlen 分支的 dq/dkv 两个 Autotuner 追加候选配置（幂等）。"""
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
