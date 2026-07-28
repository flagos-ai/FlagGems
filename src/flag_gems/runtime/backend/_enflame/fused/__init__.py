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

from .. import arch_version

__all__ = []

if arch_version == 300:
    from .gcu300.concat_and_cache_mla import concat_and_cache_mla  # noqa: F401
    from .gcu300.cross_entropy_loss import cross_entropy_loss
    from .gcu300.flash_mla import flash_mla
    from .gcu300.fused_add_rms_norm import fused_add_rms_norm
    from .gcu300.gelu_and_mul import gelu_and_mul
    from .gcu300.moe_align_block_size import (  # noqa: F401
        moe_align_block_size,
        moe_align_block_size_triton,
    )
    from .gcu300.rotary_embedding import apply_rotary_pos_emb  # noqa: F401
    from .gcu300.silu_and_mul import silu_and_mul
    from .gcu300.skip_layernorm import skip_layer_norm

    __all__ = [
        "apply_rotary_pos_emb",
        "silu_and_mul",
        "gelu_and_mul",
        "cross_entropy_loss",
        "flash_mla",
        "skip_layer_norm",
        "fused_add_rms_norm",
        "concat_and_cache_mla",
        "moe_align_block_size",
        "moe_align_block_size_triton",
    ]
elif arch_version == 400 or arch_version == 410:
    from .gcu400.bincount import bincount
    from .gcu400.cross_entropy_loss import cross_entropy_loss
    from .gcu400.flash_mla import flash_mla
    from .gcu400.fused_add_rms_norm import fused_add_rms_norm
    from .gcu400.gelu_and_mul import gelu_and_mul
    from .gcu400.outer import outer
    from .gcu400.rotary_embedding import apply_rotary_pos_emb  # noqa: F401
    from .gcu400.silu_and_mul import silu_and_mul
    from .gcu400.skip_layernorm import skip_layer_norm
    from .gcu400.sparse_attention import sparse_attn_triton
    from .gcu400.sparse_mla import triton_sparse_mla_fwd_interface

    __all__ = [
        "apply_rotary_pos_emb",
        "gelu_and_mul",
        "silu_and_mul",
        "cross_entropy_loss",
        "flash_mla",
        "moe_sum",
        "outer",
        "fused_add_rms_norm",
        "skip_layer_norm",
        "rwkv_ka_fusion",
        "bincount",
        "sparse_attn_triton",
        "triton_sparse_mla_fwd_interface",
    ]
