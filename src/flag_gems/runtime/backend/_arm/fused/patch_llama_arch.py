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

"""Apply Qwen3-equivalent TLE patches to HF Llama models (e.g. MiniCPM5-0.9B).

Llama interfaces are nearly identical to Qwen3 for the three op-density patches:
- apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)   — same sig
- LlamaRMSNorm.forward(self, hidden_states)               — same math (weight * x)
- LlamaDecoderLayer.forward                                — same structure

This file reuses the kernels from patch_qwen3_* and just retargets Llama's
modeling module.
"""

import logging

from flag_gems.runtime.backend._arm.fused.patch_qwen3_layer_norm import (
    _PATCHED as _LAYER_PATCHED,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_layer_norm import (
    _make_patched_forward as _make_layer_patched,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rmsnorm import (
    _PATCHED as _RMS_PATCHED,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rmsnorm import (
    _make_patched_forward as _make_rmsnorm_patched,
)

# Import the kernels + patch helpers from the Qwen3 patches.
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rope import (
    _PATCHED as _ROPE_PATCHED,
)
from flag_gems.runtime.backend._arm.fused.patch_qwen3_rope import (
    _patched_apply_rotary_pos_emb,
)

logger = logging.getLogger(__name__)

_LLAMA_MODULE = "transformers.models.llama.modeling_llama"


def patch_llama_rope() -> int:
    """Replace Llama apply_rotary_pos_emb with our TLE @triton.jit kernel."""
    try:
        mod = __import__(_LLAMA_MODULE, fromlist=["apply_rotary_pos_emb"])
    except (ImportError, AttributeError):
        logger.warning("GEMS_ARM Llama modeling module not available")
        return 0
    if not hasattr(mod, "apply_rotary_pos_emb"):
        return 0
    if _LLAMA_MODULE in _ROPE_PATCHED:
        return 0
    original = getattr(mod, "apply_rotary_pos_emb")
    _ROPE_PATCHED["original"] = original  # NOTE: shared with qwen3 patch
    _ROPE_PATCHED[_LLAMA_MODULE] = True
    setattr(mod, "apply_rotary_pos_emb", _patched_apply_rotary_pos_emb)
    logger.info(f"GEMS_ARM Patched {_LLAMA_MODULE}.apply_rotary_pos_emb")
    return 1


def patch_llama_rmsnorm() -> int:
    """Replace LlamaRMSNorm.forward with single Triton kernel."""
    try:
        mod = __import__(_LLAMA_MODULE, fromlist=["LlamaRMSNorm"])
    except (ImportError, AttributeError):
        return 0
    if not hasattr(mod, "LlamaRMSNorm"):
        return 0
    cls = getattr(mod, "LlamaRMSNorm")
    key = (_LLAMA_MODULE, "LlamaRMSNorm")
    if key in _RMS_PATCHED:
        return 0
    orig = cls.forward
    _RMS_PATCHED[key] = (cls, orig)
    cls.forward = _make_rmsnorm_patched(orig)
    logger.info(f"GEMS_ARM Patched {_LLAMA_MODULE}.LlamaRMSNorm.forward")
    return 1


def patch_llama_layer_norm() -> int:
    """Wire fused_add_rms_norm into LlamaDecoderLayer.forward.

    Note: the kernel reads `self.post_attention_layernorm.weight` and
    `.variance_epsilon` — both present on LlamaRMSNorm. ✓
    """
    try:
        mod = __import__(_LLAMA_MODULE, fromlist=["LlamaDecoderLayer"])
    except (ImportError, AttributeError):
        return 0
    if not hasattr(mod, "LlamaDecoderLayer"):
        return 0
    cls = getattr(mod, "LlamaDecoderLayer")
    key = (_LLAMA_MODULE, "LlamaDecoderLayer")
    if key in _LAYER_PATCHED:
        return 0
    orig = cls.forward
    _LAYER_PATCHED[key] = (cls, orig)
    cls.forward = _make_layer_patched(orig)
    logger.info(f"GEMS_ARM Patched {_LLAMA_MODULE}.LlamaDecoderLayer.forward")
    return 1


def patch_llama_arch() -> dict:
    """Apply all three patches; returns counts."""
    return {
        "rope": patch_llama_rope(),
        "rmsnorm": patch_llama_rmsnorm(),
        "layer_norm": patch_llama_layer_norm(),
    }
