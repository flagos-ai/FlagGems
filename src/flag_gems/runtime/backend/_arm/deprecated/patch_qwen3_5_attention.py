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

"""Monkey-patch F.scaled_dot_product_attention to route the M=1 BF16
decode path through the existing flash_attn_decode_bf16 TLE C kernel,
replacing the bmm + softmax + bmm sequence (9% of decode time per
profiler).

This is a minimal patch that only swaps SDPA — does NOT pull in the
full FlagGems _arm.ops package (which would also register Triton mm /
addmm kernels that are slower than ATen for our small decode shapes).

Other shapes (prefill, M>1, non-BF16, with attn_mask) fall through to
the original ATen SDPA without recursion (we capture the original
function pointer at patch time).
"""

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Imported at module load. If triton-cpu lacks the runtime module
# (older build), keep flash_attn_decode unavailable and fall through
# to ATen.
try:
    import triton
    import triton.language as tl
    from triton.language.extra.cpu.tle_ops import (
        flash_attn_decode as _tle_flash_attn_decode,
    )

    @triton.jit
    def _flash_attn_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        seq_len,
        head_dim: tl.constexpr,
        sm_scale: tl.constexpr,
        num_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_vn: tl.constexpr,
    ):
        # Coarse TLE op: the whole M=1 flash-attention decode in one launch.
        # seq_len is runtime (grows per token); the rest are constexpr.
        _tle_flash_attn_decode(
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            seq_len,
            head_dim,
            sm_scale,
            num_heads,
            num_kv_heads,
            stride_kn,
            stride_vn,
        )

except ImportError:
    _flash_attn_kernel = None

# Capture the *original* SDPA before any patching so our fallback
# call doesn't recurse.
_orig_sdpa = F.scaled_dot_product_attention

_PATCHED = False


def _patched_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """SDPA with M=1 BF16 fast path using flash_attn_decode_bf16."""
    if _flash_attn_kernel is None:
        return _orig_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    B, Hq, M, D = query.shape

    if (
        M == 1
        and B == 1
        and query.dtype == torch.bfloat16
        and attn_mask is None
        and dropout_p == 0.0
        and query.is_contiguous()
        and key.is_contiguous()
        and value.is_contiguous()
    ):
        Hkv = key.shape[1]
        seq_len = key.shape[2]
        sm_scale = scale if scale is not None else D**-0.5
        q_flat = query.squeeze(0).squeeze(1).contiguous()
        k_flat = key.squeeze(0).contiguous()
        v_flat = value.squeeze(0).contiguous()
        out_flat = torch.empty(Hq, D, dtype=torch.bfloat16)
        _flash_attn_kernel[(1,)](
            q_flat,
            k_flat,
            v_flat,
            out_flat,
            seq_len,
            head_dim=D,
            sm_scale=sm_scale,
            num_heads=Hq,
            num_kv_heads=Hkv,
            stride_kn=k_flat.stride(1),
            stride_vn=v_flat.stride(1),
        )
        return out_flat.unsqueeze(0).unsqueeze(2)

    # Non-decode shapes: fall back to original ATen SDPA.
    return _orig_sdpa(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def patch_qwen3_5_attention(model=None) -> int:
    """Install the patched SDPA. The `model` parameter is ignored and only
    accepted for API consistency with other patches.

    Returns 1 if installed (or already installed), 0 if flash_attn_decode
    is unavailable.
    """
    global _PATCHED
    if _flash_attn_kernel is None:
        logger.warning(
            "GEMS_ARM flash_attn_decode_bf16 not available; SDPA patch skipped"
        )
        return 0
    if _PATCHED:
        return 1
    F.scaled_dot_product_attention = _patched_sdpa
    _PATCHED = True
    logger.info(
        "GEMS_ARM Patched F.scaled_dot_product_attention with TLE flash_attn_decode_bf16"
    )
    return 1


def unpatch_qwen3_5_attention(model=None) -> int:
    global _PATCHED
    if not _PATCHED:
        return 0
    F.scaled_dot_product_attention = _orig_sdpa
    _PATCHED = False
    return 1
