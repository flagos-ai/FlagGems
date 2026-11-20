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

import logging

import torch

from flag_gems import runtime

from .attention import flash_attention_forward

logger = logging.getLogger(__name__)


def _materialize_bshd_view(t):
    """Return ``t`` (B, S, H, D) as a transpose view of a (B, H, S, D)-contiguous
    buffer, the layout the kernel is validated against (see #5896)."""
    bhsd = t.transpose(1, 2)
    if bhsd.is_contiguous():
        return t
    return bhsd.contiguous().transpose(1, 2)


def _flash_attention_forward(
    query,
    key,
    value,
    cumulative_sequence_length_q,
    cumulative_sequence_length_k,
    max_q,
    max_k,
    dropout_p,
    is_causal,
    return_debug_mask,
    *,
    scale=None,
    window_size_left=None,
    window_size_right=None,
    seqused_k=None,
    alibi_slopes=None,
):
    """Run the canonical ATen entry through the existing
    FlashAttention implementation.
    """
    logger.debug("GEMS_ENFLAME_GCU400 _FLASH_ATTENTION_FORWARD")
    query = _materialize_bshd_view(query)
    key = _materialize_bshd_view(key)
    value = _materialize_bshd_view(value)
    return flash_attention_forward(
        query,
        key,
        value,
        cumulative_sequence_length_q,
        cumulative_sequence_length_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        seqused_k=seqused_k,
        alibi_slopes=alibi_slopes,
    )


# ---------------------------------------------------------------------------
# torch_gcu dispatch adaptation (aten::_flash_attention_forward has no GCU kernel)
#
# This torch_gcu build only registers a CPU fallback for `aten::_flash_attention_forward`
# (aten_cpu_fallback.cpp [backend fallback]); calling it with GCU tensors moves them to CPU and then hits
# "no CPU kernel" with NotImplementedError, so processes outside FlagGems enable()/use_gems()
# (accuracy framework and benchmark reference legs that call torch.ops.aten.* directly) all fail.
# Following the existing pattern in `resize_as_sparse_.py`, register this implementation idempotently
# on the device key: an explicit kernel outranks the backend fallback and takes effect on import flag_gems.
# Later duplicate registrations of the same key by enable()/use_gems() are caught by the registrar as a warning;
# they register the same function (_FULL_CONFIG resolves to this file via SpecOpRegistrar),
# so behavior is unchanged.
# ---------------------------------------------------------------------------
_FLASH_ATTN_FWD_KEY_FLAG = "_flag_gems_gcu400_flash_attn_fwd_key_registered"
_FLASH_ATTN_FWD_LIBRARY = None


def _ensure_flash_attn_fwd_key_registration():
    global _FLASH_ATTN_FWD_LIBRARY
    if getattr(torch, _FLASH_ATTN_FWD_KEY_FLAG, False):
        return
    _FLASH_ATTN_FWD_LIBRARY = torch.library.Library("aten", "IMPL")
    try:
        _FLASH_ATTN_FWD_LIBRARY.impl(
            "_flash_attention_forward",
            _flash_attention_forward,
            runtime.device.dispatch_key,
        )
    except RuntimeError as error:
        logger.warning(
            "_flash_attention_forward device-key registration skipped: %s", error
        )
    setattr(torch, _FLASH_ATTN_FWD_KEY_FLAG, True)


_ensure_flash_attn_fwd_key_registration()
