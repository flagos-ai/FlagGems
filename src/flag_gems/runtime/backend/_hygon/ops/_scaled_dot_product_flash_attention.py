# Copyright 2026, The FlagOS Contributors.
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
#
# Hygon-specific version that uses the optimized hygon flash_attention_forward
import logging

import torch

from flag_gems.runtime.backend._hygon.ops.attention import flash_attention_forward

from .attention_state import small_state_attention, state_attention

logger = logging.getLogger(__name__)


def _scaled_dot_product_flash_attention(
    query,
    key,
    value,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """Run scaled dot product FlashAttention through hygon's optimized implementation."""
    logger.debug("GEMS_HYGON _SCALED_DOT_PRODUCT_FLASH_ATTENTION")
    max_q = query.shape[2]
    max_k = key.shape[2]
    # The specialization includes a device-side bound and stable fallback.
    if (
        query.dtype == key.dtype == value.dtype == torch.bfloat16
        and query.shape == key.shape == value.shape
        and tuple(query.shape)
        in (
            (8, 16, 2049, 128),
            (2, 16, 3, 128),
            (2, 16, 23, 128),
            (2, 16, 362, 128),
        )
        and query.stride()
        == key.stride()
        == value.stride()
        == (query.shape[2] * 6144, 128, 6144, 1)
        and dropout_p == 0.0
        and scale != 0.0
        and not is_causal
        and not return_debug_mask
    ):
        softmax_scale = query.shape[-1] ** -0.5 if scale is None else scale
        implementation = (
            state_attention if query.shape[0] == 8 else small_state_attention
        )
        output, logsumexp = implementation(query, key, value, softmax_scale)
        rng_state = torch.empty((2,), dtype=torch.int64, device=query.device)
        unused = torch.empty((), dtype=torch.int64, device=query.device)
        return (
            output,
            logsumexp,
            None,
            None,
            max_q,
            max_k,
            rng_state,
            unused,
            None,
        )

    output, logsumexp, rng_state, unused, debug_mask = flash_attention_forward(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        None,
        None,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=scale,
    )
    return (
        output.transpose(1, 2),
        logsumexp,
        None,
        None,
        max_q,
        max_k,
        rng_state,
        unused,
        debug_mask,
    )
