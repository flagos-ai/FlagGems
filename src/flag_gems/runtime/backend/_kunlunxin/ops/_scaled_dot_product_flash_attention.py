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
import math

import torch

logger = logging.getLogger("flag_gems.ops._scaled_dot_product_flash_attention")

_FALLBACK_KEYSET = torch._C.DispatchKeySet(
    torch._C.DispatchKey.CompositeImplicitAutograd
)


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
    """Compute SDPA with safe Kunlunxin primitives.

    Both FlashAttention implementations hang for some non-square sequence
    lengths, and the FlagGems softmax kernel faults when the reduced dimension
    is not aligned. Keep the more accurate FlagGems matrix multiplications, but
    bypass the softmax override with the stable vendor implementation.
    """
    logger.debug("GEMS _SCALED_DOT_PRODUCT_FLASH_ATTENTION")
    assert dropout_p == 0.0, "Kunlunxin fallback only supports dropout_p=0.0"
    assert not return_debug_mask, "Kunlunxin fallback does not support debug masks"

    q_seq_len = query.shape[-2]
    kv_seq_len = key.shape[-2]
    softmax_scale = scale if scale is not None else 1.0 / math.sqrt(query.shape[-1])

    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores = scores * softmax_scale
    if is_causal:
        causal_bias = torch.full(
            (q_seq_len, kv_seq_len),
            float("-inf"),
            dtype=scores.dtype,
            device=query.device,
        )
        scores = scores + torch.triu(causal_bias, diagonal=1)

    logsumexp = torch.logsumexp(scores, dim=-1)
    probabilities = torch.ops.aten._softmax.default.redispatch(
        _FALLBACK_KEYSET, scores, -1, False
    )
    output = torch.matmul(probabilities, value.float()).to(query.dtype)

    philox_seed = torch.empty((), dtype=torch.int64, device=query.device)
    philox_offset = torch.empty((), dtype=torch.int64, device=query.device)
    debug_mask = torch.empty((0,), dtype=query.dtype, device=query.device)
    return (
        output,
        logsumexp,
        None,
        None,
        q_seq_len,
        kv_seq_len,
        philox_seed,
        philox_offset,
        debug_mask,
    )
