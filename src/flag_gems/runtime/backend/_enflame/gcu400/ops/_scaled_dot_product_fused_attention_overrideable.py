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

from .attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)


def _scaled_dot_product_fused_attention_overrideable(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: torch.Tensor = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    return_debug_mask: bool = False,
    scale: float = None,
):
    """gcu400 override of ``aten::_scaled_dot_product_fused_attention_overrideable``.

    Drives the backend's own fused attention kernel.  The generic
    implementation imports ``flag_gems.ops.attention.scaled_dot_product_attention_forward``
    at module load and therefore never used the gcu400 kernel; routing through
    the backend entry point keeps the operator on the same implementation as
    the rest of this backend (and bit-compatible with the device flash
    reference).
    """
    logger.debug("GEMS_ENFLAME _SCALED_DOT_PRODUCT_FUSED_ATTENTION_OVERRIDEABLE")
    assert dropout_p == 0.0, "Only dropout_p=0.0 is supported"

    output, logsumexp = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=(query.shape[1] != key.shape[1]),
    )

    batch_size = query.shape[0]
    seq_len_q = query.shape[2]
    seq_len_k = key.shape[2]

    # For non-variable-length attention, cumulative sequences are just [0, seq_len]
    cum_seq_q = torch.tensor(
        [0, seq_len_q] * batch_size,
        dtype=torch.int32,
        device=query.device,
    ).reshape(batch_size, 2)
    cum_seq_k = torch.tensor(
        [0, seq_len_k] * batch_size,
        dtype=torch.int32,
        device=key.device,
    ).reshape(batch_size, 2)

    max_q = seq_len_q
    max_k = seq_len_k

    # Philox seed and offset (dummy values since dropout_p=0)
    philox_seed = torch.tensor([0], dtype=torch.int64, device=query.device)
    philox_offset = torch.tensor([0], dtype=torch.int64, device=query.device)

    if return_debug_mask:
        debug_attn_mask = torch.zeros(
            (batch_size, query.shape[1], seq_len_q, seq_len_k),
            dtype=query.dtype,
            device=query.device,
        )
    else:
        debug_attn_mask = torch.empty(0, dtype=query.dtype, device=query.device)

    return (
        output,
        logsumexp,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        philox_seed,
        philox_offset,
        debug_attn_mask,
    )
