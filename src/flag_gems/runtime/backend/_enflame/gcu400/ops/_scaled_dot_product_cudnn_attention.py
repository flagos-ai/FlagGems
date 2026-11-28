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

from flag_gems import runtime  # noqa: F401
from flag_gems.ops.attention import (
    scaled_dot_product_attention_forward as _generic_attention_forward,
)

from .attention import scaled_dot_product_attention_forward

# The backend kernel uses HEAD_DIM_ACTUAL to apply a zero-padding mask on head_dim inside the kernel, so it can directly
# consume
# non-power-of-2 head_dim (D=72 verified); therefore only extremely large head_dim values are left to the generic
# implementation.
_BACKEND_MAX_HEAD_DIM = 256

logger = logging.getLogger(__name__)

# The attention kernels accumulate the row log-sum-exp ``M`` in log2 base,
# while ``aten::_scaled_dot_product_cudnn_attention`` (and the cudnn/flash
# backward that consumes it) report logsumexp in the natural base. Rescale at
# the boundary so the autograd-saved LSE is numerically correct -- the same
# contract as ``_scaled_dot_product_efficient_attention``.
_LN2 = math.log(2.0)


def _scaled_dot_product_cudnn_attention(
    query,
    key,
    value,
    attn_bias=None,
    compute_log_sumexp=True,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """
    Scaled dot product attention using FlagGems Triton kernels.

    This function wraps the FlagGems attention implementation to provide
    the same interface as torch._scaled_dot_product_cudnn_attention.

    Args:
        query: Query tensor of shape (batch, num_heads, seq_len_q, head_dim)
        key: Key tensor of shape (batch, num_heads, seq_len_k, head_dim)
        value: Value tensor of shape (batch, num_heads, seq_len_k, head_dim)
        attn_bias: Optional attention bias tensor
        compute_log_sumexp: Whether to compute and return log-sum-exp values
        dropout_p: Dropout probability (currently must be 0.0)
        is_causal: Whether to apply causal masking
        return_debug_mask: Whether to return debug attention mask
        scale: Optional scale factor for attention scores

    Returns:
        tuple: (output, logsumexp, cum_seq_q, cum_seq_k, max_q, max_k,
                philox_seed, philox_offset, debug_attn_mask)
    """
    logger.debug("GEMS_ENFLAME _SCALED_DOT_PRODUCT_CUDNN_ATTENTION")

    # Always use the enflame backend's own gcu400 attention: on the host side, use broadcasted_stride
    # to collapse singleton/missing dims into zero strides, and route masks with "zero q-dim stride" ([B,1,1,Sk] /
    # [Sq,Sk] / expanded [B,H,Sq,Sk]) through MASK_Q_BROADCAST's 1D load branch.
    # The generic implementation flag_gems.ops.attention has no such branch; a 2D mask tile with zero stride
    # faults the GCU device (measured Aborted / core dumped), so it cannot serve as a fallback for
    # broadcastable bias; only shapes beyond the backend's head_dim capability fall back to the generic implementation.
    head_dim = query.shape[-1]
    if head_dim <= _BACKEND_MAX_HEAD_DIM:
        attention_forward = scaled_dot_product_attention_forward
    else:
        attention_forward = _generic_attention_forward

    # Only the backend's own attention forward supports pushing the ln2 conversion down into the kernel.
    lse_base_e_supported = attention_forward is scaled_dot_product_attention_forward

    # Convert attn_bias to attn_mask if provided
    attn_mask = attn_bias

    # Call the attention forward implementation selected above.
    # ``lse_base_e`` lets the backend fast path produce the natural-log LSE directly inside the kernel, saving one
    # elementwise launch of ``m * ln2`` (on small shapes this step is on par with the main kernel);
    # on non-fast paths, attention forward fills it in-place internally, with results consistent with host-side
    # conversion.
    forward_kwargs = {}
    if lse_base_e_supported:
        forward_kwargs["lse_base_e"] = bool(compute_log_sumexp)
    output, m = attention_forward(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=False,
        **forward_kwargs,
    )

    # The aten contract (and every backward that consumes it) expects the
    # natural-log LSE.
    if compute_log_sumexp:
        logsumexp = m if lse_base_e_supported else m * _LN2
    else:
        logsumexp = torch.empty(0, device=query.device, dtype=torch.float32)

    # Create placeholder tensors for cumulative sequence lengths
    # These are used for variable-length sequences, which we don't handle here
    cum_seq_q = torch.empty(0, device=query.device, dtype=torch.int32)
    cum_seq_k = torch.empty(0, device=query.device, dtype=torch.int32)

    # Get max sequence lengths
    max_q = query.shape[2]
    max_k = key.shape[2]

    # Philox seed and offset for dropout (currently not used as dropout_p must be 0)
    philox_seed = torch.empty((), device=query.device, dtype=torch.int64)
    philox_offset = torch.empty((), device=query.device, dtype=torch.int64)

    # Debug attention mask
    if return_debug_mask:
        # Create a debug mask of the appropriate shape
        batch = query.shape[0]
        num_heads = query.shape[1]
        seq_len_q = query.shape[2]
        seq_len_k = key.shape[2]
        debug_attn_mask = torch.zeros(
            (batch, num_heads, seq_len_q, seq_len_k),
            device=query.device,
            dtype=query.dtype,
        )
    else:
        debug_attn_mask = torch.empty(0, device=query.device, dtype=query.dtype)

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


# ---------------------------------------------------------------------------
# Each tile handles BLOCK_SIZE elements with a constant program count; an in-kernel grid-stride covers all tiles
# torch_gcu dispatch adaptation (aten::_scaled_dot_product_cudnn_attention has no GCU kernel)
#
# torch_gcu only registers a CPU fallback for this op (aten_cpu_fallback [backend fallback]);
# calling with GCU tensors moves them to CPU and then raises NotImplementedError - the accuracy framework and
# benchmark reference legs call torch.ops.aten.* directly without enable()/use_gems(), so the cudnn
# family's forward reference legs (tests/test_flash_attention_backward.py etc.) all fail.
# Following the existing pattern in `resize_as_sparse_.py` / `_flash_attention_forward.py`, register this
# implementation idempotently on the device key: an explicit kernel outranks the backend fallback and
# takes effect on import flag_gems; later duplicate registration of the same key by enable()/use_gems() is
# caught by the registrar as a warning and registers the same function, so behavior is unchanged.
# ---------------------------------------------------------------------------
_CUDNN_ATTN_FWD_KEY_FLAG = "_flag_gems_gcu400_cudnn_attn_fwd_key_registered"
_CUDNN_ATTN_FWD_LIBRARY = None
