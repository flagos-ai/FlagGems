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
from packaging import version

from flag_gems.ops.attention import scaled_dot_product_attention_forward

logger = logging.getLogger(__name__)

# 常量在模块加载时求值，避免每次调用重复解析版本号与超越函数。
_LN2 = math.log(2)
_TORCH_GE_29 = version.parse(torch.__version__.split("+")[0]).release[:2] >= (2, 9)
_METADATA_CACHE = {}


def _metadata_tensor(device, dtype, shape):
    """按 (device, dtype, shape) 复用只读的零值 metadata 张量。"""
    key = (str(device), dtype, tuple(shape))
    cached = _METADATA_CACHE.get(key)
    if cached is None:
        cached = torch.zeros(shape, device=device, dtype=dtype)
        _METADATA_CACHE[key] = cached
    return cached


def cudnn_attention_forward(
    query,
    key,
    value,
    attn_bias,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    compute_log_sumexp,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None,
):
    """Forward kernel for cuDNN attention -- BHSD-native path.

    Computes the cuDNN-style scaled dot-product attention forward pass.
    Inputs and outputs are in BHSD (batch, heads, seq, dim) layout.

    The attention output and log-sum-exp are computed via
    ``scaled_dot_product_attention_forward`` (Triton FlashAttention-2).
    The kernel tracks online-softmax state in log2 scale (``M * log(2)``
    recovers the natural-log LSE), so the O(N^2) post-processing matmul
    used in earlier versions is eliminated.
    """

    if dropout_p > 0.0:
        raise NotImplementedError(
            "cudnn_attention_forward: dropout > 0 is not yet supported"
        )

    if cum_seq_q is not None or cum_seq_k is not None:
        raise NotImplementedError(
            "cudnn_attention_forward: varlen is not yet supported"
        )

    out, m_log2 = scaled_dot_product_attention_forward(
        query,
        key,
        value,
        attn_mask=attn_bias,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )

    # M is the online-softmax accumulator in log2 scale
    # (the kernel uses exp2 / LOG2E internally).
    # Convert to natural-log LSE:  LSE_nat = M * ln(2).
    if compute_log_sumexp:
        lse = m_log2.float() * _LN2  # [B, H, S_q]
    else:
        lse = m_log2.new_zeros(m_log2.shape, dtype=torch.float32)

    # Match PyTorch aten::_cudnn_attention_forward behavior.
    # PyTorch < 2.9 returns [B, H, S_q], while PyTorch >= 2.9
    # returns [B, H, S_q, 1].
    if _TORCH_GE_29:
        lse = lse.unsqueeze_(-1)

    # Philox seed / offset: the Triton kernel does not produce these
    # (dropout is not supported), so return scalar zero tensors.
    zero_seed = _metadata_tensor(query.device, torch.int64, ())
    zero_offset = zero_seed

    # debug_attn_mask is not computed; return an empty tensor.
    debug_mask = _metadata_tensor(query.device, query.dtype, (0,))

    return (
        out,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        zero_seed,
        zero_offset,
        debug_mask,
    )
