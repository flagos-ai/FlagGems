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
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# Elements per program: large enough for a wide contiguous store, small enough
# to keep the three live tiles (and their fp32 temporaries) resident.
_BLOCK = 4096


@libentry()
@triton.jit
def _transform_bias_rescale_qkv_kernel(
    qkv_ptr,
    bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    scale,
    stride_qkv_b,
    stride_qkv_s,
    stride_out_b,
    total,
    n_tiles,
    BLOCK: tl.constexpr,
    SEQ_HD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HIDDEN: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # One program owns BLOCK consecutive elements of one (batch, q|k|v) output.
    # The three outputs are each [batch, heads, seq, head_dim] contiguous, so a
    # flat output index maps back to the packed qkv row with one div/mod pair:
    # head*head_dim + d = (off // SEQ_HD) * HEAD_DIM + off % HEAD_DIM.
    pid = tl.program_id(0)
    batch = pid // n_tiles
    tile = pid % n_tiles

    off = tile * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        valid = off < total
        off = tl.minimum(off, total - 1)
    else:
        valid = None

    # The flat output index runs [head][seq][d] (the three outputs are
    # [batch, heads, seq, head_dim]); the packed qkv row runs [seq][q|k|v][head][d].
    head_off = (off // SEQ_HD) * HEAD_DIM + (off % HEAD_DIM)
    src_off = (
        qkv_ptr
        + batch * stride_qkv_b
        + ((off % SEQ_HD) // HEAD_DIM) * stride_qkv_s
        + head_off
    )
    dst_off = batch * stride_out_b + off

    if NEED_MASK:
        q = tl.load(src_off, mask=valid, other=0.0).to(tl.float32)
        q = q + tl.load(bias_ptr + head_off, mask=valid, other=0.0).to(tl.float32)
        tl.store(q_ptr + dst_off, (q * scale).to(q_ptr.dtype.element_ty), mask=valid)
    else:
        q = tl.load(src_off).to(tl.float32)
        q = q + tl.load(bias_ptr + head_off).to(tl.float32)
        tl.store(q_ptr + dst_off, (q * scale).to(q_ptr.dtype.element_ty))

    if NEED_MASK:
        k = tl.load(src_off + HIDDEN, mask=valid, other=0.0).to(tl.float32)
        k = k + tl.load(bias_ptr + HIDDEN + head_off, mask=valid, other=0.0).to(
            tl.float32
        )
        tl.store(k_ptr + dst_off, k.to(k_ptr.dtype.element_ty), mask=valid)
    else:
        k = tl.load(src_off + HIDDEN).to(tl.float32)
        k = k + tl.load(bias_ptr + HIDDEN + head_off).to(tl.float32)
        tl.store(k_ptr + dst_off, k.to(k_ptr.dtype.element_ty))

    if NEED_MASK:
        v = tl.load(src_off + 2 * HIDDEN, mask=valid, other=0.0).to(tl.float32)
        v = v + tl.load(bias_ptr + 2 * HIDDEN + head_off, mask=valid, other=0.0).to(
            tl.float32
        )
        tl.store(v_ptr + dst_off, v.to(v_ptr.dtype.element_ty), mask=valid)
    else:
        v = tl.load(src_off + 2 * HIDDEN).to(tl.float32)
        v = v + tl.load(bias_ptr + 2 * HIDDEN + head_off).to(tl.float32)
        tl.store(v_ptr + dst_off, v.to(v_ptr.dtype.element_ty))


def _transform_bias_rescale_qkv(qkv, qkv_bias, num_heads):
    """Bias-add and rescale the packed QKV projection of a ViT block.

    Args:
        qkv: [batch, seq_len, 3 * num_heads * head_dim]
        qkv_bias: [3 * num_heads * head_dim]
        num_heads: number of attention heads

    Returns:
        (q, k, v), each [batch, num_heads, seq_len, head_dim]; q is scaled by
        1 / sqrt(head_dim).
    """
    logger.debug("GEMS_KUNLUNXIN _TRANSFORM_BIAS_RESCALE_QKV")

    batch, seq_len, total_dim = qkv.shape
    hidden = total_dim // 3
    head_dim = hidden // num_heads
    if qkv.stride(2) != 1 or qkv.stride(1) != total_dim:
        qkv = qkv.contiguous()
    if not qkv_bias.is_contiguous():
        qkv_bias = qkv_bias.contiguous()

    shape = (batch, num_heads, seq_len, head_dim)
    q = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)
    v = torch.empty(shape, dtype=qkv.dtype, device=qkv.device)

    total = seq_len * hidden
    block = min(_BLOCK, triton.next_power_of_2(total))
    n_tiles = triton.cdiv(total, block)

    with torch_device_fn.device(qkv.device):
        _transform_bias_rescale_qkv_kernel[(batch * n_tiles,)](
            qkv,
            qkv_bias,
            q,
            k,
            v,
            head_dim**-0.5,
            qkv.stride(0),
            qkv.stride(1),
            q.stride(0),
            total,
            n_tiles,
            BLOCK=block,
            SEQ_HD=seq_len * head_dim,
            HEAD_DIM=head_dim,
            HIDDEN=hidden,
            NEED_MASK=(total % block) != 0,
            num_warps=4,
        )

    return q, k, v
