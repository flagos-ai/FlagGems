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

"""Dense cross attention with the FlashAttentionScore input contract."""

import logging
import math
import numbers

import torch
import triton
import triton.experimental.tle as tle
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_MASK_DTYPES = (torch.bool, torch.uint8)
_MAX_BATCH = 2 * 1024 * 1024
_MAX_HEADS = 256
_MAX_SEQUENCE = 1024 * 1024
_MAX_HEAD_DIM = 768


def _select_910b_tiling(  # 910B4 分块策略
    query_len,
    key_len,
    qk_dim,
    value_dim,
    has_mask=False,
    low_precision=True,
):
    """Offline tiling derived from 910B4 kernel-only profiling.

    Long K/V sequences use a wide N tile to reduce online-softmax iterations.
    Larger head dimensions progressively reduce both axes to stay within UB.
    """
    largest_dim = max(qk_dim, value_dim)
    if key_len <= 128:
        # Match the K/V tile to a short sequence so power-of-two lengths use
        # the one-pass template instead of the generic tail state machine.
        block_n = max(16, triton.next_power_of_2(key_len))
        if largest_dim > 128:
            return 32, block_n
        if largest_dim > 64:
            return 32, block_n
        return 64, block_n
    if largest_dim > 512:
        return 16, 32
    if largest_dim > 256:
        return 32, 64
    if largest_dim > 128:
        # Q and the fp32 output accumulator both scale with M*D.  M=64 at
        # D=256 needs about 274 KiB after the compiler's double-buffered
        # temporaries, exceeding the 192 KiB UB on 910B4.  M=32 leaves room
        # for mask/softmax temporaries and applies to every D in (128, 256].
        if key_len <= 256 and query_len % 16 == 0:
            return 32, 256
        return 32, 64
    if query_len <= 32:
        return 16, 32
    if (
        64 < largest_dim <= 128
        and key_len <= 512
        and (key_len & (key_len - 1)) == 0
        and query_len % 32 == 0
        and not has_mask
    ):
        # A power-of-two K/V sequence can use the one-pass template when its
        # resident Q/output state leaves enough UB.  Express this as capacity
        # and alignment constraints rather than a shape-specific length.
        return 32, key_len
    if key_len >= 256:
        # Small heads leave enough UB for twice as many query rows.  M=128
        # halves the number of resident-query programs for arbitrary long
        # sequences.  The aligned low-precision mask specialization omits the
        # redundant query-row predicate and also fits M128; fp32 and partial
        # query/key tiles retain M64 headroom for predicates and tail buffers.
        if largest_dim <= 64:
            if query_len % 128 == 0 and key_len % 128 == 0 and (
                not has_mask
                or low_precision
            ):
                return 128, 128
            return 64, 128
        # For a multi-tile query with a <=128 head dimension, M=64 keeps
        # enough independent programs in flight and is measurably faster than
        # M=32/N=256 on 910B4.  Keep the wider N tile for shorter queries,
        # where it avoids an extra online-softmax iteration.
        if query_len > 128 and largest_dim <= 128:
            return 64, 128
        # A second, smaller tail tile needs its own local buffers.  On 910B4,
        # larger M tiles combined with the independent tail exceed 192 KiB UB.
        return 32, 256
    if query_len >= 128 and key_len >= 128:
        # M=128, N=128 overflows the 910B4 UB even for D=64 (by 1536
        # bytes in the no-mask specialization).  Keep N=128, but reduce M
        # according to the resident Q/output state required by the head dim.
        return (64, 128) if largest_dim <= 64 else (32, 128)
    return 64, 128


def _validate_inputs(query, key, value, attn_mask, scale):
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 4:
            raise ValueError(
                f"{name} must be 4-dimensional in BNSD layout, got "
                f"{tensor.ndim} dimensions"
            )

    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise TypeError("query, key, and value must have the same dtype")
    if query.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "cross_attention supports only torch.float16, torch.bfloat16, "
            f"and torch.float32, got {query.dtype}"
        )

    batch_q, heads_q, query_len, dim_q = query.shape
    batch_k, heads_k, key_len, dim_k = key.shape
    batch_v, heads_v, value_len, dim_v = value.shape

    if not 1 <= batch_q <= _MAX_BATCH:
        raise ValueError(f"batch size must be in [1, {_MAX_BATCH}], got {batch_q}")
    if batch_q != batch_k or batch_q != batch_v:
        raise ValueError("query, key, and value batch sizes must match")
    if not 1 <= heads_q <= _MAX_HEADS:
        raise ValueError(f"query head count must be in [1, {_MAX_HEADS}], got {heads_q}")
    if heads_k != heads_v or heads_k <= 0 or heads_q % heads_k != 0:
        raise ValueError(
            "key/value head counts must match and query_heads / kv_heads must "
            "be a non-zero integer"
        )
    if not 1 <= query_len <= _MAX_SEQUENCE:
        raise ValueError(
            f"query sequence length must be in [1, {_MAX_SEQUENCE}], got {query_len}"
        )
    if not 1 <= key_len <= _MAX_SEQUENCE:
        raise ValueError(
            f"key sequence length must be in [1, {_MAX_SEQUENCE}], got {key_len}"
        )
    if key_len != value_len:
        raise ValueError("key and value sequence lengths must match")
    if dim_q != dim_k or dim_k < dim_v:
        raise ValueError("head dimensions must satisfy query_D == key_D >= value_D")
    if not 1 <= dim_q <= _MAX_HEAD_DIM or not 1 <= dim_v <= _MAX_HEAD_DIM:
        raise ValueError(
            f"query/key and value head dimensions must be in [1, {_MAX_HEAD_DIM}]"
        )

    if attn_mask is not None:
        if not isinstance(attn_mask, torch.Tensor):
            raise TypeError("attn_mask must be a torch.Tensor or None")
        if attn_mask.device != query.device:
            raise ValueError("attn_mask must be on the same device as query")
        if attn_mask.dtype not in _SUPPORTED_MASK_DTYPES:
            raise TypeError("attn_mask dtype must be torch.bool or torch.uint8")
        allowed_shapes = {
            (query_len, key_len),
            (1, 1, query_len, key_len),
            (batch_q, 1, query_len, key_len),
            (batch_q, heads_q, query_len, key_len),
        }
        if tuple(attn_mask.shape) not in allowed_shapes:
            allowed = ", ".join(str(shape) for shape in sorted(allowed_shapes))
            raise ValueError(
                f"attn_mask shape must be one of {allowed}, got {tuple(attn_mask.shape)}"
            )

    if scale is not None:
        if isinstance(scale, bool) or not isinstance(scale, numbers.Real):
            raise TypeError("scale must be a finite real number or None")
        if not math.isfinite(float(scale)):
            raise ValueError(f"scale must be finite, got {scale}")


def _mask_strides(attn_mask):
    if attn_mask.ndim == 2:
        return 0, 0, attn_mask.stride(0), attn_mask.stride(1)
    strides = list(attn_mask.stride())
    # FlashAttentionScore broadcasts only the leading B/N singleton axes.
    # Encode that broadcast directly as zero strides; do not materialize BHSS.
    if attn_mask.shape[0] == 1:
        strides[0] = 0
    if attn_mask.shape[1] == 1:
        strides[1] = 0
    return tuple(strides)


@triton.jit
def _cross_attention_tile(
    accumulator,
    row_max,
    row_sum,
    query,
    K,
    V,
    key_base_offset,
    value_base_offset,
    AttnMask,
    mask_row_offset,
    stride_k_seq,
    stride_k_dim,
    stride_v_seq,
    stride_v_dim,
    stride_mask_key,
    scale,
    query_offsets,
    qk_offsets,
    value_offsets,
    key_start,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    N_BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    IS_TAIL: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    key_offsets = tl.arange(0, N_BLOCK)
    logical_keys = key_start + key_offsets
    valid_queries = query_offsets < QUERY_LEN
    if IS_TAIL:
        valid_keys = logical_keys < KEY_LEN
    else:
        valid_keys = key_offsets < N_BLOCK

    key_ptrs = (
        K
        + key_base_offset
        + logical_keys[None, :] * stride_k_seq
        + qk_offsets[:, None] * stride_k_dim
    )
    key = tl.load(
        key_ptrs,
        mask=(qk_offsets[:, None] < QK_DIM) & valid_keys[None, :],
        other=0.0,
    )

    value_ptrs = (
        V
        + value_base_offset
        + logical_keys[:, None] * stride_v_seq
        + value_offsets[None, :] * stride_v_dim
    )
    if PRE_LOAD_V:
        value = tl.load(
            value_ptrs,
            mask=valid_keys[:, None] & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        )

    log2e: tl.constexpr = 1.44269504
    scores = tl.dot(query, key, allow_tf32=False)
    tle.dsa.ascend.compile_hint(scores, "hivm.tile_mix_cube_num", 2)
    scores = scores * scale * log2e
    score_is_valid = valid_queries[:, None] & valid_keys[None, :]
    if HAS_MASK:
        mask_ptrs = (
            AttnMask
            + mask_row_offset
            + logical_keys[None, :] * stride_mask_key
        )
        blocked = tl.load(
            mask_ptrs,
            mask=valid_queries[:, None] & valid_keys[None, :],
            other=1,
        ) != 0
        score_is_valid = score_is_valid & ~blocked

    scores = tl.where(score_is_valid, scores, -float("inf"))
    tile_max = tl.max(scores, axis=1)
    new_max = tl.maximum(row_max, tile_max)
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    probabilities = tl.math.exp2(
        tl.where(score_is_valid, scores - safe_max[:, None], -float("inf"))
    )
    alpha = tl.where(
        row_max == -float("inf"),
        0.0,
        tl.math.exp2(row_max - safe_max),
    )
    row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)

    if not PRE_LOAD_V:
        value = tl.load(
            value_ptrs,
            mask=valid_keys[:, None]
            & (value_offsets[None, :] < VALUE_DIM),
            other=0.0,
        )
    probabilities = probabilities.to(value.dtype)
    accumulator *= alpha[:, None]
    accumulator = tl.dot(
        probabilities,
        value,
        accumulator,
        allow_tf32=False,
    )
    tle.dsa.ascend.compile_hint(accumulator, "hivm.tile_mix_cube_num", 2)
    return accumulator, new_max, row_sum


@triton.jit
def _cross_attention_inner_regular(
    accumulator,
    row_max,
    row_sum,
    query,
    K,
    V,
    key_base_offset,
    value_base_offset,
    AttnMask,
    mask_row_offset,
    stride_k_seq,
    stride_k_dim,
    stride_v_seq,
    stride_v_dim,
    stride_mask_key,
    scale,
    query_offsets,
    qk_offsets,
    value_offsets,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    key_offsets = tl.arange(0, BLOCK_N)
    valid_queries = query_offsets < QUERY_LEN
    for key_start in tle.dsa.pipeline(0, KEY_LEN, BLOCK_N, num_stages=2):
        logical_keys = key_start + key_offsets
        key = tl.load(
            K
            + key_base_offset
            + logical_keys[None, :] * stride_k_seq
            + qk_offsets[:, None] * stride_k_dim,
            mask=qk_offsets[:, None] < QK_DIM,
            other=0.0,
        )
        scores = tl.dot(query, key, allow_tf32=False)
        tle.dsa.ascend.compile_hint(scores, "hivm.tile_mix_cube_num", 2)
        scores = scores * scale * 1.44269504
        if not HAS_MASK and QUERY_LEN % BLOCK_M == 0:
            # Full query tiles have no invalid rows.  Avoid a dense predicate
            # and fully-masked-row fix-up, which also makes M128/N128 fit UB.
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, tile_max)
            safe_max = new_max
            probabilities = tl.math.exp2(scores - new_max[:, None])
        else:
            if (
                HAS_MASK
                and QK_DIM <= 64
                and VALUE_DIM <= 64
                and QUERY_LEN % BLOCK_M == 0
            ):
                blocked = tl.load(
                    AttnMask
                    + mask_row_offset
                    + logical_keys[None, :] * stride_mask_key
                ) != 0
                score_is_valid = ~blocked
            else:
                score_is_valid = valid_queries[:, None]
            if HAS_MASK and not (
                QK_DIM <= 64
                and VALUE_DIM <= 64
                and QUERY_LEN % BLOCK_M == 0
            ):
                blocked = tl.load(
                    AttnMask
                    + mask_row_offset
                    + logical_keys[None, :] * stride_mask_key,
                    mask=score_is_valid,
                    other=1,
                ) != 0
                score_is_valid = score_is_valid & ~blocked

            scores = tl.where(score_is_valid, scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            new_max = tl.maximum(row_max, tile_max)
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.math.exp2(
                tl.where(
                    score_is_valid,
                    scores - safe_max[:, None],
                    -float("inf"),
                )
            )
        alpha = tl.where(
            row_max == -float("inf"),
            0.0,
            tl.math.exp2(row_max - safe_max),
        )
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        value = tl.load(
            V
            + value_base_offset
            + logical_keys[:, None] * stride_v_seq
            + value_offsets[None, :] * stride_v_dim,
            mask=value_offsets[None, :] < VALUE_DIM,
            other=0.0,
        )
        accumulator *= alpha[:, None]
        accumulator = tl.dot(
            probabilities.to(value.dtype),
            value,
            accumulator,
            allow_tf32=False,
        )
        tle.dsa.ascend.compile_hint(
            accumulator, "hivm.tile_mix_cube_num", 2
        )
        row_max = new_max
    return accumulator, row_max, row_sum


@triton.jit
def _cross_attention_single_tile(
    query,
    K,
    V,
    key_base_offset,
    value_base_offset,
    AttnMask,
    mask_row_offset,
    stride_k_seq,
    stride_k_dim,
    stride_v_seq,
    stride_v_dim,
    stride_mask_key,
    scale,
    query_offsets,
    qk_offsets,
    value_offsets,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    """Fast path matching the one-pass Ascend C tiling template.

    With one K/V tile there is no cross-tile softmax state to update.  Keeping
    that state would consume UB and add an unnecessary alpha rescale.
    """
    key_offsets = tl.arange(0, BLOCK_N)
    valid_queries = query_offsets < QUERY_LEN
    valid_keys = key_offsets < KEY_LEN
    key = tl.load(
        K
        + key_base_offset
        + key_offsets[None, :] * stride_k_seq
        + qk_offsets[:, None] * stride_k_dim,
        mask=(qk_offsets[:, None] < QK_DIM) & valid_keys[None, :],
        other=0.0,
    )
    scores = tl.dot(query, key, allow_tf32=False)
    tle.dsa.ascend.compile_hint(scores, "hivm.tile_mix_cube_num", 2)
    scores = scores * scale * 1.44269504
    if not HAS_MASK and QUERY_LEN % BLOCK_M == 0:
        # Every row/key is valid in this specialization, so avoid materializing
        # a BLOCK_M x BLOCK_N predicate and the fully-masked-row fix-up.
        row_max = tl.max(scores, axis=1)
        probabilities = tl.math.exp2(scores - row_max[:, None])
    else:
        score_is_valid = valid_queries[:, None] & valid_keys[None, :]
        if HAS_MASK:
            blocked = tl.load(
                AttnMask
                + mask_row_offset
                + key_offsets[None, :] * stride_mask_key,
                mask=score_is_valid,
                other=1,
            ) != 0
            score_is_valid = score_is_valid & ~blocked
        scores = tl.where(score_is_valid, scores, -float("inf"))
        row_max = tl.max(scores, axis=1)
        safe_max = tl.where(row_max == -float("inf"), 0.0, row_max)
        probabilities = tl.math.exp2(
            tl.where(
                score_is_valid,
                scores - safe_max[:, None],
                -float("inf"),
            )
        )
    denominator = tl.sum(probabilities, axis=1)
    probabilities *= tl.where(
        denominator > 0.0, 1.0 / denominator, 0.0
    )[:, None]
    value = tl.load(
        V
        + value_base_offset
        + key_offsets[:, None] * stride_v_seq
        + value_offsets[None, :] * stride_v_dim,
        mask=valid_keys[:, None] & (value_offsets[None, :] < VALUE_DIM),
        other=0.0,
    )
    output = tl.dot(
        probabilities.to(value.dtype), value, allow_tf32=False
    )
    tle.dsa.ascend.compile_hint(output, "hivm.tile_mix_cube_num", 2)
    return output


@triton.jit
def _cross_attention_inner(
    accumulator,
    row_max,
    row_sum,
    query,
    K,
    V,
    key_base_offset,
    value_base_offset,
    AttnMask,
    mask_row_offset,
    stride_k_seq,
    stride_k_dim,
    stride_v_seq,
    stride_v_dim,
    stride_mask_key,
    scale,
    query_offsets,
    qk_offsets,
    value_offsets,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAIN_KEY_LEN: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    TAIL_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    HAS_MASK: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    if PIPELINE_STAGES == 1:
        for key_start in range(0, MAIN_KEY_LEN, BLOCK_N):
            accumulator, row_max, row_sum = _cross_attention_tile(
                accumulator,
                row_max,
                row_sum,
                query,
                K,
                V,
                key_base_offset,
                value_base_offset,
                AttnMask,
                mask_row_offset,
                stride_k_seq,
                stride_k_dim,
                stride_v_seq,
                stride_v_dim,
                stride_mask_key,
                scale,
                query_offsets,
                qk_offsets,
                value_offsets,
                key_start,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                QK_DIM=QK_DIM,
                VALUE_DIM=VALUE_DIM,
                N_BLOCK=BLOCK_N,
                HAS_MASK=HAS_MASK,
                IS_TAIL=False,
                PRE_LOAD_V=PRE_LOAD_V,
            )
    else:
        for key_start in tle.dsa.pipeline(
            0, MAIN_KEY_LEN, BLOCK_N, num_stages=PIPELINE_STAGES
        ):
            accumulator, row_max, row_sum = _cross_attention_tile(
                accumulator,
                row_max,
                row_sum,
                query,
                K,
                V,
                key_base_offset,
                value_base_offset,
                AttnMask,
                mask_row_offset,
                stride_k_seq,
                stride_k_dim,
                stride_v_seq,
                stride_v_dim,
                stride_mask_key,
                scale,
                query_offsets,
                qk_offsets,
                value_offsets,
                key_start,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                QK_DIM=QK_DIM,
                VALUE_DIM=VALUE_DIM,
                N_BLOCK=BLOCK_N,
                HAS_MASK=HAS_MASK,
                IS_TAIL=False,
                PRE_LOAD_V=PRE_LOAD_V,
            )

    if HAS_TAIL:
        accumulator, row_max, row_sum = _cross_attention_tile(
            accumulator,
            row_max,
            row_sum,
            query,
            K,
            V,
            key_base_offset,
            value_base_offset,
            AttnMask,
            mask_row_offset,
            stride_k_seq,
            stride_k_dim,
            stride_v_seq,
            stride_v_dim,
            stride_mask_key,
            scale,
            query_offsets,
            qk_offsets,
            value_offsets,
            MAIN_KEY_LEN,
            QUERY_LEN=QUERY_LEN,
            KEY_LEN=KEY_LEN,
            QK_DIM=QK_DIM,
            VALUE_DIM=VALUE_DIM,
            N_BLOCK=TAIL_N,
            HAS_MASK=HAS_MASK,
            IS_TAIL=True,
            PRE_LOAD_V=PRE_LOAD_V,
        )

    return accumulator, row_max, row_sum


@libentry()
@triton.jit
def cross_attention_fwd_kernel(
    Q,
    K,
    V,
    AttnMask,
    Out,
    scale,
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_v_batch,
    stride_v_head,
    stride_v_seq,
    stride_v_dim,
    stride_mask_batch,
    stride_mask_head,
    stride_mask_query,
    stride_mask_key,
    stride_o_batch,
    stride_o_head,
    stride_o_seq,
    stride_o_dim,
    QUERY_HEADS: tl.constexpr,
    KV_HEADS: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    KEY_LEN: tl.constexpr,
    QK_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    QK_BLOCK: tl.constexpr,
    VALUE_BLOCK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAIN_KEY_LEN: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    TAIL_N: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
    PRE_LOAD_V: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)

    query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    qk_offsets = tl.arange(0, QK_BLOCK)
    value_offsets = tl.arange(0, VALUE_BLOCK)
    valid_queries = query_offsets < QUERY_LEN

    query_base_offset = batch * stride_q_batch + query_head * stride_q_head
    key_base_offset = batch * stride_k_batch + kv_head * stride_k_head
    value_base_offset = batch * stride_v_batch + kv_head * stride_v_head
    output_base_offset = batch * stride_o_batch + query_head * stride_o_head

    query_ptrs = (
        Q
        + query_base_offset
        + query_offsets[:, None] * stride_q_seq
        + qk_offsets[None, :] * stride_q_dim
    )
    if QUERY_LEN % BLOCK_M == 0:
        query = tl.load(
            query_ptrs,
            mask=qk_offsets[None, :] < QK_DIM,
            other=0.0,
        )
    else:
        query = tl.load(
            query_ptrs,
            mask=valid_queries[:, None] & (qk_offsets[None, :] < QK_DIM),
            other=0.0,
        )

    mask_row_offset = 0
    if HAS_MASK:
        mask_row_offset = (
            batch * stride_mask_batch
            + query_head * stride_mask_head
            + query_offsets[:, None] * stride_mask_query
        )

    if KEY_LEN == BLOCK_N:
        output = _cross_attention_single_tile(
            query,
            K,
            V,
            key_base_offset,
            value_base_offset,
            AttnMask,
            mask_row_offset,
            stride_k_seq,
            stride_k_dim,
            stride_v_seq,
            stride_v_dim,
            stride_mask_key,
            scale,
            query_offsets,
            qk_offsets,
            value_offsets,
            QUERY_LEN=QUERY_LEN,
            KEY_LEN=KEY_LEN,
            QK_DIM=QK_DIM,
            VALUE_DIM=VALUE_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_MASK=HAS_MASK,
        )
    else:
        accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK), dtype=tl.float32)
        row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if HAS_TAIL:
            accumulator, row_max, row_sum = _cross_attention_inner(
                accumulator,
                row_max,
                row_sum,
                query,
                K,
                V,
                key_base_offset,
                value_base_offset,
                AttnMask,
                mask_row_offset,
                stride_k_seq,
                stride_k_dim,
                stride_v_seq,
                stride_v_dim,
                stride_mask_key,
                scale,
                query_offsets,
                qk_offsets,
                value_offsets,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                QK_DIM=QK_DIM,
                VALUE_DIM=VALUE_DIM,
                QK_BLOCK=QK_BLOCK,
                VALUE_BLOCK=VALUE_BLOCK,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                MAIN_KEY_LEN=MAIN_KEY_LEN,
                HAS_TAIL=HAS_TAIL,
                TAIL_N=TAIL_N,
                PIPELINE_STAGES=PIPELINE_STAGES,
                HAS_MASK=HAS_MASK,
                PRE_LOAD_V=PRE_LOAD_V,
            )
        else:
            accumulator, row_max, row_sum = _cross_attention_inner_regular(
                accumulator,
                row_max,
                row_sum,
                query,
                K,
                V,
                key_base_offset,
                value_base_offset,
                AttnMask,
                mask_row_offset,
                stride_k_seq,
                stride_k_dim,
                stride_v_seq,
                stride_v_dim,
                stride_mask_key,
                scale,
                query_offsets,
                qk_offsets,
                value_offsets,
                QUERY_LEN=QUERY_LEN,
                KEY_LEN=KEY_LEN,
                QK_DIM=QK_DIM,
                VALUE_DIM=VALUE_DIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                HAS_MASK=HAS_MASK,
            )
        output = accumulator * tl.where(
            row_sum > 0.0, 1.0 / row_sum, 0.0
        )[:, None]
    output_ptrs = (
        Out
        + output_base_offset
        + query_offsets[:, None] * stride_o_seq
        + value_offsets[None, :] * stride_o_dim
    )
    if QUERY_LEN % BLOCK_M == 0:
        tl.store(
            output_ptrs,
            output,
            mask=value_offsets[None, :] < VALUE_DIM,
        )
    else:
        tl.store(
            output_ptrs,
            output,
            mask=valid_queries[:, None] & (value_offsets[None, :] < VALUE_DIM),
        )


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute FlashAttentionScore-compatible forward cross attention.

    Inputs use BNSD layout. Q and K have the same head dimension; V may have a
    smaller head dimension. MHA, GQA, and MQA are supported. A bool/uint8 mask
    uses non-zero values for blocked positions and must have one of the exact
    FlashAttentionScore mask shapes. Fully masked rows are returned as zeros.
    """
    logger.debug("GEMS CROSS ATTENTION")
    _validate_inputs(query, key, value, attn_mask, scale)

    batch, query_heads, query_len, qk_dim = query.shape
    kv_heads, key_len, value_dim = key.shape[1], key.shape[2], value.shape[3]
    softmax_scale = 1.0 / math.sqrt(qk_dim) if scale is None else float(scale)

    output = torch.empty(
        (batch, query_heads, query_len, value_dim),
        dtype=query.dtype,
        device=query.device,
    )
    if attn_mask is None:
        mask_arg = query
        mask_strides = (0, 0, 0, 0)
    else:
        mask_arg = attn_mask
        mask_strides = _mask_strides(attn_mask)

    qk_block = max(16, triton.next_power_of_2(qk_dim))
    value_block = max(16, triton.next_power_of_2(value_dim))
    block_m, block_n = _select_910b_tiling(
        query_len,
        key_len,
        qk_dim,
        value_dim,
        has_mask=attn_mask is not None,
        low_precision=query.dtype in (torch.float16, torch.bfloat16),
    )
    main_key_len = key_len // block_n * block_n
    tail_len = key_len - main_key_len
    tail_n = max(16, triton.next_power_of_2(tail_len)) if tail_len else 16
    # A single K/V tile has no next iteration to overlap, so double buffering
    # only increases local-memory pressure regardless of the concrete shape.
    single_key_tile = key_len == block_n
    use_multibuffer = not single_key_tile
    # Ascend TLE currently accepts at most two stages on tl.range loops.
    pipeline_stages = 2 if use_multibuffer else 1
    grid = lambda meta: (
        triton.cdiv(query_len, block_m),
        batch * query_heads,
    )

    with torch_device_fn.device(query.device):
        cross_attention_fwd_kernel[grid](
            query,
            key,
            value,
            mask_arg,
            output,
            softmax_scale,
            *query.stride(),
            *key.stride(),
            *value.stride(),
            *mask_strides,
            *output.stride(),
            QUERY_HEADS=query_heads,
            KV_HEADS=kv_heads,
            QUERY_LEN=query_len,
            KEY_LEN=key_len,
            QK_DIM=qk_dim,
            VALUE_DIM=value_dim,
            QK_BLOCK=qk_block,
            VALUE_BLOCK=value_block,
            HAS_MASK=attn_mask is not None,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            MAIN_KEY_LEN=main_key_len,
            HAS_TAIL=tail_len != 0,
            TAIL_N=tail_n,
            PIPELINE_STAGES=pipeline_stages,
            PRE_LOAD_V=False,
            num_stages=pipeline_stages,
            num_warps=4 if block_n >= 512 else 8,
            multibuffer=use_multibuffer,
            limit_auto_multi_buffer_only_for_local_buffer=False,
            limit_auto_multi_buffer_of_local_buffer="no-limit",
        )
    return output


__all__ = ["cross_attention"]
