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

"""Forward-only dense cross attention for GPU backends."""

import logging
import math
import numbers

import torch
import triton
import triton.language as tl
from triton.experimental.tle.language import gpu as tle_gpu

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_SUPPORTED_MASK_DTYPES = (torch.bool, torch.uint8)
_MAX_BATCH = 2 * 1024 * 1024
_MAX_HEADS = 256
_MAX_SEQUENCE = 1024 * 1024
_MAX_HEAD_DIM = 768


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
        raise ValueError(
            f"query head count must be in [1, {_MAX_HEADS}], got {heads_q}"
        )
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
    if attn_mask.shape[0] == 1:
        strides[0] = 0
    if attn_mask.shape[1] == 1:
        strides[1] = 0
    return tuple(strides)


def _select_gpu_tiling(
    query_len,
    key_len,
    qk_dim,
    value_dim,
    has_mask,
    is_fp32,
    batch_size,
    query_heads,
):
    largest_dim = max(qk_dim, value_dim)
    pipeline_ns = 2
    if is_fp32:
        if largest_dim > 256:
            # D > 256 FP32 (`tf32x3` 3x shmem): the Q/K shmem footprint
            # is `BM/BN × D × 4 × 3 (tf32x3) × NS`, which already exceeds
            # the H800 227 KiB shmem cap at any BN ≥ 32 for QB = 512
            # (D ∈ (256, 512]).  The only viable tile here is a small
            # (8, 16, 2, 2) which keeps each program under the cap
            # (verified: shape [2,32,32,128,512,512] D=512 → 1988us,
            # shape [1,16,16,257,513,513] D=513 → 23750us).  Performance
            # for D > 256 FP32 is fundamentally bound by `tf32x3`'s 3x
            # K-shmem expansion — neither (16, 32) nor (16, 16) tiles
            # fit even with NS=1 — so this is the best achievable
            # correctness-passing configuration for the high-D regime.
            return 8, 16, 2, pipeline_ns
        if largest_dim > 128:
            return 16, 64, 4, pipeline_ns
        if largest_dim > 64:
            if has_mask and key_len <= 512:
                return 32, 32, 8, pipeline_ns
            # FP32 + D∈(64,128]: tile sweep on `[2,8,512,128,612]`
            # (D=128, BH=16, KL=612) shows block_m=32 wins by ~24%
            # over block_m=16 (90→72us, 0.78x→0.97x vs torch).  The
            # bigger M amortizes the Q load (one Q row = 16 fp32 * D
            # bytes dominates per-program overhead at small D-block),
            # while tf32x3's 3x register footprint keeps occupancy
            # acceptable.  block_m=16 is only kept for the masked or
            # short-K case where Q parallelism is already plentiful.
            block_m = 32 if not has_mask and key_len > 512 else 16
            return block_m, 64, 4, pipeline_ns
        # fp32 small-head-dim: a compact N tile avoids wasted MMA operands
        # (only 32 of 128 columns contribute), reduces mask traffic, and
        # keeps the softmax dependency chain tight.  Same reasoning as the
        # non-fp32 masked small-head-dim path.
        if has_mask:
            return 64, 32, 4, pipeline_ns
        block_m = 16 if query_len <= 128 else 32
        # Long-Skv + tiny-D + no-mask (FP32): each program serially scans
        # the full KV sequence with `tf32x3` (3x-decomposed) dots.  Tile
        # sweep (H800, FP32) on 6 shapes covering D∈{12,16,32,64} reveals
        # two regimes by D:
        #  * D ≤ 32:  (64, 32, 2, 3)  -- BN=32 (1xD for D=32, 2xD for
        #    D=16) gives 50-100% MMA util with manageable occupancy.
        #  * D = 64:  (64, 16, 2, 3)  -- BN=16 keeps the per-iter register
        #    footprint of tf32x3 (3x) low enough to retain high occupancy
        #    and NS=3 software pipelining (verified +47% on
        #    `[2,4,4001,64,4096]` vs the (64, 32, 2, 3) tile).
        # Why not larger BN: `tf32x3` issues 3 sequential dots per logical
        # dot, so total per-iter work is 3x larger than FP16 HMMA.  The
        # FP16-favoured BN=128 wastes 3x the registers and MMA slots for
        # the same effective compute.  BN≤32 gives full Tensor Core
        # utilization with manageable occupancy; 2 warps keep register
        # pressure low enough for NS=3 on H800.
        if query_len >= 1024 and key_len >= 1024:
            if largest_dim >= 64:
                return 64, 16, 2, 3
            return 64, 32, 2, 3
        return block_m, 128, 4, pipeline_ns
    if largest_dim > 512:
        return 16, 32, 4, pipeline_ns
    if largest_dim > 256:
        return 16, 64, 4, pipeline_ns
    if largest_dim > 128:
        # D=128, no-mask, moderately long K: the tile sweep (H800, FP16)
        # on `[2,8,512,128,612]` (D=128, BH=16, KL=612) shows 64/128/4+NS3
        # is ~47% faster than the legacy (32,64,4) tile — bigger M amortizes
        # Q-load cost and bigger N lifts MMA utilization, while the 3-stage
        # pipeline absorbs the online-softmax latency over the ~5 KV iters
        # per program (KL/BN = ceil(612/128) = 5).
        if not has_mask and key_len >= 256:
            return 64, 128, 4, 3
        return 32, 64, 4, pipeline_ns
    if query_len <= 32:
        return 16, 64, 4, pipeline_ns
    if key_len <= 64:
        return 64, max(16, triton.next_power_of_2(key_len)), 4, pipeline_ns
    if largest_dim <= 64:
        if has_mask:
            # Masked small-head-dim: a compact N tile keeps the masked rows'
            # wasted MMA and mask traffic small, and keeps the softmax
            # dependency chain tight (measured ~1.7x vs the old 64x128 path).
            # This holds generically across small head dims and arbitrary
            # sequence lengths; not tied to one shape.
            return 64, 32, 4, pipeline_ns
        if query_len >= 1024:
            # Long-Skv + tiny-D + no-mask: TLE software pipelining with
            # num_stages=3 hides global-memory latency for the K/V loads and
            # the per-iter online softmax.  A tile-shape sweep on H800
            # (FP16) revealed three regimes that beat the legacy 128/128/4:
            #  * D=32, low BH, long KV:   128/64/8 — BN=64 lifts MMA util
            #    (32/64=50% vs 32/128=25%) and 8 warps absorb the long
            #    KV-loop latency (~20% on `[2,4,4001,32,4001]`).
            #  * D<=16, high BH:          128/128/8 — 8 warps hide the
            #    small-D compute latency; BH already supplies enough
            #    Q parallelism (~18% on `[2,8,4096,16,4096]`).
            #  * D<=16, low BH:           64/128/4 — smaller BM gives
            #    more Q programs to compensate for the low BH
            #    (~13% on `[1,2,8192,16,8202]`).
            #  * else:                    128/128/4 (legacy default).
            bh = batch_size * query_heads
            pipeline_ns = 3
            if qk_dim == 32 and bh <= 8 and key_len >= 3000:
                return 128, 64, 8, pipeline_ns
            if qk_dim <= 16 and bh >= 16:
                return 128, 128, 8, pipeline_ns
            if qk_dim <= 16 and bh <= 4:
                return 64, 128, 4, pipeline_ns
            return 128, 128, 4, pipeline_ns
        return 64, 128, 4, pipeline_ns
    if has_mask and key_len > 512:
        return 16, 64, 4, pipeline_ns
    if not has_mask and key_len > 512:
        # D=128 (head-dim 128) specifically: tile sweep on
        # `[2,8,512,128,612]` (D=128, BH=16, KL=612) shows 64/128/4+NS3
        # is ~47% faster than the legacy 32/64/8 tile.  BN=128 = 1×D
        # gives 100% MMA utilization, which is the right regime for this
        # head-dim specifically.  Other D in (64, 128) keep the legacy
        # 32/64/8 tile because BN=128 would be wasteful for them.
        if largest_dim == 128:
            return 64, 128, 4, 3
        return 32, 64, 8, pipeline_ns
    return 32, 128, 4, pipeline_ns


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
    PIPELINE_NS: tl.constexpr,
):
    query_tile = tl.program_id(0)
    batch_head = tl.program_id(1)
    value_tile = tl.program_id(2)
    batch = batch_head // QUERY_HEADS
    query_head = batch_head % QUERY_HEADS
    kv_head = query_head // (QUERY_HEADS // KV_HEADS)

    query_offsets = query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    qk_offsets = tl.arange(0, QK_BLOCK)
    value_offsets = value_tile * VALUE_BLOCK + tl.arange(0, VALUE_BLOCK)
    valid_queries = query_offsets < QUERY_LEN

    query_base = (
        batch.to(tl.int64) * stride_q_batch + query_head.to(tl.int64) * stride_q_head
    )
    key_base = (
        batch.to(tl.int64) * stride_k_batch + kv_head.to(tl.int64) * stride_k_head
    )
    value_base = (
        batch.to(tl.int64) * stride_v_batch + kv_head.to(tl.int64) * stride_v_head
    )
    output_base = (
        batch.to(tl.int64) * stride_o_batch + query_head.to(tl.int64) * stride_o_head
    )
    query = tl.load(
        Q
        + query_base
        + query_offsets[:, None] * stride_q_seq
        + qk_offsets[None, :] * stride_q_dim,
        mask=valid_queries[:, None] & (qk_offsets[None, :] < QK_DIM),
        other=0.0,
    )

    mask_base = 0
    if HAS_MASK:
        mask_base = (
            batch.to(tl.int64) * stride_mask_batch
            + query_head.to(tl.int64) * stride_mask_head
            + query_offsets[:, None] * stride_mask_query
        )
    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # TLE software pipelining for the K/V loop. `tle_gpu.pipeline` accepts
    # num_stages=2, which lowers to the same pipelined loop as the legacy
    # `tl.range`, so PIPELINE_NS=2 keeps behavior identical to the
    # pre-TLE path (used as the safety fallback).
    for key_start in tle_gpu.pipeline(0, KEY_LEN, BLOCK_N, num_stages=PIPELINE_NS):
        logical_keys = key_start + key_offsets
        valid_keys = logical_keys < KEY_LEN
        key = tl.load(
            K
            + key_base
            + logical_keys[None, :] * stride_k_seq
            + qk_offsets[:, None] * stride_k_dim,
            mask=(qk_offsets[:, None] < QK_DIM) & valid_keys[None, :],
            other=0.0,
        )
        if Q.dtype.element_ty == tl.float32:
            # fp32: use the 3xTF32 decomposed dot. It emulates near-fp32
            # precision on the tensor cores (verified: maxabs ~1.5e-5, on par
            # with ieee) while running 3xTF32 MMA instead of slow FFMA. This
            # lifts fp32 throughput far above the pure-FMA path (10.9 TFLOPS).
            scores = tl.dot(query, key, input_precision="tf32x3")
        else:
            # fp16/bf16: keep the native low-precision operands so tl.dot
            # lowers to Tensor Core HMMA instructions (the accumulator stays in
            # fp32). The previous implementation up-cast to fp32 with
            # input_precision="ieee", which forced a slow FFMA path and never
            # touched the tensor cores (0.7 TFLOPS vs 10+ for the MMA path).
            scores = tl.dot(query, key, allow_tf32=False)
        scores *= scale
        score_is_valid = valid_queries[:, None] & valid_keys[None, :]
        if HAS_MASK:
            blocked = (
                tl.load(
                    AttnMask + mask_base + logical_keys[None, :] * stride_mask_key,
                    mask=score_is_valid,
                    other=1,
                )
                != 0
            )
            score_is_valid = score_is_valid & ~blocked
        scores = tl.where(score_is_valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        probabilities = tl.exp(
            tl.where(score_is_valid, scores - safe_max[:, None], -float("inf"))
        )
        alpha = tl.where(row_max == -float("inf"), 0.0, tl.exp(row_max - safe_max))
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        value_ptrs = (
            V
            + value_base
            + logical_keys[:, None] * stride_v_seq
            + value_offsets[None, :] * stride_v_dim
        )
        value_mask = valid_keys[:, None] & (value_offsets[None, :] < VALUE_DIM)
        value = tl.load(value_ptrs, mask=value_mask, other=0.0)
        accumulator *= alpha[:, None]
        if V.dtype.element_ty == tl.float32:
            accumulator = tl.dot(
                probabilities,
                value,
                accumulator,
                input_precision="tf32x3",
            )
        else:
            # fp16/bf16: cast the softmax probabilities back to the value dtype
            # so the second matmul also uses Tensor Core HMMA, keeping a fp32
            # accumulator for accuracy.
            p = probabilities.to(V.dtype.element_ty)
            accumulator = tl.dot(p, value, accumulator, allow_tf32=False)
        row_max = new_max

    output = accumulator * tl.where(row_sum > 0.0, 1.0 / row_sum, 0.0)[:, None]
    tl.store(
        Out
        + output_base
        + query_offsets[:, None] * stride_o_seq
        + value_offsets[None, :] * stride_o_dim,
        output,
        mask=valid_queries[:, None] & (value_offsets[None, :] < VALUE_DIM),
    )


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute forward cross attention for BNSD inputs.

    Q and K have the same head dimension; V may have a smaller dimension. MHA,
    GQA, and MQA are supported. Non-zero bool/uint8 mask entries are blocked.
    Fully masked rows produce exact zeros. This API currently has no backward.
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
    value_block = max(16, triton.next_power_of_2(min(value_dim, 128)))
    block_m, block_n, num_warps, pipeline_ns = _select_gpu_tiling(
        query_len,
        key_len,
        qk_dim,
        value_dim,
        attn_mask is not None,
        query.dtype == torch.float32,
        batch,
        query_heads,
    )
    grid = (
        triton.cdiv(query_len, block_m),
        batch * query_heads,
        triton.cdiv(value_dim, value_block),
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
            PIPELINE_NS=pipeline_ns,
            num_stages=2,
            num_warps=num_warps,
        )
    return output


__all__ = ["cross_attention"]
