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

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils.triton_version_utils import HAS_TLE

if HAS_TLE:
    from triton.experimental.tle.language import gpu as tle_gpu
else:
    tle_gpu = None


def _tle_pipeline_supported():
    """Whether the active backend whitelists the TLE ``gpu.pipeline`` loop.

    ``HAS_TLE`` is only a vendor switch; each backend keeps a stricter
    primitive whitelist, and the NVIDIA/AMD ones list ``gpu.range`` but not
    ``gpu.pipeline``, which would fail at compile time.
    """
    if not HAS_TLE:
        return False
    try:
        from triton.experimental.tle import is_primitive_supported
        from triton.runtime import driver as triton_driver

        try:
            # ``target.backend`` reports the driver name (``cuda``), which is
            # not a TLE registry key, so prefer FlagTree's backend name.
            from triton._flagtree_backend import get_active_backend_name

            backend = get_active_backend_name()
        except Exception:
            backend = triton_driver.active.get_current_target().backend
        return bool(is_primitive_supported(backend, "gpu.pipeline"))
    except Exception:
        return False


# `tle_gpu.pipeline` and `tl.range` share `num_stages` semantics, so the K/V
# loop falls back to plain `tl.range` where the primitive is unsupported.
_KEY_LOOP = tle_gpu.pipeline if _tle_pipeline_supported() else tl.range

logger = logging.getLogger(__name__)
_LOG2E = tl.constexpr(1.4426950408889634)
# fp32 runs as three bf16 MMAs over a hi/lo split (~16 mantissa bits): twice
# the tf32 throughput at half the operand footprint. False falls back to
# `input_precision="tf32x3"`.
_FP32_BF16_SPLIT = True
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
    if not (query.device == key.device == value.device):
        raise ValueError("query, key, and value must be on the same device")
    if not (query.dtype == key.dtype == value.dtype):
        raise TypeError("query, key, and value must have the same dtype")
    if query.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "cross_attention supports only torch.float16, torch.bfloat16, "
            f"and torch.float32, got {query.dtype}"
        )

    batch_q, heads_q, query_len, dim_q = query.shape
    batch_k, heads_k, key_len, dim_k = key.shape
    batch_v, heads_v, value_len, dim_v = value.shape
    for name, size, upper in (
        ("batch size", batch_q, _MAX_BATCH),
        ("query head count", heads_q, _MAX_HEADS),
        ("query sequence length", query_len, _MAX_SEQUENCE),
        ("key sequence length", key_len, _MAX_SEQUENCE),
    ):
        if not 1 <= size <= upper:
            raise ValueError(f"{name} must be in [1, {upper}], got {size}")
    if batch_q != batch_k or batch_q != batch_v:
        raise ValueError("query, key, and value batch sizes must match")
    if heads_k != heads_v or heads_k <= 0 or heads_q % heads_k != 0:
        raise ValueError(
            "key/value head counts must match and query_heads / kv_heads must "
            "be a non-zero integer"
        )
    if key_len != value_len:
        raise ValueError("key and value sequence lengths must match")
    if dim_q != dim_k or dim_k < dim_v:
        raise ValueError("head dimensions must satisfy query_D == key_D >= value_D")
    if not (1 <= dim_q <= _MAX_HEAD_DIM and 1 <= dim_v <= _MAX_HEAD_DIM):
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
        allowed = {
            (query_len, key_len),
            (1, 1, query_len, key_len),
            (batch_q, 1, query_len, key_len),
            (batch_q, heads_q, query_len, key_len),
        }
        if tuple(attn_mask.shape) not in allowed:
            shapes = ", ".join(str(shape) for shape in sorted(allowed))
            raise ValueError(
                f"attn_mask shape must be one of {shapes}, got {tuple(attn_mask.shape)}"
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
    """Pick (BLOCK_M, BLOCK_N, num_warps, pipeline_stages) for a launch.

    Populated from tile sweeps on the core shape set; every regime below was
    measured against `F.scaled_dot_product_attention`.
    """
    d = max(qk_dim, value_dim)
    bh = batch_size * query_heads
    ns = 2

    if is_fp32:
        # fp32 goes through the bf16 hi/lo split (3 MMAs), so its operand
        # footprint is 2x smaller than `tf32x3`'s and the old tf32-calibrated
        # tiles no longer apply.
        if d > 256:
            return 8, 16, 2, ns
        if d > 128:
            return 16, 64, 4, ns
        if d > 64:
            if key_len > 2048:
                return 32, 64, 4, 3
            if bh < 16 and key_len <= 1024:
                return (32, 32, 4, ns) if has_mask else (16, 32, 4, ns)
            return 64, 32, 4, ns
        # Small head dim: compact N tiles, either way wide enough for MMA.
        if has_mask:
            return 64, 32, 4, ns
        if query_len >= 1024 and key_len >= 1024:
            return (64, 16, 2, 3) if d >= 64 else (64, 32, 2, 3)
        return (16 if query_len <= 128 else 32), 128, 4, ns

    if d > 512:
        return 16, 32, 4, ns
    if d > 256:
        return 16, 64, 4, ns
    if d > 128:
        if not has_mask and key_len >= 256:
            return 64, 128, 4, 3
        return 32, 64, 4, ns
    if query_len <= 32:
        return 16, 64, 4, ns
    if key_len <= 64:
        return 64, max(16, triton.next_power_of_2(key_len)), 4, ns
    if d == 128:
        # Short KV is scanned in a single pass (BLOCK_N spans the sequence, so
        # there is no online-softmax carry). Longer KV wants a wide M tile with
        # a 3-stage pipeline, but only when the launch still fills the device.
        if key_len <= 256:
            return (64, 256, 4, 1) if has_mask else (128, 256, 8, 1)
        if bh * triton.cdiv(query_len, 128) >= 64:
            return 128, 64, 8, 3
        return (32, 64, 4, 3) if has_mask else (64, 128, 4, 2)
    if d <= 64:
        if has_mask:
            return 64, 32, 4, ns
        if query_len >= 1024:
            if qk_dim == 32 and bh <= 8 and key_len >= 3000:
                return 128, 64, 8, 3
            if qk_dim <= 16 and bh >= 16:
                return 128, 128, 8, 3
            if qk_dim <= 16 and bh <= 4:
                return 64, 128, 4, 3
            return 128, 128, 4, 3
        return 64, 128, 4, ns
    if key_len > 512:
        return (16, 64, 4, ns) if has_mask else (32, 64, 8, ns)
    return 32, 128, 4, ns


@triton.jit
def _bf16_split(x):
    """Split ``x`` into hi/lo bf16 halves so x ~= hi + lo (~16 mantissa bits)."""
    hi = x.to(tl.bfloat16)
    lo = (x - hi.to(tl.float32)).to(tl.bfloat16)
    return hi, lo


@triton.jit
def _tile_load(ptr, mask, EXACT: tl.constexpr):
    """Predicated load that drops the predicate when the tile is in bounds."""
    if EXACT:
        return tl.load(ptr)
    return tl.load(ptr, mask=mask, other=0.0)


@triton.jit
def _tile_store(ptr, value, mask, EXACT: tl.constexpr):
    if EXACT:
        tl.store(ptr, value)
    else:
        tl.store(ptr, value, mask=mask)


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
    EXACT_BOUNDS: tl.constexpr,
    FP32_SPLIT: tl.constexpr,
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

    # int64 offsets keep the pointer math in range for large sequences.
    batch64 = batch.to(tl.int64)
    qh64 = query_head.to(tl.int64)
    kh64 = kv_head.to(tl.int64)
    query_base = batch64 * stride_q_batch + qh64 * stride_q_head
    key_base = batch64 * stride_k_batch + kh64 * stride_k_head
    value_base = batch64 * stride_v_batch + kh64 * stride_v_head
    output_base = batch64 * stride_o_batch + qh64 * stride_o_head
    query_ptrs = (
        Q
        + query_base
        + query_offsets[:, None] * stride_q_seq
        + qk_offsets[None, :] * stride_q_dim
    )
    query = _tile_load(
        query_ptrs,
        valid_queries[:, None] & (qk_offsets[None, :] < QK_DIM),
        EXACT_BOUNDS,
    )
    if FP32_SPLIT:
        query_hi, query_lo = _bf16_split(query)

    mask_base = 0
    if HAS_MASK:
        mask_base = (
            batch64 * stride_mask_batch
            + qh64 * stride_mask_head
            + query_offsets[:, None] * stride_mask_query
        )
    accumulator = tl.zeros((BLOCK_M, VALUE_BLOCK), dtype=tl.float32)
    row_max = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # `_KEY_LOOP` is `tle_gpu.pipeline` or plain `tl.range`; both take
    # `num_stages`, so PIPELINE_NS drives the K/V software pipeline either way.
    for key_start in _KEY_LOOP(0, KEY_LEN, BLOCK_N, num_stages=PIPELINE_NS):
        logical_keys = key_start + key_offsets
        key_ptrs = (
            K
            + key_base
            + logical_keys[None, :] * stride_k_seq
            + qk_offsets[:, None] * stride_k_dim
        )
        valid_keys = logical_keys < KEY_LEN
        key = _tile_load(
            key_ptrs,
            (qk_offsets[:, None] < QK_DIM) & valid_keys[None, :],
            EXACT_BOUNDS,
        )
        if Q.dtype.element_ty == tl.float32:
            if FP32_SPLIT:
                # fp32 as three bf16 MMAs over a hi/lo split: near-fp32
                # precision at bf16 tensor-core rate.
                key_hi, key_lo = _bf16_split(key)
                scores = tl.dot(query_hi, key_hi, allow_tf32=False)
                scores = tl.dot(query_hi, key_lo, scores, allow_tf32=False)
                scores = tl.dot(query_lo, key_hi, scores, allow_tf32=False)
            else:
                scores = tl.dot(query, key, input_precision="tf32x3")
        else:
            # fp16/bf16 stay native so `tl.dot` lowers to Tensor Core HMMA.
            scores = tl.dot(query, key, allow_tf32=False)
        # log2(e) is folded into the scale so the softmax below can use exp2.
        scores *= scale * _LOG2E
        if HAS_MASK:
            mask_ptrs = AttnMask + mask_base + logical_keys[None, :] * stride_mask_key
            if EXACT_BOUNDS:
                score_is_valid = ~(tl.load(mask_ptrs) != 0)
            else:
                score_is_valid = valid_queries[:, None] & valid_keys[None, :]
                blocked = tl.load(mask_ptrs, mask=score_is_valid, other=1) != 0
                score_is_valid = score_is_valid & ~blocked
            scores = tl.where(score_is_valid, scores, -float("inf"))
        elif not EXACT_BOUNDS:
            score_is_valid = valid_queries[:, None] & valid_keys[None, :]
            scores = tl.where(score_is_valid, scores, -float("inf"))
        tile_max = tl.max(scores, axis=1)
        new_max = tl.maximum(row_max, tile_max)
        if EXACT_BOUNDS and not HAS_MASK:
            # No tail rows and no blocked lanes: the guards below are dead.
            probabilities = tl.math.exp2(scores - new_max[:, None])
            alpha = tl.math.exp2(row_max - new_max)
        elif not HAS_MASK:
            # Only tail rows (never stored) need a finite normalizer.
            safe_max = tl.where(valid_queries, new_max, 0.0)
            probabilities = tl.math.exp2(
                tl.where(score_is_valid, scores - safe_max[:, None], -float("inf"))
            )
            alpha = tl.math.exp2(row_max - safe_max)
        else:
            safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
            probabilities = tl.math.exp2(
                tl.where(score_is_valid, scores - safe_max[:, None], -float("inf"))
            )
            alpha = tl.where(
                row_max == -float("inf"), 0.0, tl.math.exp2(row_max - safe_max)
            )
        row_sum = row_sum * alpha + tl.sum(probabilities, axis=1)
        value_ptrs = (
            V
            + value_base
            + logical_keys[:, None] * stride_v_seq
            + value_offsets[None, :] * stride_v_dim
        )
        value = _tile_load(
            value_ptrs,
            valid_keys[:, None] & (value_offsets[None, :] < VALUE_DIM),
            EXACT_BOUNDS,
        )
        accumulator *= alpha[:, None]
        if V.dtype.element_ty == tl.float32:
            if FP32_SPLIT:
                # hi*hi + hi*lo + lo*hi, mirroring the QK split above.
                p_hi, p_lo = _bf16_split(probabilities)
                v_hi, v_lo = _bf16_split(value)
                partial = tl.dot(p_hi, v_hi, accumulator, allow_tf32=False)
                partial = tl.dot(p_hi, v_lo, partial, allow_tf32=False)
                accumulator = tl.dot(p_lo, v_hi, partial, allow_tf32=False)
            else:
                accumulator = tl.dot(
                    probabilities, value, accumulator, input_precision="tf32x3"
                )
        else:
            # Cast probabilities back so the PV matmul stays on Tensor Cores.
            p = probabilities.to(V.dtype.element_ty)
            accumulator = tl.dot(p, value, accumulator, allow_tf32=False)
        row_max = new_max

    output = accumulator * tl.where(row_sum > 0.0, 1.0 / row_sum, 0.0)[:, None]
    output_ptrs = (
        Out
        + output_base
        + query_offsets[:, None] * stride_o_seq
        + value_offsets[None, :] * stride_o_dim
    )
    _tile_store(
        output_ptrs,
        output,
        valid_queries[:, None] & (value_offsets[None, :] < VALUE_DIM),
        EXACT_BOUNDS,
    )


def cross_attention(query, key, value, attn_mask=None, scale=None):
    """Compute forward cross attention for BNSD inputs.

    Q and K have the same head dimension; V may have a smaller dimension. MHA,
    GQA, and MQA are supported. Non-zero bool/uint8 mask entries are blocked.
    Fully masked rows produce exact zeros. This API currently has no backward.
    """
    logger.debug("GEMS CROSS_ATTENTION")
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
    # Fully in-bounds launch: the kernel can skip every per-lane predicate,
    # including the score `where` that runs on each K/V iteration.
    exact_bounds = (
        query_len % block_m == 0
        and key_len % block_n == 0
        and qk_dim == qk_block
        and value_dim % value_block == 0
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
            EXACT_BOUNDS=exact_bounds,
            FP32_SPLIT=query.dtype == torch.float32 and _FP32_BF16_SPLIT,
            num_stages=2,
            num_warps=num_warps,
        )
    return output


__all__ = ["cross_attention"]
