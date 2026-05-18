import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


@triton.jit
def _fused_decode_kernel(
    q_ptr, kv_ptr, k_cache_ptr, slot_mapping_ptr,
    position_ids_ptr, cos_sin_cache_ptr,
    stride_q_tok, stride_q_head, stride_kv_tok,
    stride_cache_block, stride_cos_sin_pos,
    eps,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    num_tokens_insert: tl.constexpr,
    cache_block_size: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    num_head_groups: tl.constexpr = num_heads // HEADS_PER_PROG
    total_q_progs = num_tokens * num_head_groups
    total_progs = total_q_progs + num_tokens_insert

    if pid >= total_progs:
        return

    if pid < total_q_progs:
        tok_idx = pid // num_head_groups
        group_idx = pid % num_head_groups
        head_start = group_idx * HEADS_PER_PROG

        pos = tl.load(position_ids_ptr + tok_idx)
        half_offs = tl.arange(0, 32)
        cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
        sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

        offs = tl.arange(0, 512)
        rope_even = 448 + half_offs * 2
        rope_odd = rope_even + 1

        for h in tl.static_range(HEADS_PER_PROG):
            head_idx = head_start + h
            base = tok_idx * stride_q_tok + head_idx * stride_q_head

            x = tl.load(q_ptr + base + offs).to(tl.float32)
            re_orig = tl.load(q_ptr + base + rope_even)
            ro_orig = tl.load(q_ptr + base + rope_odd)

            var = tl.sum(x * x, axis=0) / 512.0
            rsqrt_val = tl.math.rsqrt(var + eps)
            x_norm = x * rsqrt_val

            nope_mask = offs < 448
            tl.store(q_ptr + base + offs, x_norm.to(tl.bfloat16), mask=nope_mask)

            re_f = (re_orig.to(tl.float32) * rsqrt_val).to(tl.bfloat16).to(tl.float32)
            ro_f = (ro_orig.to(tl.float32) * rsqrt_val).to(tl.bfloat16).to(tl.float32)

            tl.store(q_ptr + base + rope_even, (re_f * cos - ro_f * sin).to(tl.bfloat16))
            tl.store(q_ptr + base + rope_odd, (re_f * sin + ro_f * cos).to(tl.bfloat16))
    else:
        tok_idx = pid - total_q_progs
        slot_id = tl.load(slot_mapping_ptr + tok_idx)
        if slot_id < 0:
            return

        kv_base = tok_idx * stride_kv_tok
        pos = tl.load(position_ids_ptr + tok_idx)

        half_offs = tl.arange(0, 32)
        cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
        sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

        rope_even = 448 + half_offs * 2
        rope_odd = rope_even + 1
        x_e = tl.load(kv_ptr + kv_base + rope_even).to(tl.bfloat16).to(tl.float32)
        x_o = tl.load(kv_ptr + kv_base + rope_odd).to(tl.bfloat16).to(tl.float32)
        out_e = x_e * cos - x_o * sin
        out_o = x_e * sin + x_o * cos

        block_idx = slot_id // cache_block_size
        pos_in_block = slot_id % cache_block_size
        byte_off_tok = block_idx * stride_cache_block + pos_in_block * 576
        byte_off_scale = (block_idx * stride_cache_block
                          + cache_block_size * 576 + pos_in_block * 8)

        for b in tl.static_range(7):
            boffs = tl.arange(0, 64)
            bdata = tl.load(kv_ptr + kv_base + b * 64 + boffs).to(tl.bfloat16).to(tl.float32)
            absmax = tl.max(tl.abs(bdata), axis=0)
            absmax = tl.maximum(absmax, 1e-4)
            exponent = tl.math.ceil(tl.math.log2(absmax / 448.0))
            inv_scale = tl.math.exp2(-exponent)
            scaled = bdata * inv_scale
            scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
            fp8_vals = scaled.to(tl.float8e4nv)
            fp8_i8 = fp8_vals.to(tl.int8, bitcast=True)
            i8_ptr = (k_cache_ptr + byte_off_tok + b * 64).to(tl.pointer_type(tl.int8))
            tl.store(i8_ptr + boffs, fp8_i8)
            enc_scale = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0).to(tl.uint8)
            sc_ptr = (k_cache_ptr + byte_off_scale + b).to(tl.pointer_type(tl.uint8))
            tl.store(sc_ptr + tl.arange(0, 1), tl.full([1], enc_scale, dtype=tl.uint8))

        pad_ptr = (k_cache_ptr + byte_off_scale + 7).to(tl.pointer_type(tl.uint8))
        tl.store(pad_ptr + tl.arange(0, 1), tl.zeros([1], dtype=tl.uint8))

        bf16_ptr = (k_cache_ptr + byte_off_tok + 448).to(tl.pointer_type(tl.bfloat16))
        st_even = half_offs * 2
        st_odd = st_even + 1
        tl.store(bf16_ptr + st_even, out_e.to(tl.bfloat16))
        tl.store(bf16_ptr + st_odd, out_o.to(tl.bfloat16))


@triton.jit
def _q_kernel_prefill(
    q_ptr, position_ids_ptr, cos_sin_cache_ptr,
    stride_q_tok, stride_q_head, stride_cos_sin_pos,
    eps,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(position_ids_ptr + tok_idx)
    half_offs = tl.arange(0, 32)
    cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
    sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

    offs = tl.arange(0, 512)
    rope_even = 448 + half_offs * 2
    rope_odd = rope_even + 1
    base = tok_idx * stride_q_tok + head_idx * stride_q_head

    x = tl.load(q_ptr + base + offs).to(tl.float32)
    re_orig = tl.load(q_ptr + base + rope_even)
    ro_orig = tl.load(q_ptr + base + rope_odd)

    var = tl.sum(x * x, axis=0) / 512.0
    rsqrt_val = tl.math.rsqrt(var + eps)
    x_norm = x * rsqrt_val

    nope_mask = offs < 448
    tl.store(q_ptr + base + offs, x_norm.to(tl.bfloat16), mask=nope_mask)

    re_f = (re_orig.to(tl.float32) * rsqrt_val).to(tl.bfloat16).to(tl.float32)
    ro_f = (ro_orig.to(tl.float32) * rsqrt_val).to(tl.bfloat16).to(tl.float32)

    tl.store(q_ptr + base + rope_even, (re_f * cos - ro_f * sin).to(tl.bfloat16))
    tl.store(q_ptr + base + rope_odd, (re_f * sin + ro_f * cos).to(tl.bfloat16))


@triton.jit
def _kv_kernel_prefill(
    kv_ptr, k_cache_ptr, slot_mapping_ptr,
    position_ids_ptr, cos_sin_cache_ptr,
    stride_kv_tok, stride_cache_block, stride_cos_sin_pos,
    cache_block_size,
    num_tokens_insert: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    if tok_idx >= num_tokens_insert:
        return

    slot_id = tl.load(slot_mapping_ptr + tok_idx)
    if slot_id < 0:
        return

    kv_base = tok_idx * stride_kv_tok
    pos = tl.load(position_ids_ptr + tok_idx)

    half_offs = tl.arange(0, 32)
    cos = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + half_offs)
    sin = tl.load(cos_sin_cache_ptr + pos * stride_cos_sin_pos + 32 + half_offs)

    rope_even = 448 + half_offs * 2
    rope_odd = rope_even + 1
    x_e = tl.load(kv_ptr + kv_base + rope_even).to(tl.bfloat16).to(tl.float32)
    x_o = tl.load(kv_ptr + kv_base + rope_odd).to(tl.bfloat16).to(tl.float32)
    out_e = x_e * cos - x_o * sin
    out_o = x_e * sin + x_o * cos

    block_idx = slot_id // cache_block_size
    pos_in_block = slot_id % cache_block_size
    byte_off_tok = block_idx * stride_cache_block + pos_in_block * 576
    byte_off_scale = (block_idx * stride_cache_block
                      + cache_block_size * 576 + pos_in_block * 8)

    for b in tl.static_range(7):
        boffs = tl.arange(0, 64)
        bdata = tl.load(kv_ptr + kv_base + b * 64 + boffs).to(tl.bfloat16).to(tl.float32)
        absmax = tl.max(tl.abs(bdata), axis=0)
        absmax = tl.maximum(absmax, 1e-4)
        exponent = tl.math.ceil(tl.math.log2(absmax / 448.0))
        inv_scale = tl.math.exp2(-exponent)
        scaled = bdata * inv_scale
        scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
        fp8_vals = scaled.to(tl.float8e4nv)
        fp8_i8 = fp8_vals.to(tl.int8, bitcast=True)
        i8_ptr = (k_cache_ptr + byte_off_tok + b * 64).to(tl.pointer_type(tl.int8))
        tl.store(i8_ptr + boffs, fp8_i8)
        enc_scale = tl.maximum(tl.minimum(exponent + 127.0, 255.0), 0.0).to(tl.uint8)
        sc_ptr = (k_cache_ptr + byte_off_scale + b).to(tl.pointer_type(tl.uint8))
        tl.store(sc_ptr + tl.arange(0, 1), tl.full([1], enc_scale, dtype=tl.uint8))

    pad_ptr = (k_cache_ptr + byte_off_scale + 7).to(tl.pointer_type(tl.uint8))
    tl.store(pad_ptr + tl.arange(0, 1), tl.zeros([1], dtype=tl.uint8))

    bf16_ptr = (k_cache_ptr + byte_off_tok + 448).to(tl.pointer_type(tl.bfloat16))
    st_even = half_offs * 2
    st_odd = st_even + 1
    tl.store(bf16_ptr + st_even, out_e.to(tl.bfloat16))
    tl.store(bf16_ptr + st_odd, out_o.to(tl.bfloat16))


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps: float = 1e-6,
    cache_block_size: int = 16,
):
    """Fused QNorm + RoPE + KV cache insert for DeepSeek V4.

    Performs in-place:
      1. Q: RMSNorm(512d) -> store NoPE(448d) + RoPE(64d)
      2. KV: RoPE(64d) + FP8 quantize(448d) -> insert into paged k_cache

    Args:
        q: [N, H, 512] bfloat16, modified in-place
        kv: [N_ins, 576] bfloat16 (448 NoPE + 64 RoPE + 64 padding)
        k_cache: paged FP8 cache buffer
        slot_mapping: [N_ins] int32, page slot indices
        position_ids: [N] int64, token positions
        cos_sin_cache: [max_pos, 64] float32 (32 cos + 32 sin)
        eps: RMSNorm epsilon
        cache_block_size: tokens per cache block
    """
    logger.debug("GEMS FUSED QNORM ROPE KV FORWARD")
    N, H, D = q.shape
    N_ins = slot_mapping.shape[0]

    if N <= 64:
        # Decode path: fused kernel, v36 config
        if N <= 1:
            HEADS_PER_PROG = min(16, H)
        else:
            HEADS_PER_PROG = min(8, H)
        num_head_groups = H // HEADS_PER_PROG
        grid_size = N * num_head_groups + N_ins

        _fused_decode_kernel[(grid_size,)](
            q, kv, k_cache, slot_mapping,
            position_ids, cos_sin_cache,
            q.stride(0), q.stride(1), kv.stride(0),
            k_cache.stride(0), cos_sin_cache.stride(0),
            eps, N, H, N_ins, cache_block_size,
            HEADS_PER_PROG,
            num_warps=8, num_stages=4,
        )
    elif N <= 256:
        # Mid-range: fused decode, v36 config
        HEADS_PER_PROG = 2
        num_head_groups = H // HEADS_PER_PROG
        grid_size = N * num_head_groups + N_ins

        _fused_decode_kernel[(grid_size,)](
            q, kv, k_cache, slot_mapping,
            position_ids, cos_sin_cache,
            q.stride(0), q.stride(1), kv.stride(0),
            k_cache.stride(0), cos_sin_cache.stride(0),
            eps, N, H, N_ins, cache_block_size,
            HEADS_PER_PROG,
            num_warps=2, num_stages=4,
        )
    else:
        # Prefill path: split Q/KV, v48 config
        _q_kernel_prefill[(N, H)](
            q, position_ids, cos_sin_cache,
            q.stride(0), q.stride(1), cos_sin_cache.stride(0),
            eps, N, H,
            num_warps=2, num_stages=4,
        )

        if N_ins > 0:
            _kv_kernel_prefill[(N_ins,)](
                kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
                kv.stride(0), k_cache.stride(0), cos_sin_cache.stride(0),
                cache_block_size, N_ins,
                num_warps=2, num_stages=4,
            )
