import torch
import triton
import triton.language as tl


@triton.jit
def fused_qnorm_rope_kv_insert_kernel(
    q,
    kv,
    k_cache,
    slot_mapping,
    position_ids,
    cos_sin_cache,
    eps,
    cache_block_size: tl.constexpr,
    num_tokens: tl.constexpr,
    num_heads: tl.constexpr,
    kv_block_stride,
    num_tokens_insert: tl.constexpr,
):
    HEAD_DIM: tl.constexpr = 512
    NOPE_DIM: tl.constexpr = 448
    ROPE_DIM: tl.constexpr = 64
    HALF_ROPE_DIM: tl.constexpr = 32
    QUANT_BLOCK: tl.constexpr = 64
    NUM_QUANT_BLOCKS: tl.constexpr = NOPE_DIM // QUANT_BLOCK  # 7
    SCALE_BYTES_PER_TOKEN: tl.constexpr = NUM_QUANT_BLOCKS + 1  # 8 (7 real + 1 pad)
    TOKEN_DATA_BYTES: tl.constexpr = NOPE_DIM + 2 * ROPE_DIM  # 576
    FP8_MAX: tl.constexpr = 448.0

    pid = tl.program_id(0).to(tl.int64)  # grid = (num_tokens * (num_heads + 1),)
    blocks_per_token = num_heads + 1
    token_idx = pid // blocks_per_token
    if token_idx >= num_tokens:
        return
    slot_idx = pid % blocks_per_token
    is_kv = slot_idx == num_heads
    if is_kv and token_idx >= num_tokens_insert:  # no need to insert
        return
    q_base = q + (token_idx * num_heads + slot_idx) * HEAD_DIM
    kv_base = kv + token_idx * HEAD_DIM
    offset = tl.arange(0, HEAD_DIM)
    mask_nope = offset < NOPE_DIM
    offset_rope = tl.arange(0, ROPE_DIM)
    offset_half_rope = tl.arange(0, HALF_ROPE_DIM)
    offset_quant = tl.arange(0, QUANT_BLOCK)
    if not is_kv:
        # load q
        q_blk = tl.load(q_base + offset).to(tl.float32)  # [NOPE_DIM]
        q_blk_rope = tl.load(q_base + NOPE_DIM + offset_rope).to(
            tl.float32
        )  # [ROPE_DIM]
        # RMSNorm with no weight
        variance = tl.sum(q_blk * q_blk) / HEAD_DIM
        rsqrt = tl.rsqrt(variance + eps)
        q_blk = q_blk * rsqrt
        # store q nope
        tl.store(q_base + offset, q_blk.to(tl.bfloat16), mask=mask_nope)  # [NOPE_DIM]
        qkv_blk_rope = q_blk_rope * rsqrt
    else:
        # load kv rope
        qkv_blk_rope = tl.load(kv_base + NOPE_DIM + offset_rope).to(
            tl.float32
        )  # [ROPE_DIM]
    # load cos/sin
    position_id = tl.load(position_ids + token_idx)  # i64
    cs_base = cos_sin_cache + position_id * ROPE_DIM
    cos_blk = tl.load(cs_base + offset_half_rope)  # [HALF_ROPE_DIM], f32
    sin_blk = tl.load(
        cs_base + offset_half_rope + HALF_ROPE_DIM
    )  # [HALF_ROPE_DIM], f32
    # ROPE
    qkv_blk_rope = tl.reshape(qkv_blk_rope, HALF_ROPE_DIM, 2)
    even_blk, odd_blk = tl.split(qkv_blk_rope)  # [HALF_ROPE_DIM], f32
    new_even_blk = even_blk * cos_blk - odd_blk * sin_blk
    new_odd_blk = even_blk * sin_blk + odd_blk * cos_blk
    qkv_blk_rope = tl.reshape(tl.join(new_even_blk, new_odd_blk), ROPE_DIM).to(
        tl.bfloat16
    )
    if not is_kv:
        # store q rope
        tl.store(q_base + NOPE_DIM + offset_rope, qkv_blk_rope)  # [ROPE_DIM]
        return
    # load slot
    slot_id = tl.load(slot_mapping + token_idx)  # i64
    if slot_id < 0:
        return
    block_idx = slot_id // cache_block_size
    pos_in_block = slot_id % cache_block_size
    block_base = k_cache + block_idx * kv_block_stride
    token_fp8_ptr = block_base + pos_in_block * TOKEN_DATA_BYTES
    token_bf16_ptr = token_fp8_ptr + NOPE_DIM
    token_bf16_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    token_scale_ptr = (
        block_base
        + cache_block_size * TOKEN_DATA_BYTES
        + pos_in_block * SCALE_BYTES_PER_TOKEN
    )
    # store kv rope
    tl.store(token_bf16_ptr + offset_rope, qkv_blk_rope)  # [ROPE_DIM]
    # quantization of kv nope
    # unroll the quantization loop and co-issue loads for better performance
    kv_quant_blk0 = tl.load(kv_base + offset_quant)
    kv_quant_blk1 = tl.load(kv_base + QUANT_BLOCK + offset_quant)
    kv_quant_blk2 = tl.load(kv_base + 2 * QUANT_BLOCK + offset_quant)
    kv_quant_blk3 = tl.load(kv_base + 3 * QUANT_BLOCK + offset_quant)
    kv_quant_blk4 = tl.load(kv_base + 4 * QUANT_BLOCK + offset_quant)
    kv_quant_blk5 = tl.load(kv_base + 5 * QUANT_BLOCK + offset_quant)
    kv_quant_blk6 = tl.load(kv_base + 6 * QUANT_BLOCK + offset_quant)
    # quant group 0
    qblock_idx = 0
    kv_quant_blk = kv_quant_blk0.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # quant group 1
    qblock_idx = 1
    kv_quant_blk = kv_quant_blk1.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # quant group 2
    qblock_idx = 2
    kv_quant_blk = kv_quant_blk2.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # quant group 3
    qblock_idx = 3
    kv_quant_blk = kv_quant_blk3.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # quant group 4
    qblock_idx = 4
    kv_quant_blk = kv_quant_blk4.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # quant group 5
    qblock_idx = 5
    kv_quant_blk = kv_quant_blk5.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # quant group 6
    qblock_idx = 6
    kv_quant_blk = kv_quant_blk6.to(tl.float32)
    block_max = tl.max(tl.abs(kv_quant_blk), axis=0)
    block_max = tl.maximum(block_max, 1e-4)  # match CUDA: fmaxf(amax, 1e-4)
    # scale = 2^ceil(log2(block_max / FP8_MAX))
    raw_scale = block_max / FP8_MAX
    log_scale = tl.log2(raw_scale)
    exponent = tl.ceil(log_scale)
    scale = tl.exp2(exponent)
    # quantize to fp8: fp8_value = bf16_value / scale
    x_scaled = kv_quant_blk / scale
    x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
    # convert to fp8, then bitcast to uint8 for storage
    x_fp8 = x_clamped.to(tl.float8e4nv)
    x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
    # store quantized data
    tl.store(token_fp8_ptr + qblock_idx * QUANT_BLOCK + offset_quant, x_uint8)
    # store scale: stored_value = exponent + 127 (bias)
    encoded_scale = exponent + 127.0
    encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
    tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # padding of scale
    tl.store(token_scale_ptr + NUM_QUANT_BLOCKS, tl.zeros((), dtype=tl.uint8))


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    position_ids: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    cache_block_size: int,
):
    """
    Horizontally-fused DeepseekV4-MLA: per-head RMSNorm + GPT-J RoPE for Q, and
    GPT-J RoPE + UE8M0 FP8 quant + paged cache insert for KV, all in one kernel
    launch.
    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576

    Args:
        q: [num_tokens, num_heads, 512], bfloat16, in place
        kv: [num_tokens, 512], bfloat16, read-only
        k_cache: [num_blocks, block_bytes], uint8
        slot_mapping: [num_tokens_insert], i64
        position_ids: [num_tokens], i64
        cos_sin_cache: [max_pos, 64], fp32
        eps: used in RMSNorm
        cache_block_size: tokens per paged-cache block
    """
    assert q.is_contiguous() and kv.is_contiguous()
    num_tokens, num_heads, head_dims = q.shape
    assert head_dims == 512
    assert kv.shape == (num_tokens, 512)
    assert q.dtype == torch.bfloat16 and kv.dtype == torch.bfloat16
    assert k_cache.dtype == torch.uint8
    assert slot_mapping.dim() == 1
    num_tokens_insert = slot_mapping.shape[0]
    assert num_tokens_insert <= num_tokens
    assert slot_mapping.dtype == torch.int64
    assert position_ids.shape == (num_tokens,)
    assert position_ids.dtype == torch.int64
    assert cos_sin_cache.dim() == 2 and cos_sin_cache.shape[1] == 64
    assert cos_sin_cache.dtype == torch.float32

    grid = num_tokens * (num_heads + 1)
    fused_qnorm_rope_kv_insert_kernel[(grid,)](
        q,
        kv,
        k_cache,
        slot_mapping,
        position_ids,
        cos_sin_cache,
        eps,
        cache_block_size,
        num_tokens,
        num_heads,
        k_cache.stride(0),
        num_tokens_insert,
        num_warps=1,
        num_stages=2,
    )
